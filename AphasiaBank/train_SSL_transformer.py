#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the followering:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import numpy as np
from scipy.io import wavfile
import wave
from tqdm import tqdm
import librosa
import pandas as pd
import math
import os
import sys
import torch
import logging
import speechbrain as sb
# import speechbrain.speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from datasets import load_dataset, load_metric, Audio
import re
import time
from speechbrain.tokenizers.SentencePiece import SentencePiece

import seaborn as sns
import matplotlib.pyplot as plt
import chaipy.io

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

def props(cls):   
    return [i for i in cls.__dict__.keys() if i[:1] != '_']
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

spk2aq = chaipy.io.dict_read("/z/mkperez/speechbrain/AphasiaBank/spk2aq")

def visualize_attention(attn,str_attn,ticklabels,exp_dir,word_x=True,epoch=None):
    '''
    attn = torch tensor of attention
    str_attn = utt_id
    ylabels = list of words (including eos)
    exp_dir = hparams["output_folder"]
    '''
    stacked_attn = torch.vstack(attn)
    mean_attn = torch.mean(stacked_attn, dim=0)
    plt.clf()
    if word_x:
        ax = sns.heatmap(torch.t(mean_attn).detach().cpu().numpy(), annot=False, xticklabels=ticklabels)
        ax.set(title=f"Title", xlabel=f"Words", ylabel=f"Encoder idx")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax = sns.heatmap(mean_attn.detach().cpu().numpy(), annot=False, yticklabels=ticklabels)
        ax.set(title=f"Title", xlabel=f"Encoder id", ylabel=f"Words")
    fig = ax.get_figure()
    fig.tight_layout()
    
    if str_attn in ['ACWT09a-13','adler15a-546','BU10a-508','kurland23b-574','scale05a-91']:
        # AB
        spkr = str_attn.split("-")[0]
        sev = int(float(spk2aq[spkr]))
        outdir = f"{exp_dir}/attention/AB/sev-{sev}_{str_attn}"
    else:
        # control
        outdir = f"{exp_dir}/attention/Control/{str_attn}"
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/ep-{epoch}.png")


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)


        # forward modules
        w2v_out = self.modules.SSL_enc(wavs)
        # print(f"w2v_out: {w2v_out.shape}")

        pred,dec_mha = self.modules.Transformer(
            w2v_out, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(w2v_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0 or current_epoch==1:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(w2v_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST and not self.hparams.use_language_modelling:
            hyps, _ = self.hparams.test_search(w2v_out.detach(), wav_lens)
            

        return p_ctc, p_seq, wav_lens, hyps, dec_mha

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps,dec_attn) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        # now as training progresses we use real prediction from the prev step instead of teacher forcing

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )


        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or current_epoch==1 or (
                stage == sb.Stage.TEST
            ):
                if stage == sb.Stage.TEST and self.hparams.use_language_modelling:
                    predicted_words = []
                    for logs in p_seq:
                        text = self.decoder.decode(logs.detach().cpu().numpy(), beam_width=100)
                        predicted_words.append(text.split(" "))
                else:
                    # Decode token terms to words
                    predicted_words = [
                        "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                        for utt_seq in hyps
                    ]



                # target_words = [wrd.upper().split(" ") for wrd in batch.wrd] # for libri
                target_words = [wrd.split(" ") for wrd in batch.wrd] # AB
                # print(f"predicted: {predicted_words}")
                # print(f"target_words: {target_words}")
                # exit()
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

                # visualize attention
                if self.hparams.val_attn_ids and (ids[0] in self.hparams.val_attn_ids):
                # if ids[0] in self.hparams.val_attn_ids:
                    # tick_labels = [self.tokenizer.id_to_piece(i.item()) for i in tokens_eos[0]]
                    char_decode = [self.tokenizer.decode_ndim(i.item()) for i in tokens_eos[0]]
                    add_space = []
                    for i in range(len(char_decode)-1):
                        if char_decode[i] == '<blank>' and char_decode[i+1] == '<blank>':
                            add_space.append(" ")
                        add_space.append(char_decode[i])
                    tick_labels = "".join(add_space).split(" ")

                    tick_labels = "".join([self.tokenizer.decode_ndim(i.item()) for i in tokens_eos[0]]).split(" ")
                    visualize_attention(dec_attn,f"{ids[0]}",tick_labels,self.hparams.output_folder,word_x=True,epoch=current_epoch)



            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        # print(f"LR: {self.optimizer.param_groups[-1]['lr']}")
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.acc_metric = self.hparams.acc_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # lr = self.hparams.noam_annealing.current_lr 
            # newBOB
            lr, new_lr_model = self.hparams.lr_annealing(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr_model
            )

            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                # self.hparams.noam_annealing(self.optimizer)
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                # self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    #convert severity_cat to int
    train_data.data = {k:{k_2: (int(v_2) if k_2 == 'severity_cat' else v_2) for k_2,v_2 in v.items()} for k,v in train_data.data.items()}
    # print(train_data.data)
    # exit()
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"], "severity_cat": hparams["max_sev_train"]},
            key_min_value={"duration": hparams["min_length"], "severity_cat": hparams["min_sev_train"]},

        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True,
            # key_max_value={"duration": hparams["max_length"]}
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder}
    )
    
    valid_data = valid_data.filtered_sorted(sort_key="duration", reverse=True,
        key_max_value={"duration": hparams["max_length"]},
        key_min_value={"duration": hparams["min_length"]}
    )


    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder}
    )
    test_data = test_data.filtered_sorted(sort_key="duration",reverse=True,
        key_max_value={"duration": hparams["max_length"]},
        key_min_value={"duration": hparams["min_length"]}
    )

    datasets = [train_data, valid_data, test_data]
    valtest_datasets = [valid_data,test_data]

    # tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if hparams["speed_perturb"]:
            sig = sb.dataio.dataio.read_audio(wav)
            # factor = np.random.uniform(0.95, 1.05)
            # sig = resample(sig.numpy(), 16000, int(16000*factor))
            speed = sb.processing.speech_augmentation.SpeedPerturb(
                16000, [x for x in range(95, 105)]
            )
            sig = speed(sig.unsqueeze(0)).squeeze(0)  # torch.from_numpy(sig)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    
    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    # add special labels
    special_labels = {
        "blank_label": hparams["blank_index"],
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"]
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd","char_list", "tokens_bos", "tokens_eos", "tokens"],
    )


    return (
        train_data,
        valid_data,
        test_data,
        label_encoder
    )


def prep_exp_dir(hparams):
    save_folder = hparams['save_folder']
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(f"run_opts: {run_opts}")
    # exit()

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    prep_exp_dir(hparams)


    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=1.0,
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
        pad_id=hparams["pad_index"],
        unk_id=hparams["unk_index"],
    )


    train_data,valid_data,test_data,label_encoder = dataio_prepare(hparams)

    # # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # # the path given in the YAML file). The tokenizer is loaded at the same time.
    # run_on_main(hparams["pretrainer"].collect_files)
    # hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    tokens = {i:asr_brain.tokenizer.id_to_piece(i) for i in range(asr_brain.tokenizer.get_piece_size())}
    print(f"tokenizer: {tokens} | {len(tokens.keys())}")
    # print(f"train_data: {train_data.data}")

    # exit()
    
    # asr_brain.modules = asr_brain.modules.float()
    count_parameters(asr_brain.modules)


    # Loading the labels for the LM decoding and the CTC decoder
    if "use_language_modelling" in hparams:
        if hparams["use_language_modelling"]:
            try:
                from pyctcdecode import build_ctcdecoder
            except ImportError:
                err_msg = "Optional dependencies must be installed to use pyctcdecode.\n"
                err_msg += "Install using `pip install kenlm pyctcdecode`.\n"
                raise ImportError(err_msg)

            ind2lab = label_encoder.ind2lab
            labels = [ind2lab[x] for x in range(len(ind2lab))]
            labels = [""] + labels[1:]  # Replace the <blank> token with a blank character, needed for PyCTCdecode

            # labels  = [asr_brain.tokenizer.id_to_piece(id).lower() for id in range(asr_brain.tokenizer.get_piece_size())]
            # print(f"labels: {labels}")
            # labels[10]=' ' # '_'
            # labels[0] = '<pad>' # unk


            # labels[0] = '' # unk
            # print(f"labels: {labels}")
            # exit()

            asr_brain.decoder = build_ctcdecoder(
                labels,
                kenlm_model_path=hparams[
                    "ngram_lm_path"
                ],  # either .arpa or .bin file
                alpha=0.5,  # Default by KenLM
                beta=1.0,  # Default by KenLM
            )
    else:
        hparams["use_language_modelling"] = False
    

    with torch.autograd.detect_anomaly():
        if hparams['train_flag']:
            asr_brain.fit(
                asr_brain.hparams.epoch_counter,
                train_data,
                valid_data,
                train_loader_kwargs=hparams["train_dataloader_opts"],
                valid_loader_kwargs=hparams["valid_dataloader_opts"],
            )
        
    # exit()
    print("\nEVALUATE\n")
    # Testing
    asr_brain.hparams.wer_file = os.path.join(
        hparams["output_folder"], "wer.txt"
    )
    asr_brain.hparams.cer_file = os.path.join(
        hparams["output_folder"], "cer.txt"
    )
    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )

