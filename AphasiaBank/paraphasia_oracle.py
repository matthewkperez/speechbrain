#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the following:
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
import datetime
import transformers
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

def props(cls):   
    return [i for i in cls.__dict__.keys() if i[:1] != '_']
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        # print(props(batch))
        # print(batch.para_tokens)

        # print(batch.paraphasia)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                # print("env_corrupt")
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "augmentation"):
                if isinstance(self.hparams.augmentation, sb.lobes.augment.SpecAugment):
                    wavs = self.hparams.augmentation(wavs)    
                elif isinstance(self.hparams.augmentation, sb.lobes.augment.TimeDomainSpecAugment):
                    wavs = self.hparams.augmentation(wavs, wav_lens)

        # Handling SpeechBrain vs HuggingFance pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            latents = self.modules.extractor(wavs)
            feats = self.modules.encoder_wrapper(latents, wav_lens=wav_lens)[
                "embeddings"
            ]
        else:  # HuggingFace pretrained model
            feats = self.modules.wav2vec2(wavs)


        para_emb,_ = self.modules.para_enc(feats)
    

        para_logits = self.modules.para_lin(para_emb)
        p_para = self.hparams.log_softmax(para_logits)

        return p_para, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_para, wav_lens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens
        gt_para, para_lens = batch.para_tokens

        loss_para = self.hparams.ctc_cost(p_para, gt_para, wav_lens, para_lens)
        
        # multitask
        if self.hparams.mtl_flag:
            loss = loss_para
        else:
            loss = loss_para

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_para, wav_lens, blank_id=self.hparams.blank_index
            )

            predicted_words = [
                " ".join(self.tokenizer.decode_ndim(utt_seq))
                for utt_seq in p_tokens
            ]
            target_words = batch.paraphasia
            # print(f"predicted_words: {predicted_words}")
            # print(f"target_words: {target_words}")
            # truth_arr = []
            # for g in gt_para:
            #     # print(g[g.nonzero().squeeze().detach()])
            #     # print(g[g.nonzero().squeeze().detach()].cpu().detach().numpy())
            #     # exit()
            #     truth_arr.append(list(g[g.nonzero().squeeze().detach()].cpu().detach().numpy()))
            # print(f"truth_arr: {truth_arr}")
            # print(f"p_tokens: {p_tokens}")
            

            # # exit()
            # print(f1_score(truth_arr,p_tokens))
            # exit()
            # self.f1_score.append(ids, p_tokens, truth_arr)

            self.wer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # Old
            # self.scaler.scale(loss / self.grad_accumulation_factor).backward()
            # new
            with self.no_sync(not should_step):
                self.scaler.scale(loss / self.grad_accumulation_factor).backward()


            if should_step:
                if not self.hparams.freeze_wav2vec:
                    self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    if not self.hparams.freeze_wav2vec:
                        self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.optimizer_step += 1
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            # print(self.modules.enc)
            # print(f"Lin weight grad: {self.modules.enc.linear.w.weight.grad.isnan().any()}")
            # print(f"Lin bias grad: {self.modules.enc.linear.w.bias.grad.isnan().any()}")
            # print(f"Lin0 weight grad: {self.modules.enc.linear_0.w.weight.grad.isnan().any()}")
            # print(f"Lin0 bias grad: {self.modules.enc.linear_0.w.bias.grad.isnan().any()}")
            # print(f"self.grad_accumulation_factor: {self.grad_accumulation_factor}")
            # print(f"loss: {loss}, {loss.type()}")
            # print(f"batch tokens: {batch.tokens}")
            # print(self.modules)
            # for name, param in self.modules.enc.named_parameters():
            #     # print("Model Parameters",name, torch.isfinite(param.grad).all())
            #     print("Model Parameters",name, param.grad.isnan().any())
            # exit()
            if should_step:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.model_optimizer.step()
                self.wav2vec_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.optimizer_step += 1

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.f1_score = self.hparams.f1(metric=f1_score)

    def on_stage_end(self, stage, stage_loss, epoch,eval_time=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # self.checkpointer.save_and_keep_only(
            #     meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            # )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "loss":stage_stats["loss"]}, min_keys=["loss"],
            )
        elif stage == sb.Stage.TEST:
            stage_stats['Time'] = str(datetime.timedelta(seconds=eval_time))
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
            
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        # Handling SpeechBrain vs HuggingFance pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.encoder_wrapper.latent_encoder.parameters()
            )

        else:  # HuggingFace pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )


        # # Change learning rate to hparam setting
        # sb.nnet.schedulers.update_learning_rate(
        #     self.model_optimizer, self.model_optimizer.defaults['lr']
        # )
        # sb.nnet.schedulers.update_learning_rate(
        #     self.wav2vec_optimizer, self.wav2vec_optimizer.defaults['lr']
        # )
        # self.hparams.lr_annealing_model.hyperparam_value = self.model_optimizer.defaults['lr']
        # self.hparams.lr_annealing_wav2vec.hyperparam_value = self.wav2vec_optimizer.defaults['lr']

    def _fit_train(self, train_set, epoch, enable):
        # Training stage
        self.on_stage_start(sb.Stage.TRAIN, epoch)
        self.modules.train()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
        ) as t:
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                loss = self.fit_batch(batch)

                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes > 0
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    # This should not use run_on_main, because that
                    # includes a DDP barrier. That eventually leads to a
                    # crash when the processes'
                    # time.time() - last_ckpt_time differ and some
                    # processes enter this block while others don't,
                    # missing the barrier.
                    if sb.utils.distributed.if_main_process():
                        self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()

        # Run train "on_stage_end" on all processes
        self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        eval_start_time = time.time()
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            eval_time = time.time()-eval_start_time
            run_on_main(
                self.on_stage_end, args=[sb.Stage.TEST, avg_test_loss, None,eval_time]
            )
        self.step = 0
        return avg_test_loss


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    #convert severity_cat to int
    train_data.data = {k:{k_2: (int(v_2) if k_2 == 'severity_cat' else v_2) for k_2,v_2 in v.items()} for k,v in train_data.data.items()}

    # print(train_data.data['wright99a-93'])
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
    
    valid_data = valid_data.filtered_sorted(sort_key="duration",
        key_max_value={"duration": hparams["max_length"]},
        key_min_value={"duration": hparams["min_length"]}
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder}
    )
    
    test_data = test_data.filtered_sorted(sort_key="duration",
        key_max_value={"duration": hparams["max_length"]},
        key_min_value={"duration": hparams["min_length"]}
    )

    datasets = [train_data, valid_data, test_data]

    # binarize paraphasia
    for d in datasets:
        for utt_id,v in d.data.items():
            d.data[utt_id]['paraphasia'] = " ".join([p if p == 'c' else hparams['paraphasia_var'] for p in d.data[utt_id]['paraphasia'].split()])
        # d.data = {utt_id:{'paraphasia': ['c' if p == 'c' else hparams['paraphasia_var'] for p in v['paraphasia']]} for utt_id,v in d.data.items()}
    # print(f"train_data: {train_data.data}")
    # print(hparams['paraphasia_var'])
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    para_label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd", "paraphasia")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens","para_lst", "para_tokens"
    )
    def text_pipeline(wrd, paraphasia):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        # print(f"paraphasia: {paraphasia}")
        # paraphasia_map = {'c':0, 'p':1, 'n':2}
        
        para_lst = paraphasia.split() #pn
        yield para_lst
        para_tokens = torch.LongTensor(para_label_encoder.encode_sequence(para_lst))
        # print(f"para_tokens: {para_tokens}")
        yield para_tokens


    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)


    # sb.dataio.dataset.add_dynamic_item([test], speechbrain.dataio.dataio.read_audio, takes="file_path", provides="signal")


    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    plab_enc_file = os.path.join(hparams["save_folder"], "para_label_encoder.txt")
    # add special labels
    special_labels = {
        "blank_label": hparams["blank_index"],    
    }

    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )
    para_label_encoder.load_or_create(
        path=plab_enc_file,
        from_didatasets=[train_data],
        output_key="para_lst",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "tokens", "para_tokens", "paraphasia"],
    )


    print(f"train: {len(train_data.data)} -> {len(train_data.data_ids)} | val: {len(valid_data.data)} -> {len(valid_data.data_ids)} | test: {len(test_data.data)} -> {len(test_data.data_ids)}")
    return train_data, valid_data, test_data, para_label_encoder

def prep_exp_dir(hparams):
    save_folder = hparams['save_folder']
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

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

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data, label_encoder = dataio_prepare(
        hparams
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder
    print(f"tokenizer: {asr_brain.tokenizer.lab2ind}")
    print(f"tokenizer: {len(asr_brain.tokenizer.lab2ind.keys())}")
    count_parameters(asr_brain.modules)
    
    # # Bert Tokenizer
    # # asr_brain.bert_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
    # asr_brain.bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # asr_brain.bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
    # print(asr_brain.bert_tokenizer)
    # print(asr_brain.bert_model)


    with torch.autograd.detect_anomaly():
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

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

