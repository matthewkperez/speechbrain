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

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

def props(cls):   
    return [i for i in cls.__dict__.keys() if i[:1] != '_']
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

global check_var
# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # print(f"batch: {props(batch)}")
        # print(f"id: {batch.id}")
        # print(f"pre wavs: {wavs}, {wav_lens}")
        # print(f"forward | w2v optimizer: {self.wav2vec_optimizer.state['lr']} | optimizer: {self.model_optimizer.state['lr']}")
        # print(f"STATE w2v optimizer: {self.wav2vec_optimizer.param_groups[0]['lr']}")
        # exit()

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                # print("env_corrupt")
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "augmentation"):
                # print("aug")
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Handling SpeechBrain vs HuggingFance pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            latents = self.modules.extractor(wavs)
            feats = self.modules.encoder_wrapper(latents, wav_lens=wav_lens)[
                "embeddings"
            ]
        else:  # HuggingFace pretrained model
            feats = self.modules.wav2vec2(wavs)

        x = self.modules.enc(feats)
    

        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        if stage != sb.Stage.TRAIN:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        # print(f"batch: {props(batch)}")
        # print(f"batch mtl: {batch.mtl}")
        # exit()

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        # multitask
        if self.hparams.mtl_flag:
            loss = loss_ctc
        else:
            loss = loss_ctc

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

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

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
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
                meta={"WER": stage_stats["WER"], "CER": stage_stats["CER"], "loss":stage_stats["loss"]}, min_keys=["loss"],
            )
        elif stage == sb.Stage.TEST:
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

        # print("check model")
        # print(f"STATE w2v optimizer: {self.wav2vec_optimizer}")
        # print(f"STATE w2v optimizer: {props(self.wav2vec_optimizer)}")
        # print(f"STATE w2v optimizer: {self.wav2vec_optimizer.param_groups[0]['lr']}")
        # print(f"STATE w2v optimizer: {self.wav2vec_optimizer.state['lr']} | optimizer: {self.model_optimizer.state['lr']}")
        # Change learning rate to hparam setting
        sb.nnet.schedulers.update_learning_rate(
            self.model_optimizer, self.model_optimizer.defaults['lr']
        )
        sb.nnet.schedulers.update_learning_rate(
            self.wav2vec_optimizer, self.wav2vec_optimizer.defaults['lr']
        )

        self.hparams.lr_annealing_model.hyperparam_value = self.model_optimizer.defaults['lr']
        self.hparams.lr_annealing_wav2vec.hyperparam_value = self.wav2vec_optimizer.defaults['lr']
        # print(f"DEFAULT w2v optimizer: {self.wav2vec_optimizer} | optimizer: {self.model_optimizer}")
        # exit()

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

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )
    train_data.data = {k:{k_2: (int(v_2) if k_2 == 'severity_cat_num' else v_2) for k_2,v_2 in v.items()} for k,v in train_data.data.items()}
    if hparams["sorting"] == "ascending":
        if 'tr_speaker' in hparams:
            # print(train_data.data['kansas12a-59'])
            # create numeric speaker_id
            print(f"hparams: {hparams['tr_speaker']}")
            tr_speaker_int = int(re.findall(r'\d+', hparams["tr_speaker"])[0])
            train_data.data = {k:{k_2: (int(re.findall(r'\d+', v_2)[0]) if k_2 == 'spk_id' else v_2) for k_2,v_2 in v.items()} for k,v in train_data.data.items()}
            # we sort training data to speed up training and get better results.
            train_data = train_data.filtered_sorted(sort_key="duration",
                key_max_value={"duration": hparams["max_length"], 
                    "severity_cat_num": hparams["max_sev_train"],
                    "spk_id": tr_speaker_int
                },
                key_min_value={"duration": hparams["min_length"], 
                    "severity_cat_num": hparams["min_sev_train"],
                    "spk_id": tr_speaker_int
                },
            )

        else:
            # we sort training data to speed up training and get better results.
            train_data = train_data.filtered_sorted(sort_key="duration",
                key_max_value={"duration": hparams["max_length"], "severity_cat_num": hparams["max_sev_train"]},
                key_min_value={"duration": hparams["min_length"], "severity_cat_num": hparams["min_sev_train"]},

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
    if 'tr_speaker' in hparams:
        valid_data.data = {k:{k_2: (int(re.findall(r'\d+', v_2)[0]) if k_2 == 'spk_id' else v_2) for k_2,v_2 in v.items()} for k,v in valid_data.data.items()}
        # we sort training data to speed up training and get better results.
        valid_data = valid_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"], 
                "spk_id": tr_speaker_int
            },
            key_min_value={"duration": hparams["min_length"], 
                "spk_id": tr_speaker_int
            },
        )
    else:
        valid_data = valid_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"]},
            key_min_value={"duration": hparams["min_length"]}
        )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder}
    )
    if 'tr_speaker' in hparams:
        test_data.data = {k:{k_2: (int(re.findall(r'\d+', v_2)[0]) if k_2 == 'spk_id' else v_2) for k_2,v_2 in v.items()} for k,v in test_data.data.items()}
        # we sort training data to speed up training and get better results.
        test_data = test_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"], 
                "spk_id": tr_speaker_int
            },
            key_min_value={"duration": hparams["min_length"], 
                "spk_id": tr_speaker_int
            },
        )
    else:
        test_data = test_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"]},
            key_min_value={"duration": hparams["min_length"]}
        )

    datasets = [train_data, valid_data, test_data]


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd", "severity_cat_num")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens", "mtl"
    )
    def text_pipeline(wrd, severity_cat_num):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        yield severity_cat_num


    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)


    # sb.dataio.dataset.add_dynamic_item([test], speechbrain.dataio.dataio.read_audio, takes="file_path", provides="signal")


    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
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

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "tokens", "mtl"],
    )


    print(f"train: {len(train_data.data)} -> {len(train_data.data_ids)} | val: {len(valid_data.data)} -> {len(valid_data.data_ids)} | test: {len(test_data.data)} -> {len(test_data.data_ids)}")
    # exit()
    return train_data, valid_data, test_data, label_encoder

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
    # exit()
    
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
    # exit()
    
    # asr_brain.modules = asr_brain.modules.float()
    count_parameters(asr_brain.modules)

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
