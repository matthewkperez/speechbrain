'''
Evaluate Model with Language model
'''

import torch
from speechbrain.utils.data_utils import undo_padding

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

    def on_stage_end(self, stage, stage_loss, epoch,eval_time=None):
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



# Define your ASR and language models
asr_model_dir = "/z/mkperez/speechbrain/AphasiaBank/results/duc_process_ES/No-LM/wavlm-large/freeze-True"
hparams_file = f"{asr_model_dir}/hyperparams.yaml"
run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 'device': 'cuda:0', 'data_parallel_backend': False, 'distributed_launch': False, 'distributed_backend': 'nccl', 'find_unused_parameters': False}
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin)

# hparams_file, run_opts, overrides = sb.parse_arguments(hparams_file)
asr_model = ASR(
    modules=hparams["modules"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"])

lm_model = MyLMModel()

# Load your trained ASR model weights
checkpoint = torch.load(f"{asr_model_dir}/")
asr_model.load_state_dict(checkpoint["model_state_dict"])

# Set your decoding parameters
beam_size = 10
lm_weight = 0.5

# Load your dataset and dataloader
data = MyDataset(...)
dataloader = sb.dataio.dataloader.make_dataloader(data, batch_size=1)

# Set up your output transcription list
transcriptions = []

# Set the model to evaluation mode
asr_model.eval()

# Loop over the data
for batch in dataloader:

    # Get the batch of audio signals and their lengths
    signals, lengths = batch.sig

    # Convert the signals and lengths to PyTorch tensors
    signals = signals.float().cuda()
    lengths = lengths.long().cuda()

    # Run the ASR model forward pass
    encoded_signals = asr_model.encode(signals, lengths)
    logits = asr_model.ctc_lin(encoded_signals)
    log_probs = torch.softmax(logits, dim=-1)

    # Run the language model forward pass
    lm_logits = lm_model.forward(log_probs)

    # Decode the output with beam search
    decoded_output, _ = sb.decoders.ctc_beam_search(
        lm_logits=log_probs.log_softmax(dim=-1),
        log_probs=log_probs,
        blank_id=asr_model.tokenizer.blank_index,
        beam_size=beam_size,
        lm_weight=lm_weight,
        ctc_decode=True,
    )

    # Convert the decoded output to text and add to the transcription list
    transcription = asr_model.tokenizer.decode_output(decoded_output[0])
    transcriptions.append(transcription)

# Undo padding and print transcriptions
transcriptions = undo_padding(transcriptions, data, return_lens=False)
print(transcriptions)
