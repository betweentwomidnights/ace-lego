"""Target-latent preparation helpers for handler batch conditioning."""

from typing import List, Optional, Tuple

import torch
from loguru import logger


class ConditioningTargetMixin:
    """Mixin containing target-audio to latent preparation helpers.

    Depends on host members:
    - Attributes: ``device``, ``dtype``, ``sample_rate``, ``silence_latent``.
    - Methods: ``_ensure_silence_latent_on_device ``, ``_load_model_context``,
      ``is_silence``, ``_encode_audio_to_latents``, ``_decode_audio_codes_to_latents``.
    """

    def _prepare_target_latents_and_wavs(
        self,
        batch_size: int,
        target_wavs: torch.Tensor,
        audio_code_hints: List[Optional[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Encode target audio/codes to latents and pad batch tensors."""
        self._ensure_silence_latent_on_device()

        with torch.inference_mode():
            target_latents_list = []
            latent_lengths = []
            target_wavs_list = [target_wavs[i].clone() for i in range(batch_size)]
            if target_wavs.device != self.device:
                target_wavs = target_wavs.to(self.device)

            with self._load_model_context("vae"):
                _cached_wav_ref: Optional[torch.Tensor] = None
                _cached_latent: Optional[torch.Tensor] = None

                for i in range(batch_size):
                    code_hint = audio_code_hints[i]
                    if code_hint:
                        logger.info(f"[generate_music] Decoding audio codes for item {i}...")
                        decoded_latents = self._decode_audio_codes_to_latents(code_hint)
                        if decoded_latents is not None:
                            decoded_latents = decoded_latents.squeeze(0)
                            target_latents_list.append(decoded_latents)
                            latent_lengths.append(decoded_latents.shape[0])
                            frames_from_codes = max(1, int(decoded_latents.shape[0] * 1920))
                            target_wavs_list[i] = torch.zeros(2, frames_from_codes)
                            continue

                    current_wav = target_wavs_list[i].to(self.device).unsqueeze(0)
                    if self.is_silence(current_wav):
                        expected_latent_length = current_wav.shape[-1] // 1920
                        target_latent = self.silence_latent[0, :expected_latent_length, :]
                    else:
                        if (
                            _cached_wav_ref is not None
                            and _cached_latent is not None
                            and _cached_wav_ref.shape == current_wav.shape
                            and torch.equal(_cached_wav_ref, current_wav)
                        ):
                            logger.info(
                                f"[generate_music] Reusing cached VAE latents for item {i} (same audio as previous item)"
                            )
                            target_latent = _cached_latent.clone()
                        else:
                            # Detect silence head/tail (e.g. zero-padded prelude/complete
                            # mode audio) and encode only the real audio portion, then
                            # concat with clean silence_latent to avoid VAE boundary
                            # artifacts.
                            wav_2d = current_wav.squeeze(0)  # [2, samples]
                            energy = wav_2d.pow(2).sum(0)     # per-sample energy
                            nonzero = (energy > 1e-10).nonzero(as_tuple=True)[0]
                            total_samples = wav_2d.shape[-1]

                            if len(nonzero) > 0:
                                first_audio_sample = nonzero[0].item()
                                last_audio_sample = nonzero[-1].item() + 1
                            else:
                                first_audio_sample = 0
                                last_audio_sample = 0

                            silence_head_samples = first_audio_sample
                            silence_tail_samples = total_samples - last_audio_sample

                            if silence_head_samples > 1920 and last_audio_sample > 0:
                                # Prelude mode: silence at the beginning, audio at the end.
                                # Round down head to nearest latent frame boundary so audio
                                # starts on a clean frame.
                                head_latent_frames = silence_head_samples // 1920
                                encode_start = head_latent_frames * 1920
                                audio_portion = wav_2d[:, encode_start:]
                                # Also strip any silence tail
                                real_end = last_audio_sample - encode_start
                                encode_end = ((real_end + 1919) // 1920) * 1920
                                encode_end = min(encode_end, audio_portion.shape[-1])
                                audio_portion = audio_portion[:, :encode_end]
                                logger.info(
                                    f"[generate_music] Encoding source audio "
                                    f"({audio_portion.shape[-1] / 48000:.2f}s) separately from "
                                    f"silence head ({silence_head_samples / 48000:.2f}s) "
                                    f"for item {i}..."
                                )
                                target_latent = self._encode_audio_to_latents(audio_portion)
                                # Prepend clean silence_latent for the head
                                full_latent_length = total_samples // 1920
                                if head_latent_frames + target_latent.shape[0] < full_latent_length:
                                    tail_pad = full_latent_length - head_latent_frames - target_latent.shape[0]
                                    target_latent = torch.cat(
                                        [
                                            self.silence_latent[0, :head_latent_frames, :].to(target_latent.device),
                                            target_latent,
                                            self.silence_latent[0, :tail_pad, :].to(target_latent.device),
                                        ],
                                        dim=0,
                                    )
                                else:
                                    target_latent = torch.cat(
                                        [
                                            self.silence_latent[0, :head_latent_frames, :].to(target_latent.device),
                                            target_latent,
                                        ],
                                        dim=0,
                                    )
                                    # Trim to exact length if needed
                                    if target_latent.shape[0] > full_latent_length:
                                        target_latent = target_latent[:full_latent_length]
                            elif silence_tail_samples > 1920 and last_audio_sample > 0:
                                # Complete mode: audio at the beginning, silence at the end.
                                # Round up to nearest latent frame boundary
                                encode_samples = ((last_audio_sample + 1919) // 1920) * 1920
                                encode_samples = min(encode_samples, total_samples)
                                audio_portion = wav_2d[:, :encode_samples]
                                logger.info(
                                    f"[generate_music] Encoding source audio "
                                    f"({encode_samples / 48000:.2f}s) separately from "
                                    f"silence tail ({silence_tail_samples / 48000:.2f}s) "
                                    f"for item {i}..."
                                )
                                target_latent = self._encode_audio_to_latents(audio_portion)
                                # Pad with clean silence_latent to full length
                                full_latent_length = total_samples // 1920
                                if target_latent.shape[0] < full_latent_length:
                                    pad_len = full_latent_length - target_latent.shape[0]
                                    target_latent = torch.cat(
                                        [target_latent, self.silence_latent[0, :pad_len, :].to(target_latent.device)],
                                        dim=0,
                                    )
                            else:
                                logger.info(f"[generate_music] Encoding target audio to latents for item {i}...")
                                target_latent = self._encode_audio_to_latents(wav_2d)

                            _cached_wav_ref = current_wav
                            _cached_latent = target_latent
                    target_latents_list.append(target_latent)
                    latent_lengths.append(target_latent.shape[0])

            max_target_frames = max(wav.shape[-1] for wav in target_wavs_list)
            padded_target_wavs = []
            for wav in target_wavs_list:
                if wav.shape[-1] < max_target_frames:
                    pad_frames = max_target_frames - wav.shape[-1]
                    wav = torch.nn.functional.pad(wav, (0, pad_frames), "constant", 0)
                padded_target_wavs.append(wav)
            target_wavs = torch.stack(padded_target_wavs)

            max_latent_length = max(latent.shape[0] for latent in target_latents_list)
            max_latent_length = max(128, max_latent_length)
            silence_latent_tiled = self.silence_latent[0, :max_latent_length, :]

            padded_latents = []
            for latent in target_latents_list:
                latent_length = latent.shape[0]
                if latent_length < max_latent_length:
                    pad_length = max_latent_length - latent_length
                    latent = torch.cat([latent, self.silence_latent[0, :pad_length, :]], dim=0)
                padded_latents.append(latent)

            target_latents = torch.stack(padded_latents)
            latent_masks = torch.stack(
                [
                    torch.cat(
                        [
                            torch.ones(l, dtype=torch.long, device=self.device),
                            torch.zeros(max_latent_length - l, dtype=torch.long, device=self.device),
                        ]
                    )
                    for l in latent_lengths
                ]
            )
            return target_wavs, target_latents, latent_masks, max_latent_length, silence_latent_tiled
