# Lego V3 Fix — Minimal Upstream Lego Marker Detection

## Summary

Minimal lego fix: only 2 files changed. Lego now enters the repainting branch
(like upstream) but **skips silencing** the source latents via instruction marker
detection. No repaint_mask plumbing, no chunk_mask_modes, no return signature changes.

Also includes input peak normalization to -1 dB on all audio inputs.

## What Changed (2 files + 1 for peak norm)

### 1. `conditioning_masks.py` — LEGO MARKER DETECTION
- Added `_LEGO_INSTRUCTION_MARKER = "based on the audio context"` constant
- In the src_latents loop, when a batch item is in `repainting_ranges`,
  checks instruction text for the lego marker
- If lego: **skips** the `src_latent[start:end] = silence` step
- If not lego: silences as before (repaint/complete behavior unchanged)
- Return signature unchanged (still 4 values)

### 2. `padding_utils.py` — LEGO_FIX REPLACED
- Removed the V1 `is_lego_task` special case that set `repainting_end_batch = None`
- Lego now falls through to the same `adjusted_end = src_audio_duration` path
  as other repaint tasks
- This lets lego enter the repainting branch in conditioning_masks, where the
  marker detection correctly preserves src_latents

### 3. `io_audio.py` — PEAK NORMALIZATION (independent fix)
- `_normalize_audio_to_stereo_48k` now peak-normalizes all audio to -1 dB
- Affects src_audio, reference_audio, and target_audio paths
- Intent: keep input energy in the range the model expects

## Key Difference from Failed V2

| | V2 (broke everything) | V3 (this) |
|---|---|---|
| Files changed | 9 | 2 (+1 for peak norm) |
| Lego marker detection | Yes | Yes |
| Skip src_latent silencing | Yes | Yes |
| chunk_mask_modes (2.0) | Yes | **No** |
| repaint_mask return + plumbing | Yes (5 files) | **No** |
| Return signature changes | Yes | **No** |

V2 failed because it bundled the marker detection with chunk_mask=2.0 and
repaint_mask plumbing across 5 files. V3 isolates the single upstream change
that matters: don't silence src_latents for lego.

## How to Revert

### Revert conditioning_masks.py
Remove the `_LEGO_INSTRUCTION_MARKER` constant and the `is_lego` check.
Replace:
```python
                    instruction_i = instructions[i] if instructions and i < len(instructions) else ""
                    is_lego = _LEGO_INSTRUCTION_MARKER in instruction_i.lower()
                    if not is_lego:
                        src_latent[start_latent:end_latent] = silence_latent_tiled[start_latent:end_latent]
```
With the original unconditional silencing:
```python
                    src_latent[start_latent:end_latent] = silence_latent_tiled[start_latent:end_latent]
```

### Revert padding_utils.py (restore V1 LEGO_FIX)
In the `repainting_end` handling (~line 193), replace:
```python
                        if repainting_end is None or repainting_end < 0:
                            if is_lego_task:
                                adjusted_end = src_audio_duration + padding_info_batch[0]["left_padding_duration"]
                                repainting_end_batch = [adjusted_end] * actual_batch_size
                            else:
```
With the V1 version:
```python
                        if repainting_end is None or repainting_end < 0:
                            if is_lego_task:
                                repainting_end_batch = None
                            else:
```

### Revert io_audio.py (remove peak normalization)
In `_normalize_audio_to_stereo_48k`, remove the block after `torch.clamp`:
```python
        # Peak-normalize to -1 dB ...
        peak = audio.abs().max()
        if peak > 1e-6:
            target_peak = 10 ** (-1.0 / 20.0)
            audio = audio * (target_peak / peak)
```

## Known Issues

Garbled diffusion squeaks still occur occasionally — the V3 fix significantly reduced
their frequency but did not eliminate them entirely. This appears to be an inherent
limitation of the lego conditioning approach. Retrying with a different seed is the
current workaround. Investigating further upstream for potential fixes.

## Docker Images

- **V3 image**: `ace-step-spark:xl-base-lego-v2` (reused the tag)
- **Baseline image**: `ace-step-spark:xl-base` (V1 LEGO_FIX, no peak norm, no marker detection)
- To revert in compose: change `image:` back to `ace-step-spark:xl-base`
  then `docker compose up -d ace-step`
