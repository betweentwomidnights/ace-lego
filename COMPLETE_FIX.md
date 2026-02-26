# Complete Task Fix

> ## ⚠️ NOTE: WE DID NOT UNDERSTAND THIS TASK TYPE
>
> After getting this working we discovered that `complete` was NOT originally designed as a
> continuation/extension task. Its actual intended use case is: **give it a stem (e.g. just
> vocals, or just guitar), specify which track classes you want added, and it fills out the
> full arrangement at the same duration.** Think "band-in-a-box" — drop in a vocal stem,
> get back a full song.
>
> The reason it was returning source-length audio before was arguably *correct* for that
> primary use case. It only broke when you wanted `audio_duration > src_audio_duration`.
>
> What our fix actually added: **temporal extension / continuation**. Give it 10s of guitar,
> ask for 120s, and it free-runs the arrangement from where the source ends. With no caption
> and no track_classes it goes completely feral — random instrumentation, vocals, full
> compositions — which turns out to be extremely useful for sampling and musical exploration,
> just not what the task was originally for.
>
> **but now it does continuations lol**

---

## Problem

`task_type=complete` with `audio_duration > src_audio_duration` returned output matching
the source audio length (e.g. 10s) instead of the requested duration (e.g. 30s). The
output also behaved like a cover — no harmonic relationship to the source.

## Root Cause

Four independent issues conspired to produce the broken behavior:

---

### 1. `task_utils.py` — `complete` excluded from `can_use_repainting`

**File:** `acestep/core/generation/handler/task_utils.py`

```python
# BEFORE (broken)
can_use_repainting = is_repaint_task or is_lego_task
return is_repaint_task, is_lego_task, is_cover_task, can_use_repainting
```

`complete` was not recognized as a repainting-capable task, so `can_use_repainting=False`
for all complete requests. This caused `padding_utils.py` to skip all repainting
coordinate logic and set `repainting_start_batch = None, repainting_end_batch = None`.

---

### 2. `padding_utils.py` — `complete` fell into the bare "else" branch

**File:** `acestep/core/generation/handler/padding_utils.py`

```python
# BEFORE (broken)
elif is_repaint_task or is_lego_task:
    # ... padding logic ...
else:
    # Other tasks: Use src_audio directly without padding
    batch_target_wavs = processed_src_audio   # <- always 10s, never extended
    padding_info_batch.append({"left_padding_duration": 0.0, "right_padding_duration": 0.0})
```

The `complete` task fell into the catch-all else, so `target_wavs_tensor` was always
the source audio duration regardless of `audio_duration`.

---

### 3. `padding_utils.py` — `repainting_end_batch` not set to None for `complete`

Also in `padding_utils.py`, the lego special-case that leaves `repainting_end_batch = None`
(preserving the full-mask path in `conditioning_masks.py`) only checked `is_lego_task`.
A positive `repainting_end` value would route through the repainting branch, which silences
`src_latents` and destroys the source audio context — the same bug documented in LEGO_FIX.md.

---

### 4. `batch_prep.py` — metadata duration ignored `audio_duration` when src_audio was present

**File:** `acestep/core/generation/handler/batch_prep.py`

```python
# BEFORE (broken)
if processed_src_audio is not None:
    calculated_duration = processed_src_audio.shape[-1] / 48000.0   # always src duration
elif audio_duration is not None and float(audio_duration) > 0:
    calculated_duration = float(audio_duration)
```

The metadata passed to the DiT always reflected the source audio duration (10s), so
the model targeted 10s even if `target_wavs_tensor` had been extended correctly.

---

## Fix

### `task_utils.py`

Add `is_complete_task` flag and include it in `can_use_repainting`. Return it as a
fifth value so downstream callers can distinguish it from repaint/lego:

```python
# AFTER
is_complete_task = task_type == "complete"
can_use_repainting = is_repaint_task or is_lego_task or is_complete_task
return is_repaint_task, is_lego_task, is_cover_task, can_use_repainting, is_complete_task
```

### `generate_music_request.py`

Update unpacking and pass `is_complete_task` to `prepare_padding_info`:

```python
is_repaint_task, is_lego_task, is_cover_task, can_use_repainting, is_complete_task = \
    self.determine_task_type(task_type, audio_code_string)
repainting_start_batch, repainting_end_batch, target_wavs_tensor = self.prepare_padding_info(
    ..., is_complete_task,
)
```

### `padding_utils.py`

Add `is_complete_task=False` parameter. Add explicit branch before the catch-all else:

```python
elif is_complete_task:
    # Pad source audio to the desired audio_duration if longer.
    src_audio_duration = processed_src_audio.shape[-1] / 48000.0
    target_duration = (
        float(audio_duration)
        if audio_duration is not None and float(audio_duration) > src_audio_duration
        else src_audio_duration
    )
    right_padding_frames = int(max(0, target_duration - src_audio_duration) * 48000)
    if right_padding_frames > 0:
        batch_target_wavs = torch.nn.functional.pad(
            processed_src_audio, (0, right_padding_frames), "constant", 0
        )
    else:
        batch_target_wavs = processed_src_audio
    padding_info_batch.append(
        {"left_padding_duration": 0.0, "right_padding_duration": target_duration - src_audio_duration}
    )
```

For `repainting_end_batch`, extend the lego guard to cover complete:

```python
if is_lego_task or is_complete_task:
    # Leave as None → full-mask branch → src_latents preserved as context
    repainting_end_batch = None
```

### `batch_prep.py`

Prefer `audio_duration` over src duration when user requests longer output:

```python
if processed_src_audio is not None:
    src_duration = processed_src_audio.shape[-1] / 48000.0
    if audio_duration is not None and float(audio_duration) > src_duration:
        calculated_duration = float(audio_duration)
    else:
        calculated_duration = src_duration
elif audio_duration is not None and float(audio_duration) > 0:
    calculated_duration = float(audio_duration)
```

---

## Result

- `audio_duration=30` with a 10s source file → 30s output ✓
- Source audio context preserved via full-mask path (same mechanism as LEGO_FIX) ✓
- Metadata duration reflects the requested target, not the source ✓

## Notes

- `thinking=false` is required (same as lego) — the LM provides no benefit here and
  would force `is_cover_task=True` via audio code generation, conflicting with the mask.
- `inference_steps=50` required — `complete` is base-model only.
- `track_classes` or an explicit `instruction` should specify what to add:
  `"instruction=Complete the input track with DRUMS | BASS | PIANO | STRINGS:"`
- The `repainting_start` parameter defaults to 0 (full source as context). Partial
  completion from a specific time offset is untested.
