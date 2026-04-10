# Complete Mode — Continuation via Repaint

## Current Approach (2026-04-08)

Complete mode now works by **translating to a repaint task** at the wrapper level.
Instead of custom `is_complete_task` code paths through padding/conditioning, the
wrapper sends `task_type=repaint` with:

- `repainting_start = source_duration` — locks source audio frames as context
- `repainting_end = target_duration` — generates the continuation region

This is simpler and more reliable than the original custom-hack approach (below).
The ace-step backend just sees a normal repaint job.

### Model routing

- **xl-turbo** (default): 8 steps, CFG 1.0 — clean, fast, good for most cases
- **xl-base**: 50 steps, CFG 7.0 — more creative/complex, slower
- Regular turbo/base also work with their respective step/CFG defaults
- Wrapper accepts `"model": "xl-base"` to route to base; defaults to turbo

### Key parameters

- `thinking=false` — always. LM would force `is_cover_task=True` via audio codes.
- `caption` — style description steers the continuation
- `lyrics` — optional lyrics for the generated portion
- `key_scale` — optional, helps harmonic continuity
- `use_src_as_ref=true` — pass source as ref_audio for timbre anchoring (optional)

### Wrapper translation (ace-step-wrapper/main.py)

```python
# In _build_form_data:
data["task_type"] = "repaint"  # not "complete"
data["repainting_start"] = str(job.duration)      # source audio end
data["repainting_end"] = str(req.audio_duration)   # user's target duration
```

---

## Historical Context: Original Custom Hack

The approach below was our first attempt — modifying 4+ files inside ace-step to make
`task_type=complete` work as a continuation task. It worked but was fragile and
unnecessary once we realized that `repaint` with the right start/end coordinates
does the same thing natively.

The code changes from this phase are still present in the codebase (task_utils.py
recognizes `is_complete_task`, padding_utils.py has a complete branch, etc.) but
they are effectively **dead code** now that the wrapper sends `task_type=repaint`.

> ### Background
>
> `complete` was NOT originally designed as a continuation/extension task. Its actual
> intended use case is: give it a stem (e.g. just vocals), specify track classes, and
> it fills out the full arrangement at the same duration. Think "band-in-a-box."
>
> We repurposed it for temporal continuation — give it 10s of audio, ask for 120s,
> and it generates from where the source ends. The repaint approach does this cleanly.

### Original problems solved (for reference)

1. **`task_utils.py`** — `complete` excluded from `can_use_repainting`
2. **`padding_utils.py`** — `complete` fell into bare "else" branch, always source-length
3. **`padding_utils.py`** — `repainting_end_batch` not set to None for `complete`
4. **`batch_prep.py`** — metadata duration ignored `audio_duration` when src_audio present
5. **`conditioning_target.py`** — VAE boundary artifact (silence gap at audio→padding boundary)

The VAE boundary fix (encoding only the audio portion, then splicing canonical
`silence_latent` for the padding region) is still active and beneficial — it helps
any task with trailing silence, including lego mode.
