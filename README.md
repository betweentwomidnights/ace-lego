# ace-step-gary

A heavily-experimented-on fork of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step), originally the server side of [gary4juce](https://github.com/betweentwomidnights/gary4juce) — a VST3 plugin for AI-assisted music inside Ableton.

Honestly, this is mostly here for posterity now. **Don't use this as a normal ACE-Step 1.5 backend — go use upstream.** Upstream has caught up to (and in some cases improved on) basically everything we were patching here. This repo just contains a pile of experiments.

## The one thing actually worth grabbing

The **LoRA load/unload bug fix**. `unload_lora()` tore an adapter down with `get_base_model()`, which leaves the PEFT modules attached — so the base-weight restore silently matched almost nothing, the adapter was never really removed, and stale adapter state leaked across generations. The visible symptom: **a fixed seed produced non-deterministic output whenever a LoRA was loaded**, plus adapter "bleed" when switching LoRAs in a single session.

The fix is basically a one-liner in `acestep/core/generation/handler/lora/lifecycle.py`: swap `get_base_model()` → `PeftModel.unload()` (keep the `load_state_dict` restore right after — it actually works once the structure is clean). This one's PR-worthy for upstream.

## Stuff that used to live here and doesn't anymore

- **LEGO_FIX** — gone. Upstream fixed the lego/repaint routing, possibly in a nicer way than we did.
- **COMPLETE_FIX** — gone. "complete" is now just complete-as-repaint (continuation mode), no special patch needed.
- **cover** — drastically better now thanks to `cover-nofsq`, which is also in upstream.

## DCW

I cannot for the life of me figure out how to get DCW to work. At all. It produces garbled insanity. My only half-theory is that it's because we don't initialize the LM when generating with ACE-Step — but that sounds like a weird reason for DCW to break absolutely everything. If you've gotten it working, I'd love to know how.
