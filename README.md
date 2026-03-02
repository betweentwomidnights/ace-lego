# ace-step-gary

ACE-Step 1.5 deployment fork — patched REST API backend for [gary4juce](https://github.com/betweentwomidnights/gary4juce), a VST3 plugin for AI-assisted music production inside Ableton Live.

This repo is the server side. It needs to be paired with one of the gary installers for local use:
- **Windows:** gary-localhost-installer
- **Mac:** gary-localhost-installer-mac

On a remote GPU (T4), it runs inside a Docker container with `Dockerfile.t4` at the project root.

---

## Modes

All modes below require `ACESTEP_CONFIG_PATH=acestep-v15-base` and `inference_steps=50`.
See [`COMMANDS.md`](COMMANDS.md) for working curl examples for every mode.

### lego
Add a new stem (drums, bass, vocals, etc.) to an existing piece of audio. The model generates the new track in context with the source audio's harmonic and rhythmic content.

Patch: [`LEGO_FIX.md`](LEGO_FIX.md) — fixed `repainting_end_batch` routing that was silencing `src_latents` and causing the DiT to generate freely from text alone.

### complete
Extend a short audio clip into a longer arrangement, or fill out a stem into a full mix. Drop in 10s of guitar, get back 120s of full composition. With no caption or track_classes it goes fully feral — random instrumentation, vocals, full arrangements — which is useful for sampling.

Patch: [`COMPLETE_FIX.md`](COMPLETE_FIX.md) — four-file fix enabling temporal extension: `task_utils.py`, `padding_utils.py`, `generate_music_request.py`, `batch_prep.py`.

### cover
Style-transfer a piece of audio. "Remix" in the Gradio UI is the same task. Key parameter: `cover_noise_strength` — controls how much the diffusion starts from the source vs. pure noise. Default `0.2` is the sweet spot for style transfer with structure retained.

Patch: [`COVER_API_FIX.md`](COVER_API_FIX.md) — `cover_noise_strength` was missing from `api_server.py`, causing the API to always start from pure noise regardless of what was passed.

### extract
Separate a stem (drums, strings, vocals, etc.) from a full mix. Works best on human-produced audio with distinct stem separation. Diffusion-output audio is harder. No caption needed — the instruction `"Extract the {TRACK_NAME} track from the audio:"` is sufficient.

No patch required — works out of the box.

---

## api_server.py patches

All fixes land in `ACE-Step-1.5/acestep/api_server.py` plus the handful of handler files documented in each fix file. To hot-patch a running container:

```bash
docker cp ACE-Step-1.5/acestep/api_server.py <container_name>:/app/acestep/api_server.py
docker restart <container_name>
```

---

## Structure

```
ACE-Step-1.5/   patched fork of ace-step/ACE-Step-1.5
wrapper/        FastAPI VRAM lifecycle service (T4 deployment, legacy)
Dockerfile.t4   T4 GPU container build (moved to root for clarity)
```

---

## TODOs

- [ ] Add `extract` mode to gary4juce VST UI
- [ ] Remove `Dockerfile.t4` and `wrapper/` from main — split into dedicated branches:
  - `branch/t4` — T4 GPU remote deployment
  - `branch/mac` — local Mac (gary-localhost-installer-mac)
  - `branch/pc` — local Windows (gary-localhost-installer)
- [ ] Test `lego` with vocals + lyrics once LM is stable (`ACESTEP_INIT_LLM=true`)
