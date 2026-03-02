# Cover API Fix — cover_noise_strength Missing from api_server.py

## Problem

`cover_noise_strength` was completely absent from `api_server.py`. The REST API always
passed `cover_noise_strength=0.0` to the model regardless of what was sent in the request.

At `0.0` the diffusion process starts from **pure Gaussian noise**. The only structural
guidance was the lossy VQ-VAE semantic codes via `audio_cover_strength`. This produced
thin, garbled outputs that barely resembled the source audio.

The parameter existed throughout the rest of the codebase:
- `inference.py:140` — defined in `GenerationParams` dataclass
- `inference.py:603` — forwarded to `model.generate_audio()`
- `models/*/modeling_acestep_v15_*.py` — consumed by all three model variants
- Gradio UI — exposed as "Cover Strength" slider, recommended range 0.1–0.25

The gap was exclusively in `api_server.py`.

---

## Fix — 4 Changes to `api_server.py`

### 1. FIELD_ALIASES dict (~line 366)

Allows camelCase alias `coverNoiseStrength` from client requests.

```python
# Before
    "audio_cover_strength": ["audio_cover_strength", "audioCoverStrength"],

# After
    "audio_cover_strength": ["audio_cover_strength", "audioCoverStrength"],
    "cover_noise_strength": ["cover_noise_strength", "coverNoiseStrength"],
```

---

### 2. `GenerateMusicRequest` Pydantic model (~line 494)

Adds the field to the JSON request model with a safe default of `0.0`.

```python
# Before
    audio_cover_strength: float = 1.0
    task_type: str = "text2music"

# After
    audio_cover_strength: float = 1.0
    cover_noise_strength: float = 0.0
    task_type: str = "text2music"
```

---

### 3. `GenerationParams` construction in the JSON endpoint (~line 1779)

Forwards the value from the request into the inference params object.

```python
# Before
                    audio_cover_strength=req.audio_cover_strength,
                    # LM parameters

# After
                    audio_cover_strength=req.audio_cover_strength,
                    cover_noise_strength=req.cover_noise_strength,
                    # LM parameters
```

---

### 4. Multipart form parser in `/release_task` endpoint (~line 2500)

Parses the field from multipart/form-data requests (curl `-F` syntax).

```python
# Before
                audio_cover_strength=p.float("audio_cover_strength", 1.0),
                reference_audio_path=ref_audio,

# After
                audio_cover_strength=p.float("audio_cover_strength", 1.0),
                cover_noise_strength=p.float("cover_noise_strength", 0.0),
                reference_audio_path=ref_audio,
```

---

## Deployment

The fix was hot-patched into the running container without a full image rebuild:

```bash
docker cp /home/kev/ace/ACE-Step-1.5/acestep/api_server.py ace-step-spark:/app/acestep/api_server.py
docker restart ace-step-spark
```

To make the fix permanent, rebuild the image from `ACE-Step-1.5/Dockerfile.spark`.

---

## How the Parameter Flows

```
curl -F "cover_noise_strength=0.2"
  → api_server.py multipart parser (change 4)
  → GenerateMusicRequest.cover_noise_strength (change 2)
  → GenerationParams(cover_noise_strength=req.cover_noise_strength) (change 3)
  → inference.py:603 cover_noise_strength=params.cover_noise_strength
  → model.generate_audio(cover_noise_strength=0.2)
  → if cover_noise_strength > 0.0:
        effective_noise_level = 1.0 - 0.2  # = 0.8
        xt = renoise(src_latents, 0.8, noise)  # start 20% toward source
        t = t[start_idx:]  # truncate schedule accordingly
```

## What cover_noise_strength Does

| Value | Behaviour |
|-------|-----------|
| `0.0` | Pure noise start — model reconstructs from VQ-VAE codes only. Garbled. |
| `0.1–0.2` | Lightly noised source latents. Good style transfer + structure preserved. |
| `0.5` | Half noise / half source. Structure very clear, less style transformation. |
| `1.0` | Starts from raw source latents. Minimal change, near-identical output. |

Recommended default for the VST: **`0.2`**
