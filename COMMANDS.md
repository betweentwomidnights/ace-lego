# ACE-Step Commands

## Container

```bash
# Text2music (turbo, default)
docker run --gpus all -p 8001:8001 \
  -v /home/kev/ace/checkpoints:/app/checkpoints \
  -v /home/kev/ace/data:/app/data \
  ace-step-spark:latest

# Lego / repaint / cover (base model required)
docker run --gpus all -p 8001:8001 \
  -v /home/kev/ace/checkpoints:/app/checkpoints \
  -v /home/kev/ace/data:/app/data \
  -e ACESTEP_CONFIG_PATH=acestep-v15-base \
  ace-step-spark:latest
```

> First boot downloads models to `/home/kev/ace/checkpoints/` — persisted after that.

---

## Health

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

---

## Text2Music

```bash
# Submit
curl -s -X POST http://localhost:8001/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "fingerpicked acoustic guitar, soft Rhodes piano, indie folk, warm",
    "bpm": 90,
    "key_scale": "A minor",
    "audio_duration": 30,
    "thinking": false,
    "audio_format": "mp3"
  }' | tee /tmp/task.json

# Poll (repeat until status=1)
TASK_ID=$(cat /tmp/task.json | jq -r '.data.task_id')
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" | jq '.data[0].status'

# Download
curl -o output.mp3 "http://localhost:8001/v1/audio/<filename>.mp3"
```

**Key parameters:**
| Field | Example | Notes |
|-------|---------|-------|
| `prompt` | `"lo-fi hip hop, Rhodes, dusty drums"` | Describe vibe + instrumentation, not key/BPM |
| `bpm` | `90` | Integer |
| `key_scale` | `"A minor"`, `"C major"`, `"A# major"` | Separate conditioning, not in prompt |
| `audio_duration` | `30` | Seconds |
| `thinking` | `false` | DiT-only, faster. `true` requires LLM (ACESTEP_INIT_LLM=true) |
| `audio_format` | `"mp3"` / `"wav"` / `"flac"` | |
| `batch_size` | `2` | Returns multiple candidates |

---

## Lego (add a stem to existing audio)

Requires `ACESTEP_CONFIG_PATH=acestep-v15-base`.
Audio is uploaded as multipart — no absolute paths in JSON.

**Important — always set these for lego:**
- `inference_steps=50` (default 8 is for turbo; base model needs 50)
- `thinking=false` — **critical**: when `thinking=true`, the LM generates audio semantic
  codes which force `is_cover_task=True` internally, conflicting with the lego repainting
  mask. The DiT receives contradictory conditioning and outputs garbled noise. Lego mode
  does not use LM codes by design; always disable it.
- `bpm`, `key_scale`, `time_signature` explicitly — without them there is no rhythm anchor.
- `audio_duration` — set this to match your source audio length. Without it the LM
  estimates duration and will truncate longer files (e.g. a 2:27 clip came back as 1:36).

```bash
# Submit
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=lego" \
  -F "ctx_audio=@/home/kev/ace/asharp_89bpm_1.wav" \
  -F "track_name=drums" \
  -F "caption=live acoustic drum kit, tight kick and snare, brushed hi-hats" \
  -F "bpm=89" \
  -F "key_scale=A# major" \
  -F "time_signature=4" \
  -F "inference_steps=50" \
  -F "thinking=false" \
  -F "audio_duration=20" \
  -F "repainting_start=0.0" \
  -F "repainting_end=-1" \
  -F "batch_size=2" | tee /tmp/task.json

# Poll
TASK_ID=$(cat /tmp/task.json | jq -r '.data.task_id')
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" | jq '.data[0].status'

# Download (saves each file as UUID.mp3 in current dir)
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" \
  | jq -r '.data[0].result | fromjson | .[].file' \
  | while IFS= read -r path; do
      fname=$(echo "$path" | awk -F'%2F' '{print $NF}')
      curl -o "$fname" "http://localhost:8001${path}"
    done
```

**Supported track names:** `drums` `bass` `guitar` `piano` `strings` `synth`
`keyboard` `percussion` `brass` `woodwinds` `vocals` `backing_vocals`

### Vocals (most recent working example)

```bash
# Submit — vocals over cc3_17-beatening.wav (70 BPM, E minor, 147.6s)
# Best results: E minor gave lego_cc3_vox_warm_1.mp3 (outstanding)
#               F# minor also works well on F# sections
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=lego" \
  -F "ctx_audio=@/home/kev/ace/cc3_17-beatening.wav" \
  -F "track_name=vocals" \
  -F "caption=soulful indie vocalist, warm, wordless melody, expressive, intimate" \
  -F "bpm=70" \
  -F "key_scale=E minor" \
  -F "time_signature=4" \
  -F "inference_steps=50" \
  -F "thinking=false" \
  -F "audio_duration=147.6" \
  -F "repainting_start=0.0" \
  -F "repainting_end=-1" \
  -F "batch_size=2" | tee /tmp/task.json

# Background vocals over an isolated vocal stem
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=lego" \
  -F "ctx_audio=@/home/kev/ace/lego_cc3_vox_warm_1.mp3" \
  -F "track_name=backing_vocals" \
  -F "caption=background vocals, close harmony, wordless, warm, following the lead vocal" \
  -F "bpm=70" \
  -F "key_scale=F# minor" \
  -F "time_signature=4" \
  -F "inference_steps=50" \
  -F "thinking=false" \
  -F "audio_duration=147.6" \
  -F "repainting_start=0.0" \
  -F "repainting_end=-1" \
  -F "batch_size=2" | tee /tmp/task.json
```

---

## Extract (stem separation)

Requires `ACESTEP_CONFIG_PATH=acestep-v15-base`.

**Key parameters:**
- `ctx_audio` — source audio to extract from
- `track_name` — stem to extract (`drums` `bass` `guitar` `piano` `strings` `synth`
  `keyboard` `percussion` `brass` `woodwinds` `vocals` `backing_vocals`)
- `caption` — optional, omit it. Caption is not needed; the instruction
  `"Extract the {TRACK_NAME} track from the audio:"` is sufficient and
  adding a caption shows no meaningful improvement.
- `inference_steps=50` — base model needs 50
- `thinking=false` — same reason as lego; LM would set `is_cover_task=True`
- No `repainting_start` / `repainting_end` — extract bypasses the mask entirely

**Notes:**
- Works out of the box — no patch required (unlike lego)
- Best on human-produced audio with distinct stems (e.g. live drums, real strings)
- Diffusion-output audio is harder (blended spectral content)
- Strings / tonal stems are a good pairing with melodyflow for further cleanup
- Results are imperfect but musically coherent — useful as a starting point

```bash
# Submit
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=extract" \
  -F "ctx_audio=@/home/kev/ace/adam_ldt.wav" \
  -F "track_name=drums" \
  -F "bpm=133" \
  -F "inference_steps=50" \
  -F "thinking=false" \
  -F "audio_duration=147.6" \
  -F "batch_size=2" | tee /tmp/task.json

# Poll
TASK_ID=$(cat /tmp/task.json | jq -r '.data.task_id')
while true; do
  STATUS=$(curl -s -X POST http://localhost:8001/query_result \
    -H "Content-Type: application/json" \
    -d "{\"task_id_list\": [\"$TASK_ID\"]}" | jq -r '.data[0].status')
  [ "$STATUS" = "1" ] && break
  [ "$STATUS" = "2" ] && echo "FAILED" && exit 1
  printf "."; sleep 3
done
echo " done"

# Download
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" \
  | jq -r '.data[0].result | fromjson | .[].file' \
  | while IFS= read -r path; do
      fname=$(echo "$path" | awk -F'%2F' '{print $NF}')
      curl -o "$fname" "http://localhost:8001${path}"
    done
```

---

## Repaint (regenerate a time segment)

```bash
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=repaint" \
  -F "ctx_audio=@/path/to/source.wav" \
  -F "caption=your description" \
  -F "repainting_start=8.0" \
  -F "repainting_end=16.0" \
  -F "bpm=120" \
  -F "batch_size=2" | tee /tmp/task.json
```

---

## Cover (Remix)

Requires `ACESTEP_CONFIG_PATH=acestep-v15-base`.
"Remix" in the Gradio UI == `task_type=cover` in the REST API — same task.

**Critical parameters:**
- `ctx_audio` — source audio for semantic code (VQ-VAE) conditioning (melody/structure)
- `ref_audio` — same or different audio for timbre/acoustic conditioning (strongly recommended)
- `cover_noise_strength` — **key parameter**. Controls how much the diffusion starts from source vs pure noise.
  - `0.0` = pure noise start → garbled, barely resembles source (BAD for covers)
  - `0.2` = lightly noised src → good style transfer + melody retention (Gradio recommended)
  - `0.5` = more source structure preserved
  - `0.8` = very close to source, minimal style transfer
- `audio_cover_strength` — fraction of diffusion steps using cover conditioning (1.0 = all steps)
- `inference_steps=50` — base model needs 50, not the turbo default of 8
- `thinking=false` — LM is hardcoded out for cover anyway; always false

```bash
# Submit — cover with melody retention (recommended starting point)
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=cover" \
  -F "ctx_audio=@/home/kev/ace/adam_ldt.wav" \
  -F "ref_audio=@/home/kev/ace/adam_ldt.wav" \
  -F "caption=techno, driving 4-on-the-floor kick, heavy bass, dark synthesizers, industrial" \
  -F "bpm=133" \
  -F "key_scale=F minor" \
  -F "audio_duration=147.6" \
  -F "inference_steps=50" \
  -F "thinking=false" \
  -F "audio_cover_strength=1.0" \
  -F "cover_noise_strength=0.2" \
  -F "batch_size=2" > /tmp/task.json
cat /tmp/task.json

# Poll
TASK_ID=$(python3 -c "import json; print(json.load(open('/tmp/task.json'))['data']['task_id'])")
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['status'])"

# Download
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" \
  | python3 -c "
import json, sys, subprocess
d = json.load(sys.stdin)
files = json.loads(d['data'][0]['result'])
for f in files:
    path = f['file']
    fname = path.split('%2F')[-1]
    subprocess.run(['curl', '-s', '-o', fname, 'http://localhost:8001' + path])
    print('saved:', fname)
"
```

---

## One-liner poll + download

```bash
# Wait for job and download all audio files
TASK_ID="paste-task-id-here"
while true; do
  STATUS=$(curl -s -X POST http://localhost:8001/query_result \
    -H "Content-Type: application/json" \
    -d "{\"task_id_list\": [\"$TASK_ID\"]}" | jq -r '.data[0].status')
  [ "$STATUS" = "1" ] && break
  [ "$STATUS" = "2" ] && echo "FAILED" && exit 1
  printf "."; sleep 3
done
echo " done"
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" \
  | jq -r '.data[0].result | fromjson | .[].file' \
  | while IFS= read -r path; do
      fname=$(echo "$path" | awk -F'%2F' '{print $NF}')
      curl -o "$fname" "http://localhost:8001${path}"
    done
```
