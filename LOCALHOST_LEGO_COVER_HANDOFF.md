# Localhost Handoff — Cover-NoFSQ + Lego Upstream Parity

Two narrow fixes the localhost stack (`gary-localhost-installer`) needs to pick up so
Mac/PC users get the same lego reliability and cover quality we ship remotely.

Scope is intentionally tight: **one new task type** for cover, **one set of
deletions** for lego. Nothing else in the API surface changes.

- **Reference wrapper:** `~/gary-backend-spark/ace-step-wrapper/main.py` (remote production)
- **Reference ace-step code:** `~/ace/ACE-Step-1.5/acestep/` (already carries fix 1; fix 2 is the *removal* of patches it still has)
- **Pristine upstream lego behavior:** `~/ace-sanity-check/acestep/` (what fix 2 brings carey toward)

---

## Fix 1 — Add `cover-nofsq` task type (cover quality on turbo)

**Why:** plain `cover` runs source latents through the DiT's FSQ quantizer
(tokenize → detokenize) so the model sees a *semantic-code* version of the
reference, not the raw VAE latents. That roundtrip gives the model freedom to
morph timbre but costs acoustic fidelity. `cover-nofsq` skips the roundtrip,
preserving the source's full-resolution VAE conditioning. In practice the
remote wrapper exposes it as a `no_fsq: bool` flag on `CoverRequest`; users
hitting cover quality ceilings flip it on.

### A. Wrapper change (`gary-localhost-installer` carey_wrapper.py)

Mirror the remote pattern:

```python
class CoverRequest(BaseModel):
    # ... existing fields ...
    no_fsq: bool = Field(
        False,
        description="Bypass DiT's FSQ roundtrip on src latents (higher acoustic fidelity, less timbre morph)",
    )
```

In the form-data builder (the cover branch), translate the flag into the
backend `task_type`:

```python
if job.task_type == "cover" and getattr(req, "no_fsq", False):
    backend_task_type = "cover-nofsq"
else:
    backend_task_type = job.task_type

data["task_type"] = backend_task_type
```

Remote reference: `~/gary-backend-spark/ace-step-wrapper/main.py:287` (field
declaration) and `main.py:594-595` (translation). The localhost wrapper's
cover branch already lives near `_build_form_data` — drop the two-line
translation in the same spot.

Model routing is **unchanged**: `cover-nofsq` still uses the turbo model with
8 steps / CFG 1.0, exactly like `cover`. The only difference is the task_type
string sent to the backend.

### B. ace-step backend change (`services/carey/acestep/`)

`cover-nofsq` must be a registered task type. Five files, ~10 line-edits total.
All edits mirror what's already in `~/ace/ACE-Step-1.5/acestep/` — copy from
there.

| File | Change |
|---|---|
| `acestep/constants.py` | Add `"cover-nofsq"` to `TASK_TYPES`, `TASK_TYPES_TURBO`, `TASK_TYPES_BASE`. Add `TASK_INSTRUCTIONS["cover-nofsq"]` with the same string as `"cover"`. |
| `acestep/core/generation/handler/task_utils.py` | In `generate_instruction`, add a branch returning `TASK_INSTRUCTIONS["cover"]` when `task_type == "cover-nofsq"`. In `determine_task_type`, include `"cover-nofsq"` in the `is_cover_task` tuple. |
| `acestep/core/generation/handler/conditioning_masks.py` | In the per-item loop where `is_cover` is computed, add a `task_type == "cover-nofsq"` short-circuit that sets `is_cover = False` **before** the instruction-string match. This is the load-bearing line — it's what bypasses the FSQ roundtrip. |
| `acestep/inference.py` | Add `"cover-nofsq"` to the `skip_lm_tasks` set (skip LM Chain-of-Thought) and to the `task_type in (...)` tuple that copies caption/lyrics from params (the repaint/cover params-passthrough block). |
| `acestep/api_server.py` (optional) | If carey runs MPS unload optimizations for cover, include `"cover-nofsq"` in the MPS-only LM-unload tuples. Not strictly required for correctness. |

Concrete reference lines in `~/ace/ACE-Step-1.5/acestep/`:

- `constants.py:77,84,91,135`
- `core/generation/handler/conditioning_masks.py:63-65` ← the FSQ bypass
- `core/generation/handler/task_utils.py:78,114`
- `inference.py:389,573`
- `api_server.py:1597,1930` (optional)

### Why is_cover=False is the whole trick

In `models/turbo/modeling_acestep_v15_turbo.py:1646` (and the xl-turbo / base /
sft equivalents — all byte-identical for this expression):

```python
src_latents = torch.where(
    is_covers.unsqueeze(-1).unsqueeze(-1) > 0,
    lm_hints_25Hz,        # FSQ-roundtripped semantic codes
    src_latents,          # raw VAE latents
)
```

Cover with `is_cover=True` ⇒ FSQ codes flow as context.
Cover-nofsq with `is_cover=False` ⇒ raw VAE latents flow as context.
Same model code, different `is_covers` input. No model-file edits needed.

---

## Fix 2 — Lego: drop LEGO_FIX, use upstream-native task_type detection

**Why:** the LEGO_FIX patch series (V1: bypass repainting branch; V3: marker-
string detection + dead is_lego padding branch) was load-bearing when upstream
lacked lego-aware repaint handling. Pristine upstream now handles lego inside
the standard repaint path via a plain `task_type == "lego"` check — no patch
required. The carey copy of ace-step should drop the V1/V3 hacks and rely on
upstream's native behavior. Locally observed result on the remote
(`ace-step-spark:lego-sanity`, upstream HEAD, regular base): lego is more
reliable and quality-competitive with the LEGO_FIX prod fork.

**Status in `~/ace`:** ✓ landed (2026-05-12). The marker-string detection in
`conditioning_masks.py` has been replaced with `is_lego = (task_type == "lego")`,
and the dead `is_lego_task` branch in `padding_utils.py`'s repainting-end setup
has been collapsed. The outer `elif is_repaint_task or is_lego_task:` gate
(which lets lego enter the repaint code path at all) stays — same as pristine
upstream. The carey backend can be aligned by mirroring those two files from
`~/ace/ACE-Step-1.5/acestep/core/generation/handler/`.

### What pristine upstream looks like

Reference: `~/ace-sanity-check/acestep/`.

**`conditioning_masks.py:88`** — lego detection:

```python
is_lego = (task_type == "lego")
if not is_lego:
    src_latent[start_latent:end_latent] = silence_latent_tiled[start_latent:end_latent]
```

No `_LEGO_INSTRUCTION_MARKER` constant. No instruction-string lowercase scan.
`task_type` is already a parameter passed into `_build_chunk_masks_and_src_latents`,
so the check is just a value comparison.

**`padding_utils.py` (the repainting-end handling section)** — no
lego-specific branch. Repaint and lego share the same code path:

```python
if processed_src_audio is not None:
    src_audio_duration = processed_src_audio.shape[-1] / 48000.0
    if repainting_end is None or repainting_end < 0:
        adjusted_end = src_audio_duration + padding_info_batch[0]["left_padding_duration"]
        repainting_end_batch = [adjusted_end] * actual_batch_size
    else:
        adjusted_end = repainting_end + padding_info_batch[0]["left_padding_duration"]
        repainting_end_batch = [adjusted_end] * actual_batch_size
```

That's it. No `if is_lego_task:` special case.

### What to remove from the carey copy

If carey's ace-step started from `~/ace`'s tree, it carries:

1. **`conditioning_masks.py` top of file:** the `_LEGO_INSTRUCTION_MARKER = "based on the audio context"` constant (line ~10) and the
   ```python
   instruction_i = instructions[i] if instructions and i < len(instructions) else ""
   is_lego = _LEGO_INSTRUCTION_MARKER in instruction_i.lower()
   ```
   block in the src_latents loop (~lines 86-87).

   **Replace with:** `is_lego = (task_type == "lego")`. The `task_type` parameter is already in the function signature.

2. **`padding_utils.py` is_lego_task branch in repainting-end setup** (around
   the `if repainting_end is None or repainting_end < 0:` block, the
   `if is_lego_task:` arm). In `~/ace` that arm computes
   `adjusted_end = src_audio_duration + left_padding_duration` —
   **identical** to the else arm right below it. Collapse both arms into the
   else (drop the `if is_lego_task: ... else:` wrapper entirely in that
   conditional).

3. **(`io_audio.py`)** — Independent of the lego fix, `~/ace` also carries a
   peak-normalize-to-`-1`-dB block in `_normalize_audio_to_stereo_48k`. That
   block is **not** part of the lego-upstream-parity work. Decide separately
   whether to keep it: it's worked fine in production but isn't required. Pristine upstream doesn't have it. Recommend: keep it (it's
   a safety net) unless carey already does peak normalization elsewhere.

4. **`LEGO_FIX.md` and `LEGO_V2.md`** in carey docs (if mirrored) — historical
   only; can be deleted or marked superseded.

### What about the V1 LEGO_FIX (`repainting_end_batch = None`) — is that gone too?

Yes. `~/ace`'s V3 already deleted that. The "removal" in fix 2 only targets
the V3 leftovers (marker string + dead padding branch). If carey is on an even
older snapshot that still has the V1 `repainting_end_batch = None` for lego,
that has to go too.

### Wrapper-side: no change

The wrapper already sends `task_type=lego` (with `repainting_start=0`,
`repainting_end=-1`, `track_name=<stem>`). With pristine-upstream backend
code that's enough — the backend's `task_type == "lego"` check fires inside
the repainting branch and preserves src_latents the same way the marker hack
used to.

---

## Validation plan for carey

After landing both fixes:

1. **Cover-nofsq smoke test.** Hit the carey `/cover` endpoint with `no_fsq=true`
   and a source clip. Expect: clean cover output, audibly higher fidelity to
   the source's timbre than the same call with `no_fsq=false` (less morph).
2. **Lego smoke test.** Hit `/lego` for `vocals` then `backing_vocals` then
   `brass`. Expect: no garble across at least 2 of 3 retries each. The
   structural-sync-with-source character should be present.
3. **Cover (FSQ on) regression test.** Hit `/cover` with `no_fsq=false`. Should
   be identical to pre-change behavior — the cover-nofsq code path only fires
   when the flag is set.
4. **Repaint regression test.** Hit `/repaint` on any clip. Should be
   identical to pre-change behavior — the lego cleanup only affects the
   `task_type == "lego"` arm.

---

## Files-touched summary

| Repo | Files | Direction |
|---|---|---|
| `gary-localhost-installer` carey wrapper | `carey_wrapper.py` (or equivalent) | **Add** `no_fsq` field on CoverRequest + 2-line task_type translation |
| `gary-localhost-installer` carey ace-step | `acestep/constants.py`, `acestep/core/generation/handler/{conditioning_masks,task_utils}.py`, `acestep/inference.py`, optional `acestep/api_server.py` | **Add** cover-nofsq registration (~10 line-edits) |
| `gary-localhost-installer` carey ace-step | `acestep/core/generation/handler/{conditioning_masks,padding_utils}.py` | **Remove** marker constant + dead is_lego_task padding branch; replace marker check with `task_type == "lego"` |

`~/ace` itself has been cleaned up as part of this handoff (2026-05-12):
- `acestep/core/generation/handler/conditioning_masks.py` — `_LEGO_INSTRUCTION_MARKER`
  constant removed; `is_lego` now derived from `task_type`.
- `acestep/core/generation/handler/padding_utils.py` — dead `if is_lego_task:`
  arm in the repainting-end setup collapsed into the shared else branch.

Production already routes lego to the sanity image, so this is a cosmetic
cleanup on the remote side. The motivation is making `~/ace` a clean reference
for carey: an agent reading `~/ace`'s `acestep/core/generation/handler/` for
lego now sees what pristine upstream looks like (minus sanity-only repaint
additions — `repaint_step_injection.py`, `chunk_mask_modes`, the
`repaint_mask` 5th return value — which are not load-bearing for lego).
