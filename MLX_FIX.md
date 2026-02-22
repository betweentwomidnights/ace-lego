# MLX Fix For Reliable LEGO Inference Steps

## Context

This repo is being tuned for production use of **ACE-Step LEGO mode** (layer/stem generation from context audio), not general text2music workflows.

On Apple Silicon (MPS + MLX path), we observed that `inference_steps=50` requests behaved like ~8-step runs. The output quality/timing pattern confirmed the requested step count was not being honored in MLX diffusion.

## Root Cause

The MLX diffusion path used a turbo-style fixed timestep schedule and did not receive the requested `infer_steps` from service generation.

Specifically:

1. `service_generate_execute.py` called `_mlx_run_diffusion(...)` without passing `infer_steps`.
2. `diffusion.py` did not forward `infer_steps` (or model turbo/base mode) into `mlx_generate_diffusion(...)`.
3. `models/mlx/dit_generate.py` always generated a turbo schedule when custom `timesteps` were absent.

Result: base model requests in LEGO mode defaulted to turbo-like scheduling behavior.

## Files Changed

Only these files were modified:

1. `ACE-Step-1.5/acestep/core/generation/handler/service_generate_execute.py`
2. `ACE-Step-1.5/acestep/core/generation/handler/diffusion.py`
3. `ACE-Step-1.5/acestep/models/mlx/dit_generate.py`

## What Changed

### 1) Forward requested inference steps into MLX execution

`service_generate_execute.py`

- Passes `infer_steps=generate_kwargs.get("infer_steps", 8)` into `_mlx_run_diffusion(...)`.

### 2) Propagate infer steps + model mode into MLX diffusion core

`diffusion.py`

- `_mlx_run_diffusion(...)` now accepts `infer_steps`.
- Forwards both:
  - `infer_steps`
  - `is_turbo=bool(self.config.is_turbo)`
  into `mlx_generate_diffusion(...)`.

### 3) Make timestep scheduling base-aware (not always turbo)

`models/mlx/dit_generate.py`

- `get_timestep_schedule(...)` now accepts:
  - `infer_steps`
  - `is_turbo`
- Behavior:
  - **Turbo**: keeps existing fixed mapped schedule behavior.
  - **Base**: uses PyTorch-parity schedule:
    - `linspace(1.0 -> 0.0, infer_steps + 1)`
    - optional shift transform
    - uses `t[:-1]` for diffusion steps.
- `mlx_generate_diffusion(...)` now accepts and uses `infer_steps` and `is_turbo`.

## Validation (Apple Silicon)

Validated with LEGO smoke runs (`task_type=lego`, `track_name=vocals`, context audio = `smoke.wav`):

- Before fix: runtime profile matched ~8-step behavior even when requesting 50.
- After fix:
  - MLX progress showed `50/50` steps.
  - Time-cost logs matched 50-step diffusion duration.
  - Output quality aligned with expected 50-step result.

## Scope Notes

- This fix intentionally minimizes framework changes and only addresses step-schedule correctness in MLX diffusion.
- No broad architectural refactors were introduced.
- This keeps the repo suitable as a clean reference for LEGO-focused reliability fixes across environments (Apple Silicon now, T4 next).

