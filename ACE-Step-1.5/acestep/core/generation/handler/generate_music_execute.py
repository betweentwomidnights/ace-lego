"""Execution helper for ``generate_music`` service invocation with progress tracking."""

from typing import Any, Callable, Dict, List, Optional, Sequence


class GenerateMusicExecuteMixin:
    """Run service generation under diffusion progress estimation lifecycle."""

    def _run_generate_music_service_with_progress(
        self,
        progress: Any,
        actual_batch_size: int,
        audio_duration: Optional[float],
        inference_steps: int,
        timesteps: Optional[Sequence[float]],
        service_inputs: Dict[str, Any],
        refer_audios: Optional[List[Any]],
        guidance_scale: float,
        actual_seed_list: Optional[List[int]],
        audio_cover_strength: float,
        cover_noise_strength: float,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        shift: float,
        infer_method: str,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """Invoke ``service_generate`` while maintaining background progress estimation."""
        infer_steps_for_progress = len(timesteps) if timesteps else inference_steps
        progress_desc = f"Generating music (batch size: {actual_batch_size})..."
        progress(0.52, desc=progress_desc)

        # Per-step DiT callback: maps `step_idx/total` into the diffusion band
        # [0.52, 0.79] that the thread estimator also targets. This is TRUTH
        # (driven from inside the DiT loop) and replaces the time-interpolation
        # estimator for paths that thread it through (PyTorch xl-base/xl-turbo
        # generate_audio). The estimator is kept as a fallback for paths that
        # don't (e.g., MLX); api_server._progress_cb throttles by delta ≥ 0.01
        # so the per-step monotonic ticks dominate when both fire.
        _DIFF_START, _DIFF_END = 0.52, 0.79

        def _per_step_progress(step_idx: int, total: int) -> None:
            if progress is None or total <= 0:
                return
            frac = min(1.0, max(0.0, (step_idx + 1) / total))
            try:
                progress(_DIFF_START + (_DIFF_END - _DIFF_START) * frac, desc=progress_desc)
            except Exception:
                pass

        stop_event = None
        progress_thread = None
        try:
            stop_event, progress_thread = self._start_diffusion_progress_estimator(
                progress=progress,
                start=_DIFF_START,
                end=_DIFF_END,
                infer_steps=infer_steps_for_progress,
                batch_size=actual_batch_size,
                duration_sec=audio_duration if audio_duration and audio_duration > 0 else None,
                desc=progress_desc,
            )
            outputs = self.service_generate(
                captions=service_inputs["captions_batch"],
                lyrics=service_inputs["lyrics_batch"],
                metas=service_inputs["metas_batch"],
                vocal_languages=service_inputs["vocal_languages_batch"],
                refer_audios=refer_audios,
                target_wavs=service_inputs["target_wavs_tensor"],
                infer_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=actual_seed_list,
                repainting_start=service_inputs["repainting_start_batch"],
                repainting_end=service_inputs["repainting_end_batch"],
                instructions=service_inputs["instructions_batch"],
                audio_cover_strength=audio_cover_strength,
                cover_noise_strength=cover_noise_strength,
                use_adg=use_adg,
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                shift=shift,
                infer_method=infer_method,
                audio_code_hints=service_inputs["audio_code_hints_batch"],
                return_intermediate=service_inputs["should_return_intermediate"],
                timesteps=timesteps,
                task_type=task_type,
                progress_callback=_per_step_progress,
            )
        finally:
            if stop_event is not None:
                stop_event.set()
            if progress_thread is not None:
                progress_thread.join(timeout=1.0)
        return {"outputs": outputs, "infer_steps_for_progress": infer_steps_for_progress}
