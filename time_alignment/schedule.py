"""Pure schedule helpers for exact Scene/Gaussian optimizer-step accounting."""


def support_switch_iteration(calibration_steps, start_iteration=1):
    """First iteration after the alternating calibration/reconstruction prefix."""
    if calibration_steps <= 0 or start_iteration < 1:
        raise ValueError("invalid support switch budget")
    return start_iteration + 2 * calibration_steps


def freeze_records_valid(records, last_clock_iteration, total_iterations,
                         clock_steps, final_offset, legacy_status_records=False):
    """Validate a transition event, or explicitly identified old status logs."""
    import math

    expected_iterations = (list(range(last_clock_iteration, total_iterations + 1))
                           if legacy_status_records else [last_clock_iteration])
    if [row.get("iteration") for row in records] != expected_iterations:
        return False
    for row in records:
        offset = row.get("offset_frames")
        if (row.get("clock_optimizer_steps") != clock_steps
                or row.get("requires_grad_next") is not False
                or not isinstance(offset, (int, float))
                or not math.isfinite(offset)
                or not isinstance(final_offset, (int, float))
                or not math.isfinite(final_offset)
                or abs(offset - final_offset) > 1e-8):
            return False
    return True


def should_step_scene(full_scene_steps, optimizer_phase):
    if optimizer_phase not in {
            "calibration", "reconstruction", "scene_tail", "warmup", "joint"}:
        raise ValueError(f"invalid optimizer phase: {optimizer_phase!r}")
    return bool(full_scene_steps) or optimizer_phase != "calibration"


def should_run_optimizer_block(iteration, total_iterations, full_scene_steps,
                               strict_step_budget_v2=False):
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if iteration < 1 or iteration > total_iterations:
        raise ValueError("iteration outside the main training budget")
    if strict_step_budget_v2:
        return True
    return iteration < total_iterations or bool(full_scene_steps)


def phase_for_iteration(iteration, phase_length=8, start_iteration=1,
                        calibration_steps=None, scene_steps=None):
    if phase_length <= 0:
        raise ValueError("phase_length must be positive")
    if iteration < start_iteration:
        return "warmup"
    if calibration_steps is not None or scene_steps is not None:
        if calibration_steps is None or scene_steps is None:
            raise ValueError("calibration_steps and scene_steps must be paired")
        if calibration_steps <= 0 or scene_steps < calibration_steps:
            raise ValueError("invalid strict step budget")
        main_outer_steps = 2 * calibration_steps
        if iteration > start_iteration + main_outer_steps - 1:
            return "scene_tail"
    phase_index = (iteration - start_iteration) // phase_length
    return "calibration" if phase_index % 2 == 0 else "reconstruction"


def strict_block_step_counts(total_iterations, phase_length=8,
                             start_iteration=1, calibration_steps=None,
                             scene_steps=None):
    """Return exact optimizer-step counts for strict block training.

    The legacy call preserves the historical outer-loop accounting. The
    explicit budget form defines calibration and Scene steps independently;
    the alternating prefix is followed by a Scene-only tail.
    """
    if calibration_steps is not None or scene_steps is not None:
        if calibration_steps is None or scene_steps is None:
            raise ValueError("calibration_steps and scene_steps must be paired")
        if total_iterations != calibration_steps + scene_steps:
            raise ValueError(
                "total_iterations must equal calibration_steps + scene_steps")
        if calibration_steps <= 0 or scene_steps < calibration_steps:
            raise ValueError("invalid strict step budget")
        if calibration_steps % phase_length:
            raise ValueError("calibration_steps must be divisible by phase_length")
        return {
            "scene": int(scene_steps),
            "calibration": int(calibration_steps),
            "reconstruction": int(calibration_steps),
            "scene_tail": int(scene_steps - calibration_steps),
            "outer": int(total_iterations),
        }

    counts = {"scene": 0, "calibration": 0}
    for iteration in range(1, total_iterations):
        phase = phase_for_iteration(
            iteration, phase_length=phase_length,
            start_iteration=start_iteration)
        if should_step_scene(False, phase):
            counts["scene"] += 1
        if phase in {"joint", "calibration"}:
            counts["calibration"] += 1
    return counts
