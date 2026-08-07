"""Frame-aligned online TRQ quantization schedules."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QuantizationEvent:
    boundary_frame: int
    global_start_frame: int
    global_end_frame: int
    schedule: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class OnlineQuantizationSchedule:
    """Resolve bulk or gradual quantization spans in global frame space."""

    def __init__(
        self,
        *,
        first_quant_frame: int = 24,
        interval_frames: int = 24,
        schedule: str = "bulk",
        gradual_frames: int = 3,
        protected_sink_frames: int = 0,
    ) -> None:
        if first_quant_frame <= 0:
            raise ValueError("first_quant_frame must be positive")
        if interval_frames <= 0:
            raise ValueError("interval_frames must be positive")
        if schedule not in {"bulk", "gradual"}:
            raise ValueError(f"schedule must be bulk or gradual, got {schedule!r}")
        if gradual_frames <= 0:
            raise ValueError("gradual_frames must be positive")
        if protected_sink_frames < 0:
            raise ValueError("protected_sink_frames must be non-negative")
        self.first_quant_frame = int(first_quant_frame)
        self.interval_frames = int(interval_frames)
        self.schedule = schedule
        self.gradual_frames = int(gradual_frames)
        self.protected_sink_frames = int(protected_sink_frames)
        self._gradual_cursor = max(
            self.protected_sink_frames,
            self.first_quant_frame - self.interval_frames,
        )

    def event_at(self, completed_frames: int) -> QuantizationEvent | None:
        """Return the span to convert before generating the next block."""
        completed_frames = int(completed_frames)
        if completed_frames < self.first_quant_frame:
            return None

        if self.schedule == "bulk":
            offset = completed_frames - self.first_quant_frame
            if offset % self.interval_frames != 0:
                return None
            start = max(
                self.protected_sink_frames,
                completed_frames - self.interval_frames,
            )
            end = completed_frames
        else:
            if self._gradual_cursor >= completed_frames:
                return None
            start = self._gradual_cursor
            end = min(start + self.gradual_frames, completed_frames)
            self._gradual_cursor = end

        if start >= end:
            return None
        return QuantizationEvent(
            boundary_frame=completed_frames,
            global_start_frame=start,
            global_end_frame=end,
            schedule=self.schedule,
        )


def global_span_to_local(
    event: QuantizationEvent,
    *,
    completed_frames: int,
    local_frames: int,
    attention_sink_frames: int = 0,
) -> tuple[int, int]:
    """Map a global event into the compacted Self-Forcing cache window."""
    if local_frames <= 0 or completed_frames < local_frames:
        raise ValueError(
            f"invalid cache frame counts: completed={completed_frames}, local={local_frames}"
        )
    if not 0 <= attention_sink_frames <= local_frames:
        raise ValueError(
            f"attention_sink_frames must be in [0, {local_frames}], got {attention_sink_frames}"
        )

    sink = int(attention_sink_frames)
    tail_frames = local_frames - sink
    tail_global_start = completed_frames - tail_frames

    def map_frame(frame: int) -> int:
        if frame <= sink:
            return frame
        if frame < tail_global_start:
            raise ValueError(
                f"global frame {frame} has already been evicted; live tail starts at {tail_global_start}"
            )
        return sink + frame - tail_global_start

    local_start = map_frame(event.global_start_frame)
    local_end = map_frame(event.global_end_frame)
    if not 0 <= local_start < local_end <= local_frames:
        raise ValueError(
            f"mapped event is outside live cache: [{local_start}, {local_end}) vs {local_frames}"
        )
    return local_start, local_end
