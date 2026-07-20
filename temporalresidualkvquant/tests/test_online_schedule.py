from unittest import TestCase

from trq.online_schedule import (
    OnlineQuantizationSchedule,
    QuantizationEvent,
    global_span_to_local,
)


class OnlineQuantizationScheduleTests(TestCase):
    def test_bulk_schedule_preserves_existing_24_frame_behavior(self):
        schedule = OnlineQuantizationSchedule()

        self.assertIsNone(schedule.event_at(21))
        first = schedule.event_at(24)
        second = schedule.event_at(48)

        self.assertEqual((first.global_start_frame, first.global_end_frame), (0, 24))
        self.assertEqual((second.global_start_frame, second.global_end_frame), (24, 48))

    def test_delayed_bulk_converts_only_nearest_interval(self):
        schedule = OnlineQuantizationSchedule(first_quant_frame=48)

        self.assertIsNone(schedule.event_at(24))
        event = schedule.event_at(48)

        self.assertEqual((event.global_start_frame, event.global_end_frame), (24, 48))

    def test_protected_sink_moves_first_actual_event(self):
        schedule = OnlineQuantizationSchedule(protected_sink_frames=24)

        self.assertIsNone(schedule.event_at(24))
        event = schedule.event_at(48)

        self.assertEqual((event.global_start_frame, event.global_end_frame), (24, 48))

    def test_gradual_schedule_converts_three_frames_per_block(self):
        schedule = OnlineQuantizationSchedule(schedule="gradual", gradual_frames=3)

        events = [schedule.event_at(frame) for frame in (24, 27, 30)]

        self.assertEqual(
            [(event.global_start_frame, event.global_end_frame) for event in events],
            [(0, 3), (3, 6), (6, 9)],
        )

    def test_global_span_maps_into_rolling_tail_after_eviction(self):
        event = QuantizationEvent(186, 159, 162, "gradual")

        local = global_span_to_local(
            event,
            completed_frames=186,
            local_frames=180,
        )

        self.assertEqual(local, (153, 156))

    def test_global_span_maps_around_attention_sink(self):
        event = QuantizationEvent(200, 176, 200, "bulk")

        local = global_span_to_local(
            event,
            completed_frames=200,
            local_frames=180,
            attention_sink_frames=24,
        )

        self.assertEqual(local, (156, 180))
