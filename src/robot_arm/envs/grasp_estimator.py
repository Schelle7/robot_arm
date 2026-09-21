class GraspEstimator:
    def __init__(self, config):
        self.min_position, self.max_position = config.position_range_radians
        self.min_closing_duty = config.min_closing_duty
        self.max_velocity = config.max_velocity_radians_per_second
        self.hold_duration_ns = config.hold_seconds * 1_000_000_000
        self.reset()

    def reset(self):
        self.contact_candidate = False
        self.candidate_started_ns = 0

    def update(self, position, velocity, duty, sample_time_ns) -> bool:
        candidate = self.min_position <= position <= self.max_position and duty <= -self.min_closing_duty and abs(velocity) <= self.max_velocity
        if not candidate:
            self.reset()
            return False
        if not self.contact_candidate:
            self.contact_candidate = True
            self.candidate_started_ns = sample_time_ns
        return bool(sample_time_ns - self.candidate_started_ns >= self.hold_duration_ns)
