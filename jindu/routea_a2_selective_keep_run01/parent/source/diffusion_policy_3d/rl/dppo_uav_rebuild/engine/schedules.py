"""Explicit, checkpointable scalar schedules without automatic damping."""

from dataclasses import dataclass


@dataclass
class LinearSchedule:
    start: float
    end: float
    total_steps: int
    step_index: int = 0

    @property
    def value(self):
        fraction = min(max(self.step_index / max(self.total_steps, 1), 0.0), 1.0)
        return self.start + fraction * (self.end - self.start)

    def step(self):
        self.step_index += 1

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        if (state["start"], state["end"], state["total_steps"]) != (
            self.start,
            self.end,
            self.total_steps,
        ):
            raise ValueError("schedule identity mismatch")
        self.step_index = int(state["step_index"])

