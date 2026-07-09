"""Discrete-time PID controller with anti-windup.

    u[t] = Kp·e[t]
         + Ki·(I[t-1] + e[t]·dt)
         + Kd·(e[t] - e[t-1]) / dt

Anti-windup: when ``u[t]`` saturates at ``output_min`` or
``output_max``, the integrator is held at its previous value rather
than accumulating further error.  Without this, a long-running
positive error with a low output clamp would lock the controller into
saturation forever (the "integrator windup" failure mode).

The controller is stateless across ``reset()``; nothing else mutates
its internal state.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PidGains(BaseModel):
    """Proportional / Integral / Derivative gains."""

    model_config = ConfigDict(extra="forbid")

    kp: float = Field(default=1.0)
    ki: float = Field(default=0.0, ge=0.0)
    kd: float = Field(default=0.0, ge=0.0)


class PidController:
    """Discrete-time PID with output clamping + anti-windup."""

    def __init__(
        self,
        gains: PidGains,
        output_min: float,
        output_max: float,
    ) -> None:
        if output_min >= output_max:
            raise ValueError("output_min must be < output_max")
        self._gains = gains
        self._out_min = float(output_min)
        self._out_max = float(output_max)
        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        """Zero integrator + previous-error state."""
        self._integral = 0.0
        self._prev_error = None

    @property
    def integral(self) -> float:
        return self._integral

    @property
    def output_min(self) -> float:
        return self._out_min

    @property
    def output_max(self) -> float:
        return self._out_max

    def update(self, error: float, dt: float = 1.0) -> float:
        """Compute one PID step.  Returns the (clamped) control output."""
        if dt <= 0:
            raise ValueError("dt must be > 0")

        kp = self._gains.kp * error
        ki = self._gains.ki * (self._integral + error * dt)
        kd = 0.0 if self._prev_error is None else self._gains.kd * (error - self._prev_error) / dt

        unclamped = kp + ki + kd
        # Saturate.
        if unclamped > self._out_max:
            output = self._out_max
        elif unclamped < self._out_min:
            output = self._out_min
        else:
            output = unclamped
            # Only accumulate integrator when output is in the linear region.
            self._integral += error * dt

        self._prev_error = error
        return output


__all__ = ["PidController", "PidGains"]