"""PID controller

A minimal PID controller.

Example usage:
import tinypid

controller = tinypid.PID()

output = controller(10)

"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Gain:
    """
    A class to store PID gains and the setpoint range for which they apply.
    The range is inclusive of the lower bound and exclusive of the upper bound.
    """

    setpoint_scope: tuple[float, float]
    k_p: float
    k_i: float
    k_d: float


DEFAULT_ANTI_WINDUP = True


class PIDMinimal:
    """
    The minimal Proportional-Integral-Derivative calculation and state.
    """

    def __init__(
        self,
        k_p: float = 1.0,
        k_i: float = 0.1,
        k_d: float = 0.0,
        setpoint: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        """
        Initialize PID controller

        Parameters:
            k_p : Proportional gain
            k_i : Integral gain
            k_d : Derivative gain
            dt : Time step
            setpoint : Initial setpoint
        """
        if dt <= 0:
            raise ValueError("Time step (dt) must be positive.")

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.P, self.I, self.D = 0.0, 0.0, 0.0
        self.dt = dt
        self._setpoint = setpoint
        self.integral = 0.0
        self._previous_error = 0.0

    def reset(self) -> None:
        """
        Clear the history and reset the controller state.
        """
        self.integral = 0.0
        self._previous_error = 0.0

    @property
    def setpoint(self) -> float:
        """
        Get the setpoint.
        """
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value: float) -> None:
        """
        Set the setpoint and reset the controller state to avoid derivative kick and integral windup.

        Parameters:
            value : The new setpoint.
        """
        self._setpoint = value
        self._previous_error = 0.0

    def update_gains(self, k_p: float, k_i: float, k_d: float) -> None:
        """
        Update the PID gains.

        Parameters:
            k_p : The new proportional gain
            k_i : The new integral gain
            k_d : The new derivative gain
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def __call__(
        self,
        process_variable: float,
        manual_output: float | None = None,
    ) -> float:
        """
        Process the input signal and return the controller output.

        Parameters:
            process_variable : The current value of the process variable.
            manual_output: Manually controlled output value (optional).
        """

        error = self._setpoint - process_variable
        self.integral += error * self.dt
        derivative = (error - self._previous_error) / self.dt

        self._previous_error = error

        self.P = self.k_p * error
        self.I = self.k_i * self.integral
        self.D = self.k_d * derivative

        output = self.P + self.I + self.D

        if manual_output is not None:
            output = manual_output

        return output

    def __repr__(self) -> str:
        return (
            f"PID controller\nSetpoint: {self.setpoint}, Unlimited output: {self.P + self.I + self.D}\n"
            f"P: {self.P}, I: {self.I}, D: {self.D}"
        )


class PIDCore(PIDMinimal):
    """
    The shared Proportional-Integral-Derivative calculation and state.
    """

    def __init__(
        self,
        k_p: float = 1.0,
        k_i: float = 0.1,
        k_d: float = 0.0,
        setpoint: float = 0.0,
        dt: float = 1.0,
        derivative_lowpass: float = 1.0,
    ) -> None:
        """
        Initialize PID controller

        Parameters:
            k_p : Proportional gain
            k_i : Integral gain
            k_d : Derivative gain
            dt : Time step
            derivative_lowpass: lowpass constant (between 1 and 0, 1 meaning no lowpass)
        """
        if not 0 <= derivative_lowpass <= 1:
            raise ValueError("derivative_lowpass must be between 0 and 1")

        super().__init__(k_p, k_i, k_d, setpoint, dt)
        self.FF = 0.0
        self.alpha = derivative_lowpass
        self._previous_derivative = 0.0

    def reset(self) -> None:
        """
        Clear the history and reset the controller state.
        """
        super().reset()
        self._previous_derivative = 0.0

    @PIDMinimal.setpoint.setter
    def setpoint(self, value: float) -> None:
        """
        Set the setpoint and reset the controller state to avoid derivative kick and integral windup.

        Parameters:
            value : The new setpoint.
        """
        self._setpoint = value
        self._previous_error = 0.0
        self._previous_derivative = 0.0

    def _feedforward(self) -> float:
        """
        Compute the feedforward term. Returns 0.0 by default.
        Override in subclasses or via FeedforwardMixin.
        """
        return 0.0

    def _process_output(self, output: float, error: float, anti_windup: bool) -> float:
        """Hook for subclasses to apply output policies after PID calculation."""
        return output

    def _rewind_integral(self, error: float) -> None:
        """
        Undo one integration step to counteract windup.

        Parameters:
            error : The error value that was integrated in the step being undone.
        """
        self.integral -= error * self.dt

    def __call__(
        self,
        process_variable: float,
        manual_output: float | None = None,
        anti_windup: bool = DEFAULT_ANTI_WINDUP,
    ) -> float:
        """
        Process the input signal and return the controller output.

        Parameters:
            process_variable : The current value of the process variable.
            manual_output: Manually controlled output value (optional).
            anti_windup : Whether output saturation may rewind the integral.
        """

        error = self._setpoint - process_variable
        self.integral += error * self.dt
        derivative = (error - self._previous_error) / self.dt

        self.P = self.k_p * error
        self.I = self.k_i * self.integral
        self.D = self.k_d * (self.alpha * derivative + (1 - self.alpha) * self._previous_derivative)
        self.FF = self._feedforward()

        self._previous_error = error
        self._previous_derivative = derivative

        output = self.P + self.I + self.D + self.FF
        output = self._process_output(output, error, anti_windup)

        if manual_output is not None:
            # Use setpoint tracking by calculating integral so that the output matches the manual setpoint
            self.integral = -(self.P + self.D - manual_output) / self.k_i if self.k_i != 0 else 0
            output = manual_output

        return output

    def __repr__(self) -> str:
        return (
            f"PID controller\nSetpoint: {self.setpoint}, Unlimited output: {self.P + self.I + self.D + self.FF}\n"
            f"P: {self.P}, I: {self.I}, D: {self.D}, FF: {self.FF}"
        )


class LimitedPID(PIDCore):
    """PID core with absolute output limits and saturation anti-windup."""

    def __init__(
        self,
        k_p: float = 1.0,
        k_i: float = 0.1,
        k_d: float = 0.0,
        setpoint: float = 0.0,
        dt: float = 1.0,
        derivative_lowpass: float = 1.0,
        upper_limit: float | None = None,
        lower_limit: float | None = None,
    ) -> None:
        super().__init__(k_p, k_i, k_d, setpoint, dt, derivative_lowpass)
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def _limit(self, output: float) -> tuple[bool, float]:
        unlimited = output
        if self.upper_limit is not None:
            output = min(output, self.upper_limit)
        if self.lower_limit is not None:
            output = max(output, self.lower_limit)
        return output != unlimited, output

    def _process_output(self, output: float, error: float, anti_windup: bool) -> float:
        saturated, output = self._limit(output)
        if saturated and anti_windup:
            self._rewind_integral(error)
        return output

    def __repr__(self) -> str:
        return (
            f"PID controller\nSetpoint: {self.setpoint}, Unlimited output: {self.P + self.I + self.D + self.FF}\n"
            f"P: {self.P}, I: {self.I}, D: {self.D}, FF: {self.FF}\n"
            f"Limits: {self.lower_limit} < output < {self.upper_limit}"
        )


class FeedforwardMixin:
    """
    A mixin that adds a feedforward term to any PID-based controller.

    The feedforward value is passed directly to ``__call__`` at each update step.
    It is scaled by ``k_ff`` and added to the PID output *before* output limiting,
    so the anti-windup and saturation logic remain consistent.

    Usage::

        controller = FeedforwardPID(k_p=1.0, k_i=0.1, k_ff=0.5)
        output = controller(process_variable=3.0, feedforward=setpoint_velocity)

    A ready-made :class:`FeedforwardPID` is provided for this common case.
    :class:`PID` and :class:`PIDGainScheduler` already include this feature.
    For a custom controller based on :class:`LimitedPID`, subclass the mixin
    with that class::

        class CustomLimitedPID(FeedforwardMixin, LimitedPID):
            pass
    """

    def __init__(self, *args, k_ff: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k_ff = k_ff
        self._ff_value: float = 0.0

    def _feedforward(self) -> float:
        return self.k_ff * self._ff_value

    def __call__(
        self,
        process_variable: float,
        feedforward: float = 0.0,
        **kwargs,
    ) -> float:
        """
        Parameters:
            process_variable : The current value of the process variable.
            feedforward : The external feedforward signal (e.g. model output or
                          setpoint derivative). Scaled by ``k_ff`` before use.
            **kwargs : Passed through to the underlying PID ``__call__``
                       (``manual_output``, ``anti_windup``).
        """
        self._ff_value = feedforward
        return super().__call__(process_variable, **kwargs)  # type: ignore[misc]


class RateLimitMixin:
    """
    A mixin that restricts how fast the output may change per step
    (slew-rate limiting), independent of the absolute output limits
    (``upper_limit``/``lower_limit``).

    Rate limiting is applied *after* the underlying controller's own output
    (which may already be value-limited), comparing it against the output of
    the previous step. If the change exceeds ``max_output_rate * dt``, the
    output is clamped and, like value saturation, the integral is rewound to
    avoid windup. A ``manual_output`` override is passed through unchanged,
    since it represents an explicit, immediate operator command rather than
    the controller's own output.

    Usage::

        controller = RateLimitedPID(k_p=1.0, dt=0.1, max_output_rate=2.0)
        output = controller(process_variable=3.0)

    A ready-made :class:`RateLimitedPID` is provided for this common case.
    :class:`PID` and :class:`PIDGainScheduler` already include this feature.
    For a custom controller based on :class:`LimitedPID`, subclass the mixin
    with that class::

        class CustomRateLimitedPID(RateLimitMixin, LimitedPID):
            pass
    """

    def __init__(self, *args, max_output_rate: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if max_output_rate is not None and max_output_rate <= 0:
            raise ValueError("max_output_rate must be positive.")
        self.max_output_rate = max_output_rate
        self._rate_limited_output: float | None = None

    def reset(self) -> None:
        """
        Clear the history and reset the controller state, including the
        remembered previous output used for rate limiting.
        """
        super().reset()
        self._rate_limited_output = None

    def _rate_limit(self, output: float) -> tuple[bool, float]:
        """
        Limits how much the output may change relative to the previous step.

        Parameters:
            output : The candidate output for this step.

        Returns:
            tuple: A tuple containing a boolean indicating whether rate
                   saturation occurred and the rate-limited output.
        """
        if self.max_output_rate is None or self._rate_limited_output is None:
            return False, output

        max_delta = self.max_output_rate * self.dt
        delta = output - self._rate_limited_output
        limited_delta = min(max(delta, -max_delta), max_delta)

        return limited_delta != delta, self._rate_limited_output + limited_delta

    def __call__(
        self,
        process_variable: float,
        manual_output: float | None = None,
        anti_windup: bool = DEFAULT_ANTI_WINDUP,
        **kwargs,
    ) -> float:
        """
        Parameters:
            process_variable : The current value of the process variable.
            manual_output : Manually controlled output value (optional).
                             Not subject to rate limiting.
            anti_windup : Whether to enable anti-windup mechanism.
            **kwargs : Passed through to the underlying PID ``__call__``
                       (e.g. ``feedforward``).
        """
        output = super().__call__(  # type: ignore[misc]
            process_variable, manual_output=manual_output, anti_windup=anti_windup, **kwargs
        )

        if manual_output is None:
            saturated, output = self._rate_limit(output)
            if saturated and anti_windup:
                # Don't let the integral keep growing if the output can't follow due to the rate limit
                self._rewind_integral(self._previous_error)

        self._rate_limited_output = output
        return output


class DeadbandMixin:
    """
    A mixin that suppresses small output changes to reduce actuator chatter
    (e.g. motor noise near steady state).

    If the new output is within ``output_deadband`` of the previous output,
    the previous output is returned unchanged instead of the new value. This
    applies uniformly to both the controller's own output and a
    ``manual_output`` override, and it does not interact with anti-windup:
    the integral keeps evolving normally even while the output is held.

    Usage::

        controller = DeadbandPID(k_p=1.0, k_i=0.1, output_deadband=0.05)
        output = controller(process_variable=3.0)

    A ready-made :class:`DeadbandPID` is provided for this common case.
    :class:`PID` and :class:`PIDGainScheduler` already include this feature.
    For a custom controller based on :class:`LimitedPID`, subclass the mixin
    with that class::

        class CustomDeadbandPID(DeadbandMixin, LimitedPID):
            pass
    """

    def __init__(self, *args, output_deadband: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if output_deadband is not None and output_deadband < 0:
            raise ValueError("output_deadband must not be negative.")
        self.output_deadband = output_deadband
        self._deadband_output: float | None = None

    def reset(self) -> None:
        """
        Clear the history and reset the controller state, including the
        remembered previous output used by the dead-band.
        """
        super().reset()
        self._deadband_output = None

    def __call__(self, process_variable: float, **kwargs) -> float:
        """
        Parameters:
            process_variable : The current value of the process variable.
            **kwargs : Passed through to the underlying PID ``__call__``
                       (``manual_output``, ``anti_windup``, ``feedforward``).
        """
        output = super().__call__(process_variable, **kwargs)  # type: ignore[misc]

        if (
            self.output_deadband is not None
            and self._deadband_output is not None
            and abs(output - self._deadband_output) <= self.output_deadband
        ):
            return self._deadband_output

        self._deadband_output = output
        return output


class DitherMixin:
    """
    A mixin that adds a small pseudo-random perturbation (dither) to the
    output, e.g. to improve the effective resolution of a quantized
    actuator (a stepper motor, a digital output, ...) by randomizing the
    rounding error instead of letting it accumulate systematically.

    The dither value is drawn uniformly from
    ``[-output_dither / 2, output_dither / 2]`` and added to the output
    *after* all other processing (limiting, rate limiting, dead-band, ...),
    so it does not affect the controller's internal state (``integral``,
    anti-windup, etc.). Note that the dithered output may therefore exceed
    ``upper_limit``/``lower_limit`` by up to ``output_dither / 2``.

    Usage::

        controller = DitheredPID(k_p=1.0, k_i=0.1, output_dither=0.1)
        output = controller(process_variable=3.0)

    A ready-made :class:`DitheredPID` is provided for this common case.
    :class:`PID` and :class:`PIDGainScheduler` already include this feature.
    For a custom controller based on :class:`LimitedPID`, subclass the mixin
    with that class::

        class CustomDitheredPID(DitherMixin, LimitedPID):
            pass
    """

    def __init__(
        self, *args, output_dither: float | None = None, dither_seed: int | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if output_dither is not None and output_dither < 0:
            raise ValueError("output_dither must not be negative.")
        self.output_dither = output_dither
        self._dither_rng = random.Random(dither_seed)

    def __call__(self, process_variable: float, **kwargs) -> float:
        """
        Parameters:
            process_variable : The current value of the process variable.
            **kwargs : Passed through to the underlying PID ``__call__``
                       (``manual_output``, ``anti_windup``, ``feedforward``).
        """
        output = super().__call__(process_variable, **kwargs)  # type: ignore[misc]

        if self.output_dither:
            output += self._dither_rng.uniform(-self.output_dither / 2, self.output_dither / 2)

        return output


class FeedforwardPID(FeedforwardMixin, LimitedPID):
    """
    A ready-to-use PID controller with a feedforward component.
    See :class:`FeedforwardMixin` for details.
    """


class RateLimitedPID(RateLimitMixin, LimitedPID):
    """
    A ready-to-use PID controller with output rate limiting.
    See :class:`RateLimitMixin` for details.
    """


class DeadbandPID(DeadbandMixin, LimitedPID):
    """
    A ready-to-use PID controller with an output dead-band.
    See :class:`DeadbandMixin` for details.
    """


class DitheredPID(DitherMixin, LimitedPID):
    """
    A ready-to-use PID controller with output dithering.
    See :class:`DitherMixin` for details.
    """


class PID(DitherMixin, DeadbandMixin, RateLimitMixin, FeedforwardMixin, LimitedPID):
    """Full-featured PID controller with optional mixin features.

    Feedforward, rate limiting, dead-band, and dithering are disabled unless
    their corresponding constructor options are provided.
    """

    def __init__(
        self,
        k_p: float = 1.0,
        k_i: float = 0.1,
        k_d: float = 0.0,
        setpoint: float = 0.0,
        dt: float = 1.0,
        derivative_lowpass: float = 1.0,
        upper_limit: float | None = None,
        lower_limit: float | None = None,
        k_ff: float = 1.0,
        max_output_rate: float | None = None,
        output_deadband: float | None = None,
        output_dither: float | None = None,
        dither_seed: int | None = None,
    ) -> None:
        """
        Initialize a full-featured PID controller.

        Required/core parameters:
            k_p : Proportional gain
            k_i : Integral gain
            k_d : Derivative gain
            setpoint : Initial setpoint
            dt : Time step, must be positive

        Derivative filtering:
            derivative_lowpass : Lowpass constant for the derivative term,
                                  between 0 and 1 (1 means no filtering)

        Output limiting (optional; set either bound to enable):
            upper_limit : Maximum allowed output, or None to disable
            lower_limit : Minimum allowed output, or None to disable

        Feedforward (optional, see :class:`FeedforwardMixin`):
            k_ff : Gain applied to the ``feedforward`` argument of ``__call__``

        Rate limiting (optional, see :class:`RateLimitMixin`):
            max_output_rate : Maximum allowed change in output per time step,
                               or None to disable rate limiting

        Dead-band (optional, see :class:`DeadbandMixin`):
            output_deadband : Output changes smaller than this are suppressed,
                               or None to disable the dead-band

        Dithering (optional, see :class:`DitherMixin`):
            output_dither : Amplitude of the pseudo-random perturbation added
                             to the output, or None to disable dithering
            dither_seed : Seed for the dither random number generator
        """
        super().__init__(
            k_p,
            k_i,
            k_d,
            setpoint,
            dt,
            derivative_lowpass,
            upper_limit,
            lower_limit,
            k_ff=k_ff,
            max_output_rate=max_output_rate,
            output_deadband=output_deadband,
            output_dither=output_dither,
            dither_seed=dither_seed,
        )


class PIDGainScheduler(PID):
    """
    An extended PID controller that selects gains based on the setpoint.
    It supports the same optional features and constructor options as :class:`PID`.
    """

    def __init__(
        self,
        gains: list[Gain],
        setpoint: float = 0.0,
        dt: float = 1.0,
        derivative_lowpass: float = 1.0,
        upper_limit: float | None = None,
        lower_limit: float | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize a gain-scheduled PID controller.

        Required parameters:
            gains : List of :class:`Gain` entries, each defining ``k_p``,
                    ``k_i``, ``k_d`` and the setpoint range they apply to.
                    The ranges must cover the controller's operating setpoints
                    (inclusive of the lower bound, exclusive of the upper bound).
            setpoint : Initial setpoint, used to select the initial gains

        Core parameters:
            dt : Time step, must be positive
            derivative_lowpass : Lowpass constant for the derivative term,
                                  between 0 and 1 (1 means no filtering)

        Output limiting (optional; set either bound to enable):
            upper_limit : Maximum allowed output, or None to disable
            lower_limit : Minimum allowed output, or None to disable

        Optional feature parameters (see :class:`PID`):
            **kwargs : ``k_ff``, ``max_output_rate``, ``output_deadband``,
                       ``output_dither``, ``dither_seed``
        """
        super().__init__(0.0, 0.0, 0.0, setpoint, dt, derivative_lowpass, upper_limit, lower_limit, **kwargs)

        self.gains = gains
        self.update_gains_from_setpoint(setpoint)

    def update_gains_from_setpoint(self, setpoint: float) -> None:
        """Update the PID gains to match the supplied setpoint."""
        for gain in self.gains:
            lower, upper = gain.setpoint_scope
            if lower <= setpoint < upper:
                self.k_p, self.k_i, self.k_d = gain.k_p, gain.k_i, gain.k_d
                break
        else:
            raise ValueError("No gain found for the given setpoint.")

    def __call__(self, process_variable: float, **kwargs) -> float:
        self.update_gains_from_setpoint(self.setpoint)
        return super().__call__(process_variable, **kwargs)
