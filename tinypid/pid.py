"""PID controller

A minimal PID controller.

Example usage:
import tinypid

controller = tinypid.PID()

output = controller(10)

"""

from typing import List, Optional, Tuple


class Gain:
    """
    A simple class to store PID gains and the setpoint range for which they apply.
    """

    def __init__(self, setpoint_scope: Tuple[float, float], k_p: float, k_i: float, k_d: float) -> None:
        """
        Initializes a Gain object with a setpoint range and PID gains.

        Parameters:
            setpoint_scope: The range of setpoints for which these gains apply.
            The range is inclusive of the lower bound and exclusive of the upper bound.
            k_p : The proportional gain.
            k_i : The integral gain.
            k_d : The derivative gain.
        """
        self.setpoint_scope = setpoint_scope
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d


class PID:
    """
    A simple Proportional-Integral-Derivative controller
    with optional
        - output limiting
        - anti-windup mechanism
        - lowpass filtering of derivative component
        - bumpless transfer
    """

    def __init__(
        self,
        k_p: float = 1,
        k_i: float = 0.1,
        k_d: float = 0,
        setpoint: float = 0,
        dt: float = 1,
        derivative_lowpass: float = 1,
        upper_limit: Optional[float] = None,
        lower_limit: Optional[float] = None,
    ) -> None:
        """
        Initialize PID controller

        Parameters:
            k_p : Proportional gain
            k_i : Integral gain
            k_d : Derivative gain
            dt : Time step
            derivative_lowpass: lowpass constant (between 1 and 0, 1 meaning no lowpass)
            upper_limit : Upper limit for the output
            lower_limit : Lower limit for the output
        """
        if dt <= 0:
            raise ValueError("Time step (dt) must be positive.")
        if not 0 <= derivative_lowpass <= 1:
            raise ValueError("derivative_lowpass must be between 0 and 1")

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.P, self.I, self.D = 0.0, 0.0, 0.0
        self.dt = dt
        self.alpha = derivative_lowpass
        self._setpoint = setpoint
        self.integral = 0
        self._previous_error = 0
        self._previous_derivative = 0
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def reset(self) -> None:
        """
        Clear the history and reset the controller state.
        """
        self.integral = 0.0
        self._previous_error = 0.0
        self._previous_derivative = 0.0

    @property
    def setpoint(self) -> float:
        """
        Get the setpoint.
        """
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value: float) -> None:
        """
        Set the setpoint.

        Parameters:
            value : The new setpoint.
        """
        self._setpoint = value
        self._previous_error = 0.0
        self._previous_derivative = 0.0

    def limit(self, output: float) -> Tuple[bool, float]:
        """
        Limits the output to the specified bounds.

        Parameters:
            output : The given output.

        Returns:
            tuple: A tuple containing a boolean indicating whether saturation occurred
                   and the limited output.
        """
        unlimited = output
        if self.upper_limit is not None:
            output = min(output, self.upper_limit)
        if self.lower_limit is not None:
            output = max(output, self.lower_limit)
        saturated = output != unlimited

        return saturated, output

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
        manual_output: Optional[float] = None,
        anti_windup: bool = True,
    ) -> float:
        """
        Process the input signal and return the controller output.

        Parameters:
            manual_output: Manually controlled output value (optional).
            anti_windup : Whether to enable anti-windup mechanism.
        """

        error = self._setpoint - process_variable
        self.integral += error * self.dt
        derivative = (error - self._previous_error) / self.dt

        self.P = self.k_p * error
        self.I = self.k_i * self.integral
        self.D = self.k_d * (self.alpha * derivative + (1 - self.alpha) * self._previous_derivative)

        output = self.P + self.I + self.D

        self._previous_error = error
        self._previous_derivative = derivative

        saturated, output = self.limit(output)

        if saturated and anti_windup:
            # Don't increase integral if we are saturated
            self.integral -= error * self.dt

        if manual_output:
            # Use setpoint tracking by calculating integral so that the output matches the manual setpoint
            self.integral = -(self.P + self.D - manual_output) / self.k_i if self.k_i != 0 else 0
            output = manual_output

        return output

    def __repr__(self):
        return (
            f"PID controller\nSetpoint: {self.setpoint}, Output: {self.P + self.I + self.D}\n"
            f"P: {self.P}, I: {self.I}, D: {self.D}\n"
            f"Limits: {self.lower_limit} < output < {self.upper_limit}"
        )


class PIDGainScheduler(PID):
    """
    An extended Proportional-Integral-Derivative controller that uses
    gain scheduling to allow different gains, i.e., k_p, k_i, and k_d depending on
    the setpoint.

    """

    def __init__(
        self,
        gains: List[Gain],
        setpoint: float = 0,
        dt: float = 1,
        derivative_lowpass: float = 1,
        upper_limit: Optional[float] = None,
        lower_limit: Optional[float] = None,
    ) -> None:
        """
        Initialize PID controller

        Parameters:
            gains : List of Gain objects
            setpoint : The setpoint
            dt : Time step
            derivative_lowpass: lowpass constant (between 1 and 0, 1 meaning no lowpass)
            upper_limit : Upper limit for the output
            lower_limit : Lower limit for the output
        """

        super().__init__(None, None, None, setpoint, dt, derivative_lowpass, upper_limit, lower_limit)

        self.gains = gains
        self.update_gain(setpoint)

    def update_gain(self, setpoint: float) -> None:
        """
        Update the PID gains based on the current setpoint

        Parameters:
            output : The current output
            setpoint : The current setpoint to find gains for
        """
        for gain in self.gains:
            lower, upper = gain.setpoint_scope
            if lower <= setpoint < upper:
                self.k_p, self.k_i, self.k_d = gain.k_p, gain.k_i, gain.k_d
                break
        else:
            raise ValueError("No gain found for the given setpoint.")

    def __call__(
        self,
        process_variable: float,
        manual_output: Optional[float] = None,
        anti_windup: bool = True,
    ) -> float:
        self.update_gain(self.setpoint)

        output = super().__call__(process_variable, manual_output, anti_windup)
        return output
