from typing import Optional, Tuple


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
        K_p: float = 1,
        K_i: float = 0.1,
        K_d: float = 0,
        setpoint: float = 0,
        dt: float = 1,
        derivative_lowpass: float = 1,
        upper_limit: Optional[float] = None,
        lower_limit: Optional[float] = None,
    ) -> None:
        """
        Initialize PID controller

        Parameters:
            K_p : Proportional gain
            K_i : Integral gain
            K_d : Derivative gain
            dt : Time step
            derivative_lowpass: lowpass constant (between 1 and 0, 1 meaning no lowpass)
            upper_limit : Upper limit for the output
            lower_limit : Lower limit for the output
        """
        if dt <= 0:
            raise ValueError("Time step (dt) must be positive.")

        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.P, self.I, self.D = None, None, None
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
        Clear the history.
        """
        self.integral = 0
        self._previous_error = 0

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
        self._previous_error = 0

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

    def __call__(self, process_variable : float, manual_output: Optional[float] = None, anti_windup: bool = True) -> float:
        """
        Process the input signal and return the controller output.

        Parameters:
            manual_output: Manually controlled output value (optional).
            anti_windup : Whether to enable anti-windup mechanism.
        """

        error = self._setpoint - process_variable
        self.integral += error * self.dt
        derivative = (error - self._previous_error) / self.dt if self.dt != 0 else 0

        self.P = self.K_p * error
        self.I = self.K_i * self.integral
        self.D = self.K_d * (
            self.alpha * derivative + (1 - self.alpha) * self._previous_derivative
        )

        output = self.P + self.I + self.D

        self._previous_error = error
        self._previous_derivative = derivative

        saturated, output = self.limit(output)

        if saturated and anti_windup:
            # Don't increase integral if we are saturated
            self.integral -= error * self.dt

        if manual_output:
            # Use setpoint tracking by calculating integral so that the output matches the manual setpoint
            self.integral = -(self.P + self.D - manual_output) / self.K_i
            output = manual_output

        return output

    def __repr__(self):
        return (f"PID controller\nSetpoint: {self.setpoint}, Output: {self.P + self.I + self.D}\n"
                f"P: {self.P}, I: {self.I}, D: {self.D}\n"\
                f"Limits: {self.lower_limit} < output < {self.upper_limit}")
