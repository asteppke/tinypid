import unittest
from tinypid import PID, Gain, PIDGainScheduler


class TestPID(unittest.TestCase):
    def test_output_with_no_manual_output(self):
        pid = PID(k_p=1.0, k_i=0.5, k_d=0.2, setpoint=10.0, dt=0.1)
        process_variable = 8.0
        expected_output = 6.1  # P = 1.0 * (10.0 - 8.0) = 2.0, I = 0.5 * (10.0 - 8.0) * 0.1 = 0.1, D = 0.2 * ((10.0 - 8.0) / 0.1) = 4
        output = pid(process_variable)
        self.assertAlmostEqual(output, expected_output)

    def test_output_with_manual_output(self):
        pid = PID(k_p=1.0, k_i=0.5, k_d=0.2, setpoint=10.0, dt=0.1)
        process_variable = 8.0
        manual_output = 5.0
        expected_output = manual_output  # Since manual_output is provided, the output should be equal to it
        output = pid(process_variable, manual_output=manual_output)
        self.assertAlmostEqual(output, expected_output)

    def test_manual_output_with_anti_windup_disabled(self):
        pid = PID(k_p=1.0, k_i=0.5, k_d=0.2, setpoint=10.0, dt=0.1)
        process_variable = 8.0
        manual_output = 15.0
        expected_output = manual_output  # Since manual_output is provided, the output should be equal to it
        output = pid(process_variable, manual_output=manual_output, anti_windup=False)
        self.assertAlmostEqual(output, expected_output)

    def test_manual_output_zero(self):
        # manual_output=0 must be respected (falsy value bug guard)
        pid = PID(k_p=1.0, k_i=0.5, k_d=0.2, setpoint=10.0, dt=0.1)
        output = pid(8.0, manual_output=0.0)
        self.assertAlmostEqual(output, 0.0)

    def test_manual_output_sets_tracking_integral(self):
        # After a manual override the integral is back-calculated so the next
        # automatic output starts from the manual value (bumpless transfer).
        # k_d=0 to isolate: integral = (manual_output - P) / k_i = (5 - 2) / 0.5 = 6.0
        pid = PID(k_p=1.0, k_i=0.5, k_d=0.0, setpoint=10.0, dt=0.1)
        pid(8.0, manual_output=5.0)
        self.assertAlmostEqual(pid.integral, 6.0)

    def test_upper_limit(self):
        # Large error produces output >> upper_limit; result must be clamped.
        pid = PID(k_p=10.0, k_i=0.0, k_d=0.0, setpoint=10.0, upper_limit=5.0)
        output = pid(0.0)  # P = 10 * 10 = 100, clamped to 5
        self.assertAlmostEqual(output, 5.0)

    def test_lower_limit(self):
        # Negative error produces output << lower_limit; result must be clamped.
        pid = PID(k_p=10.0, k_i=0.0, k_d=0.0, setpoint=0.0, lower_limit=-5.0)
        output = pid(10.0)  # P = 10 * (-10) = -100, clamped to -5
        self.assertAlmostEqual(output, -5.0)

    def test_anti_windup_prevents_integral_growth(self):
        # When the output is saturated and anti_windup is enabled, the integral
        # must not accumulate between successive calls.
        pid = PID(k_p=1.0, k_i=1.0, k_d=0.0, setpoint=10.0, dt=1.0, upper_limit=5.0)
        pid(0.0)  # saturated; integral rewound to 0
        integral_after_first = pid.integral
        pid(0.0)  # saturated again
        self.assertAlmostEqual(pid.integral, integral_after_first)

    def test_anti_windup_disabled_allows_integral_growth(self):
        # Without anti_windup the integral keeps growing even under saturation.
        pid = PID(k_p=1.0, k_i=1.0, k_d=0.0, setpoint=10.0, dt=1.0, upper_limit=5.0)
        pid(0.0, anti_windup=False)
        integral_after_first = pid.integral
        pid(0.0, anti_windup=False)
        self.assertGreater(pid.integral, integral_after_first)

    def test_reset_clears_state(self):
        pid = PID(k_p=1.0, k_i=0.5, k_d=0.2, setpoint=10.0, dt=0.1)
        pid(8.0)  # build up integral and derivative history
        pid.reset()
        self.assertAlmostEqual(pid.integral, 0.0)
        self.assertAlmostEqual(pid._previous_error, 0.0)
        self.assertAlmostEqual(pid._previous_derivative, 0.0)


class TestGainSchedule(unittest.TestCase):
    def test_gain_scheduler_basic(self):
        gains = [
            Gain(setpoint_scope=(0, 20), k_p=1.0, k_i=0.5, k_d=0.2),
            Gain(setpoint_scope=(20, 30), k_p=2.0, k_i=1.0, k_d=0.4),
        ]

        pid = PIDGainScheduler(gains=gains, setpoint=10)

        pid.update_gains_from_setpoint(10)
        self.assertAlmostEqual(pid.k_p, 1.0)
        self.assertAlmostEqual(pid.k_i, 0.5)
        self.assertAlmostEqual(pid.k_d, 0.2)

        pid.update_gains_from_setpoint(25)
        self.assertAlmostEqual(pid.k_p, 2.0)
        self.assertAlmostEqual(pid.k_i, 1.0)
        self.assertAlmostEqual(pid.k_d, 0.4)

    def test_gain_scheduler_constant(self):
        gains = [
            Gain(setpoint_scope=(0, 20), k_p=1.0, k_i=0.5, k_d=0.2),
            Gain(setpoint_scope=(20, 30), k_p=2.0, k_i=1.0, k_d=0.4),
        ]

        pid = PIDGainScheduler(gains=gains, setpoint=10, dt=0.1)

        process_variable = 8.0
        expected_output = 6.1  # P = 1.0 * (10.0 - 8.0) = 2.0, I = 0.5 * (10.0 - 8.0) * 0.1 = 0.1, D = 0.2 * ((10.0 - 8.0) / 0.1) = 4
        output = pid(process_variable)
        self.assertAlmostEqual(output, expected_output)

    def test_gain_scheduler_no_gain_found(self):
        gains = [
            Gain(setpoint_scope=(0, 20), k_p=1.0, k_i=0.5, k_d=0.2),
            Gain(setpoint_scope=(20, 30), k_p=2.0, k_i=1.0, k_d=0.4),
        ]

        pid = PIDGainScheduler(gains=gains, setpoint=10)

        with self.assertRaises(ValueError):
            pid.update_gains_from_setpoint(40)


if __name__ == "__main__":
    unittest.main()
