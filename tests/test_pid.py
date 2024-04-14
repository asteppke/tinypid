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

    def test_output_with_anti_windup_disabled(self):
        pid = PID(k_p=1.0, k_i=0.5, k_d=0.2, setpoint=10.0, dt=0.1)
        process_variable = 8.0
        manual_output = 15.0
        expected_output = manual_output  # Since manual_output is provided, the output should be equal to it
        output = pid(process_variable, manual_output=manual_output, anti_windup=False)
        self.assertAlmostEqual(output, expected_output)


class TestGainSchedule(unittest.TestCase):
    def test_gain_scheduler_basic(self):
        gains = [
            Gain(setpoint_scope=(0, 20), k_p=1.0, k_i=0.5, k_d=0.2),
            Gain(setpoint_scope=(20, 30), k_p=2.0, k_i=1.0, k_d=0.4),
        ]

        pid = PIDGainScheduler(gains=gains, setpoint=10)

        pid.update_gain(10)
        self.assertAlmostEqual(pid.k_p, 1.0)
        self.assertAlmostEqual(pid.k_i, 0.5)
        self.assertAlmostEqual(pid.k_d, 0.2)

        pid.update_gain(25)
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
            pid.update_gain(40)


if __name__ == "__main__":
    unittest.main()
