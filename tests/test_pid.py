import unittest
from tinypid import PID

class TestPID(unittest.TestCase):
    def test_output_with_no_manual_output(self):
        pid = PID(K_p=1.0, K_i=0.5, K_d=0.2, setpoint=10.0, dt=0.1)
        process_variable = 8.0
        expected_output = 6.1  # P = 1.0 * (10.0 - 8.0) = 2.0, I = 0.5 * (10.0 - 8.0) * 0.1 = 0.1, D = 0.2 * ((10.0 - 8.0) / 0.1) = 4
        output = pid(process_variable)
        self.assertAlmostEqual(output, expected_output)

    def test_output_with_manual_output(self):
        pid = PID(K_p=1.0, K_i=0.5, K_d=0.2, setpoint=10.0, dt=0.1)
        process_variable = 8.0
        manual_output = 5.0
        expected_output = manual_output  # Since manual_output is provided, the output should be equal to it
        output = pid(process_variable, manual_output=manual_output)
        self.assertAlmostEqual(output, expected_output)

    def test_output_with_anti_windup_disabled(self):
        pid = PID(K_p=1.0, K_i=0.5, K_d=0.2, setpoint=10.0, dt=0.1)
        process_variable = 8.0
        manual_output = 15.0
        expected_output = manual_output  # Since manual_output is provided, the output should be equal to it
        output = pid(process_variable, manual_output=manual_output, anti_windup=False)
        self.assertAlmostEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()