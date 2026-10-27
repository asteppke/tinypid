import unittest
from tinypid import PID, Gain, PIDGainScheduler, FeedforwardMixin, RateLimitMixin, DeadbandMixin


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


class FeedforwardPID(FeedforwardMixin, PID):
    pass


class FeedforwardGainSchedulerPID(FeedforwardMixin, PIDGainScheduler):
    pass


class TestFeedforwardMixin(unittest.TestCase):
    def test_feedforward_added_to_output(self):
        # Pure proportional controller at error=0; feedforward should be the entire output.
        pid = FeedforwardPID(k_p=1.0, k_i=0.0, k_d=0.0, setpoint=5.0, k_ff=1.0)
        output = pid(5.0, feedforward=3.0)
        self.assertAlmostEqual(output, 3.0)

    def test_feedforward_scaled_by_k_ff(self):
        # With k_ff=2 and feedforward=3 the FF contribution should be 6.
        pid = FeedforwardPID(k_p=0.0, k_i=0.0, k_d=0.0, setpoint=0.0, k_ff=2.0)
        output = pid(0.0, feedforward=3.0)
        self.assertAlmostEqual(output, 6.0)
        self.assertAlmostEqual(pid.FF, 6.0)

    def test_feedforward_zero_by_default(self):
        # Omitting feedforward arg must behave identically to a plain PID call.
        plain = PID(k_p=1.0, k_i=0.5, k_d=0.0, setpoint=10.0, dt=1.0)
        ff_pid = FeedforwardPID(k_p=1.0, k_i=0.5, k_d=0.0, setpoint=10.0, dt=1.0, k_ff=1.0)
        self.assertAlmostEqual(ff_pid(8.0), plain(8.0))

    def test_feedforward_applied_before_limiting(self):
        # Output limit is 4; PID alone (P=2) + FF (3) = 5 → clamped to 4.
        pid = FeedforwardPID(k_p=1.0, k_i=0.0, k_d=0.0, setpoint=2.0, upper_limit=4.0, k_ff=1.0)
        output = pid(0.0, feedforward=3.0)
        self.assertAlmostEqual(output, 4.0)

    def test_feedforward_composes_with_gain_scheduler(self):
        gains = [Gain(setpoint_scope=(0, 20), k_p=1.0, k_i=0.0, k_d=0.0)]
        pid = FeedforwardGainSchedulerPID(gains=gains, setpoint=10.0, k_ff=1.0)
        # error = 10 - 8 = 2, P = 2, FF = 1 * 3 = 3 → output = 5
        output = pid(8.0, feedforward=3.0)
        self.assertAlmostEqual(output, 5.0)


class RateLimitedPID(RateLimitMixin, PID):
    pass


class RateLimitedGainSchedulerPID(RateLimitMixin, PIDGainScheduler):
    pass


class TestRateLimitMixin(unittest.TestCase):
    def test_rate_limit_rejects_non_positive_max_output_rate(self):
        with self.assertRaises(ValueError):
            RateLimitedPID(k_p=1.0, k_i=0.0, k_d=0.0, max_output_rate=0.0)

    def test_rate_limit_restricts_output_change(self):
        # First call (error=0) establishes previous_output=0. Second call (error=10)
        # wants P = 10 * 10 = 100, but max change per step is 2.0 * dt(1.0) = 2.0.
        pid = RateLimitedPID(k_p=10.0, k_i=0.0, k_d=0.0, setpoint=10.0, dt=1.0, max_output_rate=2.0)
        pid(10.0)  # error = 0, output = 0
        output = pid(0.0)  # error = 10, raw output = 100, limited to 0 + 2 = 2
        self.assertAlmostEqual(output, 2.0)

    def test_rate_limit_disabled_by_default(self):
        # Without max_output_rate the mixin must behave identically to a plain PID.
        plain = PID(k_p=1.0, k_i=0.5, k_d=0.0, setpoint=10.0, dt=1.0)
        rate_limited = RateLimitedPID(k_p=1.0, k_i=0.5, k_d=0.0, setpoint=10.0, dt=1.0)
        self.assertAlmostEqual(rate_limited(0.0), plain(0.0))
        self.assertAlmostEqual(rate_limited(0.0), plain(0.0))

    def test_manual_output_bypasses_rate_limit(self):
        # A manual override must take effect immediately, ignoring max_output_rate.
        pid = RateLimitedPID(k_p=1.0, k_i=0.0, k_d=0.0, setpoint=10.0, dt=1.0, max_output_rate=0.1)
        pid(10.0)  # previous_output = 0.0
        output = pid(10.0, manual_output=50.0)
        self.assertAlmostEqual(output, 50.0)

    def test_anti_windup_prevents_integral_growth_under_rate_saturation(self):
        pid = RateLimitedPID(k_p=0.0, k_i=1.0, k_d=0.0, setpoint=10.0, dt=1.0, max_output_rate=1.0)
        pid(0.0)  # previous_output is None: not rate limited; integral = 10
        pid(0.0)  # rate saturated; integral rewound
        integral_after_first_saturation = pid.integral
        pid(0.0)  # rate saturated again
        self.assertAlmostEqual(pid.integral, integral_after_first_saturation)

    def test_anti_windup_disabled_allows_integral_growth_under_rate_saturation(self):
        pid = RateLimitedPID(k_p=0.0, k_i=1.0, k_d=0.0, setpoint=10.0, dt=1.0, max_output_rate=1.0)
        pid(0.0, anti_windup=False)
        pid(0.0, anti_windup=False)
        integral_after_first_saturation = pid.integral
        pid(0.0, anti_windup=False)
        self.assertGreater(pid.integral, integral_after_first_saturation)

    def test_reset_clears_previous_output(self):
        pid = RateLimitedPID(k_p=1.0, k_i=0.0, k_d=0.0, setpoint=10.0, dt=1.0, max_output_rate=0.1)
        pid(0.0)
        pid.reset()
        self.assertIsNone(pid._previous_output)

    def test_rate_limit_composes_with_value_limit(self):
        # Value limit of 50 is stricter than what the rate limit alone would allow (0 + 2 = 2).
        pid = RateLimitedPID(
            k_p=10.0, k_i=0.0, k_d=0.0, setpoint=10.0, dt=1.0, max_output_rate=2.0, upper_limit=1.0
        )
        output = pid(0.0)  # P = 100, value-limited to 1.0 first; previous_output is None, so no rate limit yet
        self.assertAlmostEqual(output, 1.0)

    def test_rate_limit_composes_with_gain_scheduler(self):
        gains = [Gain(setpoint_scope=(0, 20), k_p=10.0, k_i=0.0, k_d=0.0)]
        pid = RateLimitedGainSchedulerPID(gains=gains, setpoint=10.0, dt=1.0, max_output_rate=2.0)
        pid(10.0)  # error = 0, output = 0
        output = pid(0.0)  # error = 10, raw output = 100, limited to 0 + 2 = 2
        self.assertAlmostEqual(output, 2.0)


class DeadbandPID(DeadbandMixin, PID):
    pass


class DeadbandGainSchedulerPID(DeadbandMixin, PIDGainScheduler):
    pass


class TestDeadbandMixin(unittest.TestCase):
    def test_deadband_rejects_negative_output_deadband(self):
        with self.assertRaises(ValueError):
            DeadbandPID(k_p=1.0, k_i=0.0, k_d=0.0, output_deadband=-0.1)

    def test_deadband_holds_small_output_change(self):
        # First call (error=0) establishes previous_output=0. Second call's raw
        # output (0.5) is within output_deadband=1.0 of it, so it's held at 0.
        pid = DeadbandPID(k_p=1.0, k_i=0.0, k_d=0.0, setpoint=10.0, dt=1.0, output_deadband=1.0)
        pid(10.0)  # error = 0, output = 0
        output = pid(9.5)  # error = 0.5, raw output = 0.5, within deadband -> held
        self.assertAlmostEqual(output, 0.0)

    def test_deadband_passes_through_large_output_change(self):
        pid = DeadbandPID(k_p=1.0, k_i=0.0, k_d=0.0, setpoint=10.0, dt=1.0, output_deadband=1.0)
        pid(10.0)  # error = 0, output = 0
        output = pid(0.0)  # error = 10, raw output = 10, exceeds deadband -> passes through
        self.assertAlmostEqual(output, 10.0)

    def test_deadband_disabled_by_default(self):
        # Without output_deadband the mixin must behave identically to a plain PID.
        plain = PID(k_p=1.0, k_i=0.5, k_d=0.0, setpoint=10.0, dt=1.0)
        deadband = DeadbandPID(k_p=1.0, k_i=0.5, k_d=0.0, setpoint=10.0, dt=1.0)
        self.assertAlmostEqual(deadband(8.0), plain(8.0))
        self.assertAlmostEqual(deadband(8.0), plain(8.0))

    def test_manual_output_is_subject_to_deadband(self):
        # Per design, a manual override is not exempt from the dead-band.
        pid = DeadbandPID(k_p=1.0, k_i=0.0, k_d=0.0, setpoint=10.0, dt=1.0, output_deadband=1.0)
        pid(10.0)  # error = 0, output = 0
        output = pid(10.0, manual_output=0.5)  # within epsilon of previous output 0 -> held
        self.assertAlmostEqual(output, 0.0)

    def test_deadband_does_not_affect_integral(self):
        # The integral keeps accumulating normally even while the output is held.
        pid = DeadbandPID(k_p=0.0, k_i=1.0, k_d=0.0, setpoint=10.0, dt=1.0, output_deadband=100.0)
        pid(0.0)  # error = 10, integral = 10, output = 10 (no previous_output yet, passes through)
        integral_after_first = pid.integral
        pid(0.0)  # error = 10, integral = 20, raw output = 20, held at 10 due to deadband
        self.assertGreater(pid.integral, integral_after_first)

    def test_reset_clears_previous_output(self):
        pid = DeadbandPID(k_p=1.0, k_i=0.0, k_d=0.0, setpoint=10.0, dt=1.0, output_deadband=1.0)
        pid(0.0)
        pid.reset()
        self.assertIsNone(pid._previous_output)

    def test_deadband_composes_with_gain_scheduler(self):
        gains = [Gain(setpoint_scope=(0, 20), k_p=1.0, k_i=0.0, k_d=0.0)]
        pid = DeadbandGainSchedulerPID(gains=gains, setpoint=10.0, dt=1.0, output_deadband=1.0)
        pid(10.0)  # error = 0, output = 0
        output = pid(9.5)  # error = 0.5, raw output = 0.5, within deadband -> held
        self.assertAlmostEqual(output, 0.0)


if __name__ == "__main__":
    unittest.main()
