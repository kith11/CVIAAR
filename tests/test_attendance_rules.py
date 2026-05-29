import unittest
from datetime import date, datetime

from modules.attendance_rules import (
    AM_SESSION,
    PM_SESSION,
    classify_login_status,
    infer_event_type,
    is_working_day,
    normalize_session,
    resolve_session,
    session_absent_status,
)


class AttendanceRulesTests(unittest.TestCase):
    def test_resolve_session_uses_fixed_midday_split(self):
        self.assertEqual(resolve_session(datetime(2026, 3, 25, 11, 59)), AM_SESSION)
        self.assertEqual(resolve_session(datetime(2026, 3, 25, 12, 0)), PM_SESSION)

    def test_normalize_session_prefers_explicit_value(self):
        self.assertEqual(normalize_session(AM_SESSION, datetime(2026, 3, 25, 14, 0)), AM_SESSION)

    def test_event_type_and_absence_status_helpers(self):
        self.assertEqual(infer_event_type("On Time"), "login")
        self.assertEqual(infer_event_type("Logout"), "logout")
        self.assertEqual(infer_event_type("AM Absent"), "auto_absent")
        self.assertEqual(session_absent_status(AM_SESSION), "AM Absent")
        self.assertEqual(session_absent_status(PM_SESSION), "PM Absent")

    def test_classify_login_status_respects_grace(self):
        # Within the 15-minute grace window.
        self.assertEqual(
            classify_login_status(datetime(2026, 3, 25, 7, 40), "07:30", AM_SESSION, 15),
            "On Time",
        )
        # Past the grace window.
        self.assertEqual(
            classify_login_status(datetime(2026, 3, 25, 7, 50), "07:30", AM_SESSION, 15),
            "Late",
        )

    def test_pm_return_for_morning_schedule_is_not_late(self):
        # A morning-scheduled staffer checking in again in the afternoon is present,
        # not late, since there is no separate PM schedule.
        self.assertEqual(
            classify_login_status(datetime(2026, 3, 25, 13, 0), "07:30", PM_SESSION, 15),
            "On Time",
        )

    def test_pm_only_schedule_can_be_late(self):
        # A staffer whose scheduled start is itself in the afternoon is judged
        # against that PM start.
        self.assertEqual(
            classify_login_status(datetime(2026, 3, 25, 13, 30), "13:00", PM_SESSION, 15),
            "Late",
        )

    def test_is_working_day_skips_weekend(self):
        self.assertTrue(is_working_day(date(2026, 3, 25), {0, 1, 2, 3, 4}))  # Wednesday
        self.assertFalse(is_working_day(date(2026, 3, 28), {0, 1, 2, 3, 4}))  # Saturday
        self.assertFalse(is_working_day(date(2026, 3, 29), {0, 1, 2, 3, 4}))  # Sunday


if __name__ == "__main__":
    unittest.main()
