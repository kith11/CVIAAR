import unittest
from datetime import datetime

from modules.attendance_rules import (
    AM_SESSION,
    PM_SESSION,
    infer_event_type,
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


if __name__ == "__main__":
    unittest.main()
