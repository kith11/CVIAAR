import os
import tempfile
import unittest
from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.analytics_engine import AnalyticsEngine
from modules.models import Attendance, Base, User, ensure_application_schema


class AnalyticsEngineTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tempdir.name, "analytics_test.sqlite3")
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        ensure_application_schema(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        user = User(name="Worker", email="worker@gmail.com", staff_code="900001",
                    schedule_start="08:00", schedule_end="17:00")
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        self.user = user

    def tearDown(self):
        self.session.close()
        self.tempdir.cleanup()

    def _add(self, status, ts, session_name, event_type):
        self.session.add(Attendance(
            sync_key=f"k-{ts.isoformat()}-{event_type}",
            user_id=self.user.id,
            timestamp=ts,
            status=status,
            session=session_name,
            event_type=event_type,
        ))
        self.session.commit()

    def test_status_distribution_only_real_categories(self):
        self._add("On Time", datetime(2026, 3, 25, 8, 0), "AM", "login")
        self._add("Late", datetime(2026, 3, 26, 9, 0), "AM", "login")
        self._add("AM Absent", datetime(2026, 3, 27, 8, 0), "AM", "auto_absent")

        dist = AnalyticsEngine(self.session).get_status_distribution(
            date(2026, 3, 1), date(2026, 3, 31)
        )

        self.assertEqual(dist["labels"], ["On Time", "Late", "Absent"])
        self.assertEqual(len(dist["labels"]), len(dist["colors"]))
        self.assertEqual(dist["data"], [1, 1, 1])

    def test_avg_daily_hours_excludes_lunch_gap(self):
        # AM session 08:00-12:00 (4h) and PM session 13:00-17:00 (4h) => 8h worked,
        # the 12:00-13:00 lunch must not be counted (a naive first-in/last-out
        # span would report 9h).
        day = date(2026, 3, 25)
        self._add("On Time", datetime(2026, 3, 25, 8, 0), "AM", "login")
        self._add("Logout", datetime(2026, 3, 25, 12, 0), "AM", "logout")
        self._add("On Time", datetime(2026, 3, 25, 13, 0), "PM", "login")
        self._add("Logout", datetime(2026, 3, 25, 17, 0), "PM", "logout")

        kpi = AnalyticsEngine(self.session).get_kpi_summary(day, day)

        self.assertEqual(kpi["avg_daily_hours"], 8.0)


if __name__ == "__main__":
    unittest.main()
