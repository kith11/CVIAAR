import os
import tempfile
import unittest
from datetime import date, datetime

from modules.models import Attendance, Base, User
from modules.sync_engine import SyncEngine


class SyncEngineTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "sync_test.sqlite3")
        self.database_url = f"sqlite:///{self.db_path}"
        self.engine = SyncEngine(database_url=self.database_url, sync_interval=1, device_id="test-device")

    def tearDown(self):
        self.engine.stop_sync_worker()
        self.tempdir.cleanup()

    def test_record_attendance_writes_local_row(self):
        session = self.engine.Session()
        try:
            user = User(name="Test User", email="test@gmail.com", staff_code="123456")
            session.add(user)
            session.commit()
            session.refresh(user)
        finally:
            session.close()

        sync_key = self.engine.record_attendance(
            user_id=user.id,
            status="On Time",
            timestamp=datetime(2026, 3, 25, 8, 30),
            event_type="login",
        )

        verify_session = self.engine.Session()
        try:
            rows = verify_session.execute(Base.metadata.tables["attendance_logs"].select()).fetchall()
        finally:
            verify_session.close()

        self.assertTrue(sync_key)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "On Time")
        self.assertEqual(rows[0].session, "AM")
        self.assertEqual(rows[0].event_type, "login")
        self.assertEqual(rows[0].auto_generated, 0)
        self.assertEqual(rows[0].device_id, "test-device")

    def test_auto_mark_absent_creates_only_missing_session_absence(self):
        session = self.engine.Session()
        try:
            user = User(name="Morning Only", email="morning@gmail.com", staff_code="111111")
            session.add(user)
            session.commit()
            session.refresh(user)
        finally:
            session.close()

        self.engine.record_attendance(
            user_id=user.id,
            status="On Time",
            timestamp=datetime(2026, 3, 25, 8, 10),
            event_type="login",
        )

        created = self.engine._auto_mark_absent_for_date(date(2026, 3, 25))

        verify_session = self.engine.Session()
        try:
            rows = (
                verify_session.query(Attendance)
                .filter(Attendance.user_id == user.id)
                .order_by(Attendance.timestamp.asc())
                .all()
            )
        finally:
            verify_session.close()

        self.assertEqual(created, 1)
        self.assertEqual([row.status for row in rows], ["On Time", "PM Absent"])
        self.assertEqual(rows[1].session, "PM")
        self.assertEqual(rows[1].event_type, "auto_absent")
        self.assertEqual(rows[1].auto_generated, 1)

    def test_auto_mark_absent_is_idempotent_and_logout_only_does_not_count_as_presence(self):
        session = self.engine.Session()
        try:
            user = User(name="Logout Only", email="logout@gmail.com", staff_code="222222")
            session.add(user)
            session.commit()
            session.refresh(user)
        finally:
            session.close()

        self.engine.record_attendance(
            user_id=user.id,
            status="Logout",
            timestamp=datetime(2026, 3, 25, 9, 0),
            event_type="logout",
        )

        first_created = self.engine._auto_mark_absent_for_date(date(2026, 3, 25))
        second_created = self.engine._auto_mark_absent_for_date(date(2026, 3, 25))

        verify_session = self.engine.Session()
        try:
            rows = (
                verify_session.query(Attendance)
                .filter(Attendance.user_id == user.id)
                .order_by(Attendance.timestamp.asc())
                .all()
            )
        finally:
            verify_session.close()

        self.assertEqual(first_created, 2)
        self.assertEqual(second_created, 0)
        self.assertEqual([row.status for row in rows], ["AM Absent", "Logout", "PM Absent"])

    def test_get_sync_stats_no_longer_requires_face_engine(self):
        stats = self.engine.get_sync_stats()

        self.assertIn("total_synced", stats)
        self.assertNotIn("thermal", stats)


if __name__ == "__main__":
    unittest.main()
