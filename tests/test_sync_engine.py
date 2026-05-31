import os
import sqlite3
import tempfile
import unittest
from datetime import date, datetime

from sqlalchemy import create_engine

from modules.models import Attendance, Base, User, ensure_application_schema
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

    def test_backfill_absences_skips_weekends(self):
        session = self.engine.Session()
        try:
            user = User(name="Backfill User", email="backfill@gmail.com", staff_code="333333")
            session.add(user)
            session.commit()
            session.refresh(user)
        finally:
            session.close()

        # Backfill the week ending Friday 2026-03-27 (Mon-Fri are work days).
        created = self.engine._backfill_absences(date(2026, 3, 27))

        verify_session = self.engine.Session()
        try:
            rows = (
                verify_session.query(Attendance)
                .filter(Attendance.user_id == user.id)
                .all()
            )
        finally:
            verify_session.close()

        marked_dates = {row.timestamp.date() for row in rows}
        # Saturday/Sunday in the window must not be marked absent.
        self.assertNotIn(date(2026, 3, 21), marked_dates)  # Saturday
        self.assertNotIn(date(2026, 3, 22), marked_dates)  # Sunday
        # Working days are marked (AM + PM per day).
        self.assertIn(date(2026, 3, 27), marked_dates)  # Friday
        self.assertTrue(all(d.weekday() < 5 for d in marked_dates))
        self.assertEqual(created, len(rows))

    def test_get_sync_stats_no_longer_requires_face_engine(self):
        stats = self.engine.get_sync_stats()

        self.assertIn("total_synced", stats)
        self.assertNotIn("thermal", stats)

    def test_schema_repair_adds_missing_user_columns(self):
        legacy_path = os.path.join(self.tempdir.name, "legacy.sqlite3")
        connection = sqlite3.connect(legacy_path)
        try:
            connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL)")
            connection.execute(
                "CREATE TABLE attendance_logs ("
                "id INTEGER PRIMARY KEY, "
                "user_id INTEGER NOT NULL, "
                "timestamp DATETIME, "
                "status VARCHAR(20) NOT NULL, "
                "notes VARCHAR(200), "
                "device_id VARCHAR(50))"
            )
            connection.commit()
        finally:
            connection.close()

        ensure_application_schema(create_engine(f"sqlite:///{legacy_path}"))

        verify = sqlite3.connect(legacy_path)
        try:
            user_columns = {
                row[1]
                for row in verify.execute("PRAGMA table_info(users)").fetchall()
            }
            attendance_columns = {
                row[1]
                for row in verify.execute("PRAGMA table_info(attendance_logs)").fetchall()
            }
        finally:
            verify.close()

        self.assertIn("email", user_columns)
        self.assertIn("staff_code", user_columns)
        self.assertIn("role", user_columns)
        self.assertIn("sync_key", attendance_columns)
        self.assertIn("synced", attendance_columns)
        self.assertIn("synced_at", attendance_columns)

    def test_schema_repair_backfills_null_sync_keys_and_adds_unique_index(self):
        legacy_path = os.path.join(self.tempdir.name, "legacy_keys.sqlite3")
        connection = sqlite3.connect(legacy_path)
        try:
            connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL)")
            # Legacy attendance table without sync_key (predates offline sync).
            connection.execute(
                "CREATE TABLE attendance_logs ("
                "id INTEGER PRIMARY KEY, "
                "user_id INTEGER NOT NULL, "
                "timestamp DATETIME, "
                "status VARCHAR(20) NOT NULL)"
            )
            connection.execute("INSERT INTO users (id, name) VALUES (1, 'Legacy')")
            for day in (1, 2, 3):
                connection.execute(
                    "INSERT INTO attendance_logs (user_id, timestamp, status) "
                    f"VALUES (1, '2026-01-0{day} 08:00:00', 'On Time')"
                )
            connection.commit()
        finally:
            connection.close()

        ensure_application_schema(create_engine(f"sqlite:///{legacy_path}"))

        verify = sqlite3.connect(legacy_path)
        try:
            keys = [row[0] for row in verify.execute("SELECT sync_key FROM attendance_logs").fetchall()]
            indexes = {row[1] for row in verify.execute("PRAGMA index_list(attendance_logs)").fetchall()}
        finally:
            verify.close()

        self.assertEqual(len(keys), 3)
        self.assertTrue(all(key for key in keys), "every legacy row should get a sync_key")
        self.assertEqual(len(set(keys)), 3, "backfilled sync_keys must be unique")
        self.assertIn("ix_attendance_logs_sync_key", indexes)

    def test_local_sqlite_engine_runs_in_wal_mode(self):
        with self.engine.engine.connect() as conn:
            from sqlalchemy import text as _text

            journal_mode = conn.execute(_text("PRAGMA journal_mode")).scalar()
            busy_timeout = conn.execute(_text("PRAGMA busy_timeout")).scalar()

        self.assertEqual(str(journal_mode).lower(), "wal")
        self.assertGreaterEqual(int(busy_timeout), 5000)

    def test_direct_postgres_sync_inserts_then_updates_remote_user(self):
        remote_path = os.path.join(self.tempdir.name, "remote.sqlite3")
        remote_url = f"sqlite:///{remote_path}"
        engine = SyncEngine(
            database_url=self.database_url,
            remote_db_url=remote_url,
            sync_interval=1,
            device_id="test-device",
        )

        local = engine.Session()
        try:
            local.add(User(id=1, name="Original", staff_code="100000", schedule_start="06:00"))
            local.commit()
        finally:
            local.close()

        engine.record_attendance(
            user_id=1, status="On Time", timestamp=datetime(2026, 3, 25, 8, 0), event_type="login"
        )
        engine._sync_pending()

        remote = engine.RemoteSession()
        try:
            remote_user = remote.query(User).filter_by(id=1).first()
            synced_rows = remote.query(Attendance).count()
        finally:
            remote.close()
        self.assertIsNotNone(remote_user)
        self.assertEqual(remote_user.name, "Original")
        self.assertEqual(remote_user.schedule_start, "06:00")
        self.assertEqual(synced_rows, 1)

        # Local rows should be flagged synced so they are not re-sent.
        local = engine.Session()
        try:
            pending = local.query(Attendance).filter(Attendance.synced == 0).count()
        finally:
            local.close()
        self.assertEqual(pending, 0)

        # Edit the user locally, record again, and re-sync: the remote profile
        # must reflect the edit instead of staying stale.
        local = engine.Session()
        try:
            user = local.query(User).filter_by(id=1).first()
            user.name = "Renamed"
            user.schedule_start = "07:30"
            local.commit()
        finally:
            local.close()

        engine.record_attendance(
            user_id=1, status="On Time", timestamp=datetime(2026, 3, 26, 8, 0), event_type="login"
        )
        engine._sync_pending()

        remote = engine.RemoteSession()
        try:
            remote_user = remote.query(User).filter_by(id=1).first()
            remote_count = remote.query(Attendance).count()
        finally:
            remote.close()
        engine.stop_sync_worker()

        self.assertEqual(remote_user.name, "Renamed")
        self.assertEqual(remote_user.schedule_start, "07:30")
        self.assertEqual(remote_count, 2)


if __name__ == "__main__":
    unittest.main()
