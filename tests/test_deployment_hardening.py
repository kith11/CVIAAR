import importlib
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient


TEMP_DIR = tempfile.TemporaryDirectory()
TEST_DB_PATH = os.path.join(TEMP_DIR.name, "admin_dashboard.sqlite3")

os.environ["APP_ROLE"] = "ADMIN_DASHBOARD"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ["SECRET_KEY"] = "unit-test-secret-key"
os.environ["ADMIN_PASSWORD"] = "unit-test-admin-password"
os.environ["DEBUG"] = "true"
os.environ["SYNC_ENABLED"] = "false"
os.environ["REDIS_URL"] = ""
os.environ["UPSTASH_REDIS_URL"] = ""
os.environ["UPSTASH_REDIS_TOKEN"] = ""

sys.modules.pop("config", None)
sys.modules.pop("app", None)

app_module = importlib.import_module("app")
config_module = importlib.import_module("config")
User = app_module.User
Attendance = app_module.Attendance
AuditEvent = app_module.AuditEvent


class DeploymentHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_module.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        TEMP_DIR.cleanup()

    def setUp(self):
        self.client.cookies.clear()
        session = app_module.SessionLocal()
        try:
            session.query(AuditEvent).delete()
            session.query(Attendance).delete()
            session.query(User).delete()
            session.commit()
        finally:
            session.close()

    def create_user(self, name: str, staff_code: str, email: str = "user@gmail.com") -> User:
        session = app_module.SessionLocal()
        try:
            user = User(name=name, email=email, staff_code=staff_code)
            session.add(user)
            session.commit()
            session.refresh(user)
            session.expunge(user)
            return user
        finally:
            session.close()

    def test_validate_runtime_settings_requires_cloud_database(self):
        with self.assertRaises(RuntimeError):
            app_module.validate_runtime_settings(
                SimpleNamespace(
                    APP_ROLE="ADMIN_DASHBOARD",
                    DATABASE_URL=None,
                    SECRET_KEY="strong-secret",
                    ADMIN_PASSWORD="strong-password",
                )
            )

    def test_legacy_supabase_db_url_alias_still_loads(self):
        previous_database_url = os.environ.pop("DATABASE_URL", None)
        os.environ["SUPABASE_DB_URL"] = "sqlite:///legacy-admin.sqlite3"
        try:
            settings = config_module.Settings(_env_file=None)
            self.assertEqual(settings.DATABASE_URL, "sqlite:///legacy-admin.sqlite3")
        finally:
            os.environ.pop("SUPABASE_DB_URL", None)
            if previous_database_url is not None:
                os.environ["DATABASE_URL"] = previous_database_url

    def test_admin_json_routes_require_authentication(self):
        self.assertEqual(self.client.get("/api/users").status_code, 403)
        self.assertEqual(self.client.get("/api/recent_logs").status_code, 403)
        self.assertEqual(
            self.client.get(
                "/api/advanced_analytics_data?start_date=2026-03-01&end_date=2026-03-26&employment_type=All&user_id=All"
            ).status_code,
            403,
        )
        self.assertEqual(self.client.post("/api/train").status_code, 403)

    def test_staff_portal_uses_staff_code_session_and_blocks_other_users(self):
        alice = self.create_user("Alice Example", "111111", email="alice@gmail.com")
        bob = self.create_user("Bob Example", "222222", email="bob@gmail.com")

        page = self.client.get("/staff_portal")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Choose Your Name", page.text)

        login_response = self.client.post("/staff_portal", data={"user_id": alice.id}, follow_redirects=False)
        self.assertEqual(login_response.status_code, 303)
        self.assertEqual(login_response.headers["location"], "/staff_portal")

        portal = self.client.get("/staff_portal")
        self.assertEqual(portal.status_code, 200)
        self.assertIn("Alice Example", portal.text)

        own_report = self.client.get(f"/staff_report/{alice.id}?month=3&year=2026")
        self.assertEqual(own_report.status_code, 200)
        self.assertIn("Daily Time Record", own_report.text)

        other_report = self.client.get(f"/staff_report/{bob.id}?month=3&year=2026", follow_redirects=False)
        self.assertEqual(other_report.status_code, 303)
        self.assertEqual(other_report.headers["location"], "/staff_portal")

    def test_staff_cannot_request_another_users_report_email(self):
        alice = self.create_user("Alice Example", "111111", email="alice@gmail.com")
        bob = self.create_user("Bob Example", "222222", email="bob@gmail.com")

        self.client.post("/staff_portal", data={"user_id": alice.id}, follow_redirects=True)
        response = self.client.post(
            "/request_early_report",
            data={"user_id": bob.id, "month": 3, "year": 2026},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("You can only request your own attendance report.", response.text)
        self.assertNotIn("Bob Example", response.text)
        self.assertIn("Alice Example", response.text)

    def test_admin_can_open_staff_portal_sign_in_page(self):
        login = self.client.post("/login", data={"password": "unit-test-admin-password"}, follow_redirects=False)
        self.assertEqual(login.status_code, 303)

        response = self.client.get("/staff_portal")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Choose Your Name", response.text)

    def test_audit_logs_show_recorded_admin_actions(self):
        login = self.client.post("/login", data={"password": "unit-test-admin-password"}, follow_redirects=True)
        self.assertEqual(login.status_code, 200)

        create_response = self.client.post(
            "/add_user",
            data={"name": "Audit User", "email": "audituser@gmail.com", "employment_type": "Full-time"},
            follow_redirects=False,
        )
        self.assertEqual(create_response.status_code, 303)

        audit_page = self.client.get("/audit_logs")
        self.assertEqual(audit_page.status_code, 200)
        self.assertIn("Activity Log", audit_page.text)
        self.assertIn("Created user Audit User", audit_page.text)


if __name__ == "__main__":
    unittest.main()
