import threading
import time
import requests
import logging
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .attendance_rules import (
    AM_SESSION,
    PM_SESSION,
    infer_event_type,
    is_absent_status,
    is_login_record,
    normalize_session,
    session_absent_status,
)
from .models import Attendance, Base, User, ensure_application_schema
import uuid

logger = logging.getLogger(__name__)
SYNC_BATCH_SIZE = 100

class SyncEngine:
    """
    Manages the synchronization of local attendance data with a remote Supabase server.

    This engine operates in two main modes:
    1.  **Producer**: It provides a method (`record_attendance`) to save attendance records
        to a local SQLite database. This allows the kiosk to operate offline.
    2.  **Consumer**: It runs a background thread that periodically checks for an internet
        connection and syncs any pending (unsynced) records to the remote server.
    """
    
    def __init__(self, database_url: str, supabase_url: str = None, supabase_key: str = None, 
                 remote_db_url: str = None, sync_interval: int = 30, device_id: str = "default_device"):
        """
        Initializes the SyncEngine.

        Args:
            database_url (str): The connection string for the local SQLite database.
            supabase_url (str, optional): The URL of the Supabase project.
            supabase_key (str, optional): The anon key for the Supabase project.
            remote_db_url (str, optional): The connection string for the remote Postgres database.
            sync_interval (int, optional): The interval in seconds between sync attempts. Defaults to 30.
            device_id (str, optional): A unique identifier for the current device. Defaults to "default_device".
        """
        self.database_url = database_url
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.remote_db_url = remote_db_url
        self.sync_interval = sync_interval
        self.device_id = device_id
        
        # Database setup for the local SQLite store
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        ensure_application_schema(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Database setup for the remote store (if provided)
        self.remote_engine = None
        self.RemoteSession = None
        if self.remote_db_url:
            try:
                remote_connect_args = {"check_same_thread": False} if self.remote_db_url.startswith("sqlite") else {}
                self.remote_engine = create_engine(self.remote_db_url, connect_args=remote_connect_args)
                Base.metadata.create_all(self.remote_engine)
                ensure_application_schema(self.remote_engine)
                self.RemoteSession = sessionmaker(bind=self.remote_engine)
                logger.info("Remote database connection initialized for syncing.")
            except Exception as e:
                logger.error(f"Failed to initialize remote database: {e}")
        
        # Thread control for the background sync worker
        self.sync_thread = None
        self.stop_sync = threading.Event()
        self.sync_lock = threading.Lock() # Ensures only one sync operation runs at a time
        
        # Statistics for monitoring sync health
        self.sync_stats = {
            'total_synced': 0,
            'total_failed': 0,
            'last_sync_time': None,
            'last_error': None,
            'consecutive_failures': 0,
            'disabled': False
        }

        self._last_absence_marked_date: str | None = None

    def start_sync_worker(self):
        """Starts the background synchronization thread."""
        if self.sync_thread is None or not self.sync_thread.is_alive():
            self.stop_sync.clear()
            self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self.sync_thread.start()
            logger.info("Sync worker thread started.")

    def _auto_mark_absent_for_date(self, target_date: date) -> int:
        session = self.Session()
        try:
            start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
            end = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, 999999)

            users = session.query(User).all()
            created = 0
            for user in users:
                day_records = (
                    session.query(Attendance)
                    .filter(
                        Attendance.user_id == user.id,
                        Attendance.timestamp >= start,
                        Attendance.timestamp <= end,
                    )
                    .all()
                )

                login_sessions = set()
                absent_sessions = set()

                for record in day_records:
                    record_status = record.status or ""
                    record_session = normalize_session(getattr(record, "session", None), record.timestamp)
                    record_event_type = getattr(record, "event_type", None)

                    if record_status == "Absent" and record_session is None:
                        absent_sessions.update({AM_SESSION, PM_SESSION})
                        continue

                    if record_session and is_login_record(record_status, record_event_type):
                        login_sessions.add(record_session)
                    if record_session and is_absent_status(record_status):
                        absent_sessions.add(record_session)

                for session_name in (AM_SESSION, PM_SESSION):
                    if session_name in login_sessions or session_name in absent_sessions:
                        continue

                    record_hour = 8 if session_name == AM_SESSION else 13
                    record_ts = datetime(target_date.year, target_date.month, target_date.day, record_hour, 0, 0)
                    record = Attendance(
                        sync_key=str(uuid.uuid4()),
                        user_id=user.id,
                        timestamp=record_ts,
                        status=session_absent_status(session_name),
                        session=session_name,
                        event_type="auto_absent",
                        auto_generated=1,
                        notes=None,
                        device_id=self.device_id,
                        synced=0,
                    )
                    session.add(record)
                    created += 1

            if created:
                session.commit()
            return created
        except Exception as e:
            session.rollback()
            logger.error(f"Auto-absence marking error: {e}")
            return 0
        finally:
            session.close()

    def stop_sync_worker(self):
        """Stops the background synchronization thread."""
        self.stop_sync.set()
        if self.sync_thread:
            self.sync_thread.join(timeout=5)

    def _sync_worker(self):
        """The main loop for the background sync worker."""
        while not self.stop_sync.is_set():
            if self.sync_stats['disabled']:
                logger.warning("Sync worker is disabled due to consecutive failures. Restart application to retry.")
                self.stop_sync.wait(3600) # Wait an hour before checking again if still disabled
                continue

            try:
                today = datetime.now().date()
                target = today - timedelta(days=1)
                target_key = target.isoformat()
                if self._last_absence_marked_date != target_key:
                    created = self._auto_mark_absent_for_date(target)
                    self._last_absence_marked_date = target_key
                    if created:
                        logger.info(f"Auto-marked {created} absences for {target_key}.")

                if self._check_internet():
                    self._sync_pending()
                
                # Dynamic sync interval based on success/failure
                interval = self.sync_interval
                if self.sync_stats['consecutive_failures'] > 0:
                    interval = min(self.sync_interval * (2 ** self.sync_stats['consecutive_failures']), 300) # Max 5 mins
                
                self.stop_sync.wait(interval)
            except Exception as e:
                logger.error(f"Sync worker loop error: {e}")
                self.sync_stats['consecutive_failures'] += 1
                time.sleep(10)

    def _check_internet(self) -> bool:
        """Checks for an active internet connection."""
        try:
            # Using a reliable hostname instead of an IP for the check
            requests.get('https://www.google.com', timeout=3)
            return True
        except requests.RequestException:
            return False

    def _sync_pending(self):
        """Fetches pending records from the local DB and attempts to sync them."""
        if self.sync_stats['consecutive_failures'] > 10:
            logger.error("Sync disabled due to too many failures.")
            self.sync_stats['disabled'] = True
            return

        with self.sync_lock:
            local_session = self.Session()
            try:
                pending = local_session.query(Attendance).filter(Attendance.synced == 0).all()
                if not pending:
                    return
                
                logger.info(f"Found {len(pending)} pending records to sync.")

                batches = [
                    pending[index:index + SYNC_BATCH_SIZE]
                    for index in range(0, len(pending), SYNC_BATCH_SIZE)
                ]
                synced_count = 0

                for batch_index, batch_records in enumerate(batches, start=1):
                    sync_success = False
                    if self.RemoteSession:
                        sync_success = self._sync_direct_postgres(batch_records)
                    else:
                        payload = []
                        for record in batch_records:
                            payload.append({
                                'sync_key': record.sync_key,
                                'user_id': record.user_id,
                                'timestamp': record.timestamp.isoformat(),
                                'status': record.status,
                                'session': record.session,
                                'event_type': record.event_type,
                                'auto_generated': record.auto_generated,
                                'notes': record.notes,
                                'device_id': record.device_id,
                                'synced': 1,
                                'synced_at': datetime.now().isoformat()
                            })
                        sync_success = self._upsert_supabase(payload)

                    if not sync_success:
                        failed_count = len(batch_records)
                        self.sync_stats['total_failed'] += failed_count
                        self.sync_stats['consecutive_failures'] += 1
                        logger.error(
                            "Failed to sync batch %s/%s (%s records).",
                            batch_index,
                            len(batches),
                            failed_count,
                        )
                        break

                    synced_at = datetime.now()
                    synced_ids = [record.id for record in batch_records]
                    (
                        local_session.query(Attendance)
                        .filter(Attendance.id.in_(synced_ids))
                        .update(
                            {
                                Attendance.synced: 1,
                                Attendance.synced_at: synced_at,
                            },
                            synchronize_session=False,
                        )
                    )
                    local_session.commit()
                    synced_count += len(batch_records)
                    self.sync_stats['total_synced'] += len(batch_records)
                    self.sync_stats['last_sync_time'] = synced_at
                    self.sync_stats['consecutive_failures'] = 0
                    logger.info(
                        "Synced batch %s/%s (%s records).",
                        batch_index,
                        len(batches),
                        len(batch_records),
                    )

                if synced_count:
                    logger.info("Completed sync for %s records.", synced_count)
            except Exception as e:
                self.sync_stats['total_failed'] += 1
                self.sync_stats['last_error'] = str(e)
                self.sync_stats['consecutive_failures'] += 1
                logger.error(f"Error during _sync_pending: {e}")
            finally:
                local_session.close()

    def _sync_direct_postgres(self, pending_records: List[Attendance]) -> bool:
        """
        Directly syncs records to the remote Postgres database using SQLAlchemy.
        """
        if not self.RemoteSession:
            return False
            
        remote_session = self.RemoteSession()
        local_session = self.Session()
        try:
            # 1. Sync users first
            unique_user_ids = {r.user_id for r in pending_records}
            for user_id in unique_user_ids:
                remote_user = remote_session.query(User).filter_by(id=user_id).first()
                if not remote_user:
                    local_user = local_session.query(User).filter_by(id=user_id).first()
                    if local_user:
                        new_remote_user = User(
                            id=local_user.id,
                            name=local_user.name,
                            email=local_user.email,
                            staff_code=local_user.staff_code,
                            created_at=local_user.created_at,
                            schedule_start=local_user.schedule_start,
                            schedule_end=local_user.schedule_end,
                            employment_type=local_user.employment_type,
                            role=local_user.role
                        )
                        remote_session.add(new_remote_user)
            
            remote_session.flush()

            # 2. Sync attendance records
            for local_r in pending_records:
                remote_r = remote_session.query(Attendance).filter_by(sync_key=local_r.sync_key).first()
                
                if not remote_r:
                    new_remote_r = Attendance(
                        sync_key=local_r.sync_key,
                        user_id=local_r.user_id,
                        timestamp=local_r.timestamp,
                        status=local_r.status,
                        session=local_r.session,
                        event_type=local_r.event_type,
                        auto_generated=local_r.auto_generated,
                        notes=local_r.notes,
                        device_id=local_r.device_id,
                        synced=1,
                        synced_at=datetime.now()
                    )
                    remote_session.add(new_remote_r)
                else:
                    remote_r.status = local_r.status
                    remote_r.session = local_r.session
                    remote_r.event_type = local_r.event_type
                    remote_r.auto_generated = local_r.auto_generated
                    remote_r.notes = local_r.notes
                    remote_r.synced_at = datetime.now()
            
            remote_session.commit()
            return True
        except Exception as e:
            remote_session.rollback()
            logger.error(f"Sync error: {e}")
            # Fallback to REST if Postgres fails (e.g., auth error)
            return False
        finally:
            remote_session.close()
            local_session.close()

    def _upsert_supabase(self, batch) -> bool:
        """
        Upserts a batch of attendance records to the Supabase server.

        Args:
            batch (list): A list of attendance records to upsert.

        Returns:
            bool: True if the upsert was successful, False otherwise.
        """
        if not self.supabase_url or not self.supabase_key:
            logger.error("Supabase URL or Key not configured. Skipping sync.")
            return False

        try:
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}',
                'Content-Type': 'application/json',
                'Prefer': 'resolution=merge-duplicates' # Important for handling duplicates
            }
            # Ensure URL has trailing slash before rest/v1/
            base_url = self.supabase_url.rstrip('/')
            url = f"{base_url}/rest/v1/attendance_logs"
            
            logger.info(f"Attempting to sync {len(batch)} records to {url}")
            # Log the first record as a sample (masking keys if any, but batch is data)
            if batch:
                logger.debug(f"Sample record: {batch[0]}")

            resp = requests.post(url, json=batch, headers=headers, timeout=10)
            
            if resp.status_code in [200, 201, 204]:
                logger.info(f"Successfully synced {len(batch)} records. Status: {resp.status_code}")
                return True
            else:
                logger.error(f"Failed to sync records. Status: {resp.status_code}, Response: {resp.text}")
                # Log headers for debugging (excluding sensitive info)
                safe_headers = {k: v if k.lower() not in ['apikey', 'authorization'] else '[MASKED]' for k, v in headers.items()}
                logger.error(f"Request Headers: {safe_headers}")
                return False
        except requests.exceptions.RequestException as re:
            logger.error(f"Network error during Supabase sync: {re}")
            return False
        except Exception as e:
            logger.error(f"Unexpected exception during Supabase sync: {e}")
            return False

    def record_attendance(
        self,
        user_id: int,
        status: str,
        notes: str = None,
        timestamp: datetime | None = None,
        session: str | None = None,
        event_type: str | None = None,
        auto_generated: bool = False,
    ) -> str:
        """
        Records a new attendance entry to the local database.

        This acts as the producer, creating a record that the sync worker will later consume.

        Args:
            user_id (int): The ID of the user.
            status (str): The attendance status (e.g., On Time, Late).
            notes (str, optional): Any notes for the record. Defaults to None.

        Returns:
            str: The unique sync key for the new record.
        """
        db_session = self.Session()
        try:
            sync_key = str(uuid.uuid4())
            record_ts = timestamp or datetime.now()
            record = Attendance(
                sync_key=sync_key,
                user_id=user_id,
                timestamp=record_ts,
                status=status,
                session=normalize_session(session, record_ts),
                event_type=infer_event_type(status, event_type),
                auto_generated=1 if auto_generated else 0,
                notes=notes,
                device_id=self.device_id,
                synced=0,
            )
            db_session.add(record)
            db_session.commit()
            return sync_key
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    def get_sync_stats(self) -> Dict[str, Any]:
        """Returns the current synchronization statistics."""
        return self.sync_stats.copy()
