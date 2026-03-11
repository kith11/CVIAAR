import threading
import time
import requests
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Attendance, Base
from .face_engine import FaceEngine
import uuid

logger = logging.getLogger(__name__)

class SyncEngine:
    """
    Manages the synchronization of local attendance data with a remote Supabase server.

    This engine operates in two main modes:
    1.  **Producer**: It provides a method (`record_attendance`) to save attendance records
        to a local SQLite database. This allows the kiosk to operate offline.
    2.  **Consumer**: It runs a background thread that periodically checks for an internet
        connection and syncs any pending (unsynced) records to the remote server.
    """
    
    def __init__(self, database_url: str, supabase_url: str, supabase_key: str, 
                 sync_interval: int = 30, device_id: str = "default_device"):
        """
        Initializes the SyncEngine.

        Args:
            database_url (str): The connection string for the local SQLite database.
            supabase_url (str): The URL of the Supabase project.
            supabase_key (str): The anon key for the Supabase project.
            sync_interval (int, optional): The interval in seconds between sync attempts. Defaults to 30.
            device_id (str, optional): A unique identifier for the current device. Defaults to "default_device".
        """
        self.database_url = database_url
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.sync_interval = sync_interval
        self.device_id = device_id
        
        # Database setup for the local SQLite store
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Face Engine (can be used for related tasks, e.g., thermal status)
        self.face_engine = FaceEngine()
        
        # Thread control for the background sync worker
        self.sync_thread = None
        self.stop_sync = threading.Event()
        self.sync_lock = threading.Lock() # Ensures only one sync operation runs at a time
        
        # Statistics for monitoring sync health
        self.sync_stats = {
            'total_synced': 0,
            'total_failed': 0,
            'last_sync_time': None,
            'last_error': None
        }

    def start_sync_worker(self):
        """Starts the background synchronization thread."""
        if self.sync_thread is None or not self.sync_thread.is_alive():
            self.stop_sync.clear()
            self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self.sync_thread.start()
            logger.info("Sync worker thread started.")

    def stop_sync_worker(self):
        """Stops the background synchronization thread."""
        self.stop_sync.set()
        if self.sync_thread:
            self.sync_thread.join(timeout=5)

    def _sync_worker(self):
        """The main loop for the background sync worker."""
        while not self.stop_sync.is_set():
            try:
                if self._check_internet():
                    self._sync_pending()
                self.stop_sync.wait(self.sync_interval)
            except Exception as e:
                logger.error(f"Sync error: {e}")
                time.sleep(5)

    def _check_internet(self) -> bool:
        """Checks for an active internet connection."""
        try:
            requests.get('https://8.8.8.8', timeout=2)
            return True
        except:
            return False

    def _sync_pending(self):
        """Fetches pending records from the local DB and attempts to sync them."""
        with self.sync_lock:
            session = self.Session()
            try:
                pending = session.query(Attendance).filter(Attendance.synced == 0).all()
                if not pending: return
                
                batch = []
                for r in pending:
                    batch.append({
                        'sync_key': r.sync_key,
                        'user_id': r.user_id,
                        'timestamp': r.timestamp.isoformat(),
                        'status': r.status,
                        'notes': r.notes,
                        'device_id': r.device_id
                    })
                
                if self._upsert_supabase(batch):
                    for r in pending:
                        r.synced = 1
                        r.synced_at = datetime.now()
                    session.commit()
                    self.sync_stats['total_synced'] += len(pending)
                    self.sync_stats['last_sync_time'] = datetime.now()
                else:
                    self.sync_stats['total_failed'] += len(pending)
            finally:
                session.close()

    def _upsert_supabase(self, batch) -> bool:
        """
        Upserts a batch of attendance records to the Supabase server.

        Args:
            batch (list): A list of attendance records to upsert.

        Returns:
            bool: True if the upsert was successful, False otherwise.
        """
        try:
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}',
                'Content-Type': 'application/json',
                'Prefer': 'resolution=merge-duplicates' # Important for handling duplicates
            }
            url = f"{self.supabase_url}/rest/v1/attendance_logs"
            resp = requests.post(url, json=batch, headers=headers, timeout=10)
            return resp.status_code in [200, 201]
        except:
            return False

    def record_attendance(self, user_id: int, status: str, notes: str = None) -> str:
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
        session = self.Session()
        try:
            sync_key = str(uuid.uuid4())
            record = Attendance(
                sync_key=sync_key, user_id=user_id, status=status,
                notes=notes, device_id=self.device_id, synced=0
            )
            session.add(record)
            session.commit()
            return sync_key
        finally:
            session.close()

    def get_sync_stats(self) -> Dict[str, Any]:
        """Returns the current synchronization statistics."""
        stats = self.sync_stats.copy()
        stats['thermal'] = self.face_engine.get_thermal_status()
        return stats
