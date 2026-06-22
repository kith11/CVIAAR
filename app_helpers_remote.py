from modules.models import User, Attendance
from datetime import datetime
import os, shutil, logging


def _get_sync_engine():
    import app
    return app.sync_engine

# Helpers applied from remote events to local DB

def _apply_remote_delete(user_id: int) -> None:
    try:
        sync_engine = _get_sync_engine()
        if not sync_engine:
            logging.warning("sync_engine not available for _apply_remote_delete")
            return
        session = sync_engine.Session()
        try:
            session.query(Attendance).filter(Attendance.user_id == user_id).delete()
            user = session.get(User, user_id)
            if user:
                session.delete(user)
            session.commit()
            # remove face files
            user_dir = os.path.join(os.path.dirname(__file__), 'data', 'faces', str(user_id))
            try:
                if os.path.exists(user_dir):
                    shutil.rmtree(user_dir)
            except Exception:
                logging.exception("Failed to remove face dir for user %s", user_id)
            logging.info("Applied remote delete for user %s", user_id)
        except Exception:
            session.rollback()
            logging.exception("Failed to apply remote delete for user %s", user_id)
        finally:
            session.close()
    except Exception:
        logging.exception("Unexpected error in _apply_remote_delete for %s", user_id)


def _apply_remote_upsert(user_obj: dict) -> None:
    try:
        sync_engine = _get_sync_engine()
        if not sync_engine:
            logging.warning("sync_engine not available for _apply_remote_upsert")
            return
        session = sync_engine.Session()
        try:
            uid = user_obj.get('id')
            if uid is None:
                logging.warning('Remote upsert missing id: %s', user_obj)
                return
            existing = session.get(User, uid)
            if not existing:
                u = User(
                    id=uid,
                    name=user_obj.get('name'),
                    email=user_obj.get('email'),
                    staff_code=user_obj.get('staff_code'),
                    created_at=(datetime.fromisoformat(user_obj.get('created_at')) if user_obj.get('created_at') else None),
                    schedule_start=user_obj.get('schedule_start'),
                    schedule_end=user_obj.get('schedule_end'),
                    employment_type=user_obj.get('employment_type'),
                    role=user_obj.get('role'),
                )
                session.add(u)
                session.commit()
                logging.info('Applied remote upsert creating user %s', uid)
            else:
                # update fields conservatively
                existing.name = user_obj.get('name') or existing.name
                existing.email = user_obj.get('email') or existing.email
                existing.staff_code = user_obj.get('staff_code') or existing.staff_code
                existing.schedule_start = user_obj.get('schedule_start') or existing.schedule_start
                existing.schedule_end = user_obj.get('schedule_end') or existing.schedule_end
                existing.employment_type = user_obj.get('employment_type') or existing.employment_type
                existing.role = user_obj.get('role') or existing.role
                session.commit()
                logging.info('Applied remote upsert updated user %s', uid)
        except Exception:
            session.rollback()
            logging.exception('Failed to apply remote upsert for %s', user_obj)
        finally:
            session.close()
    except Exception:
        logging.exception('Unexpected error in _apply_remote_upsert for %s', user_obj)
