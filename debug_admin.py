
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from modules.models import Base, User, Attendance
from modules.analytics_engine import AnalyticsEngine

# Use a local SQLite database for testing
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, "data", "offline", "cviaar_local.sqlite3")
db_url = f"sqlite:///{db_path}"

engine = create_engine(db_url)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

try:
    print("Testing AnalyticsEngine...")
    analytics = AnalyticsEngine(db)
    
    print("Fetching weekly trends...")
    trends = analytics.get_weekly_trends()
    print(f"Weekly trends: {trends}")
    
    print("Fetching risk users...")
    risks = analytics.predict_risk_users()
    print(f"Risk users: {risks}")
    
    print("Success!")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
