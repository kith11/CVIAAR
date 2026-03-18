"""
Unit tests for CVIAAR application fixes

Tests cover:
1. Standby mode functionality
2. Profile editing functionality  
3. Analytics endpoint behavior
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import the app and models
from app import app, get_db
from modules.models import Base, User, Attendance
from modules.analytics_engine import AnalyticsEngine

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Create a clean database for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function") 
def client(db_session):
    """Create a test client with mocked dependencies"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

class TestStandbyMode:
    """Test standby mode functionality"""
    
    def test_standby_overlay_elements_exist(self, client):
        """Test that standby overlay elements are present in the HTML"""
        response = client.get("/")
        assert response.status_code == 200
        
        html_content = response.text
        assert 'id="standby-overlay"' in html_content
        assert 'id="standby-names"' in html_content
        assert 'class="standby-title"' in html_content
        assert "Research Team Credits" in html_content
    
    def test_standby_css_styles_exist(self, client):
        """Test that standby CSS styles are properly defined"""
        response = client.get("/")
        assert response.status_code == 200
        
        html_content = response.text
        assert '.standby-overlay' in html_content
        assert 'position: fixed' in html_content
        assert 'z-index: 2000' in html_content
        assert '.standby-bg' in html_content
        assert '.standby-inner' in html_content
    
    def test_standby_javascript_logic(self, client):
        """Test that standby JavaScript logic is properly implemented"""
        response = client.get("/")
        assert response.status_code == 200
        
        html_content = response.text
        
        # Check for key JavaScript functions
        assert 'function markActivity()' in html_content
        assert 'function createFloatingNames()' in html_content
        assert 'function standbyLoop()' in html_content
        assert 'STANDBY_AFTER_MS' in html_content
        
        # Check for research members array
        assert 'researchMembers = [' in html_content
        assert 'Margarette G. Biro (Leader)' in html_content
        assert 'John Keith A. Barrientos' in html_content
    
    def test_standby_event_listeners(self, client):
        """Test that standby event listeners are properly attached"""
        response = client.get("/")
        assert response.status_code == 200
        
        html_content = response.text
        
        # Check for event listeners
        assert "addEventListener('mousemove'" in html_content
        assert "addEventListener('keydown'" in html_content
        assert "addEventListener('click'" in html_content
        assert "addEventListener('touchstart'" in html_content
        assert "standbyOverlay.addEventListener('click'" in html_content

class TestProfileEditing:
    """Test profile editing functionality"""
    
    def test_edit_user_page_requires_admin(self, client, db_session):
        """Test that edit user page requires admin authentication"""
        # Create test user
        user = User(name="Test User", email="test@gmail.com", staff_code="123456")
        db_session.add(user)
        db_session.commit()
        
        # Try to access without login
        response = client.get(f"/edit_user/{user.id}")
        assert response.status_code == 307  # Redirect to login
        assert "/login" in response.headers.get("location", "")
    
    def test_edit_user_page_shows_form(self, client, db_session):
        """Test that edit user page displays the form correctly"""
        # Create test user
        user = User(name="Test User", email="test@gmail.com", staff_code="123456")
        db_session.add(user)
        db_session.commit()
        
        # Login as admin
        with client:
            client.post("/login", data={"password": "admin123"})
            
            response = client.get(f"/edit_user/{user.id}")
            assert response.status_code == 200
            
            html_content = response.text
            assert "Edit User Profile" in html_content
            assert 'name="name"' in html_content
            assert 'name="email"' in html_content
            assert 'name="employment_type"' in html_content
            assert 'name="schedule_start"' in html_content
            assert 'name="schedule_end"' in html_content
            assert 'name="role"' in html_content
            assert 'value="Test User"' in html_content
            assert 'value="test@gmail.com"' in html_content
    
    def test_update_user_validates_gmail(self, client, db_session):
        """Test that user updates validate Gmail addresses"""
        # Create test user
        user = User(name="Test User", email="test@gmail.com", staff_code="123456")
        db_session.add(user)
        db_session.commit()
        
        # Login as admin
        with client:
            client.post("/login", data={"password": "admin123"})
            
            # Try to update with non-Gmail address
            response = client.post(f"/update_user/{user.id}", data={
                "name": "Updated User",
                "email": "test@yahoo.com",
                "employment_type": "Full-time",
                "schedule_start": "06:00",
                "schedule_end": "19:00",
                "role": "staff"
            })
            
            assert response.status_code == 303
            # Should redirect back to edit page with flash message
            
            # Verify user was not updated
            db_session.refresh(user)
            assert user.email == "test@gmail.com"
    
    def test_update_user_prevents_duplicate_email(self, client, db_session):
        """Test that user updates prevent duplicate email addresses"""
        # Create two test users
        user1 = User(name="User 1", email="user1@gmail.com", staff_code="123456")
        user2 = User(name="User 2", email="user2@gmail.com", staff_code="654321")
        db_session.add_all([user1, user2])
        db_session.commit()
        
        # Login as admin
        with client:
            client.post("/login", data={"password": "admin123"})
            
            # Try to update user2 with user1's email
            response = client.post(f"/update_user/{user2.id}", data={
                "name": "User 2 Updated",
                "email": "user1@gmail.com",  # This should fail
                "employment_type": "Full-time",
                "schedule_start": "06:00",
                "schedule_end": "19:00",
                "role": "staff"
            })
            
            assert response.status_code == 303
            
            # Verify user2 was not updated
            db_session.refresh(user2)
            assert user2.email == "user2@gmail.com"
    
    def test_update_user_success(self, client, db_session):
        """Test successful user profile update"""
        # Create test user
        user = User(name="Test User", email="test@gmail.com", staff_code="123456")
        db_session.add(user)
        db_session.commit()
        
        # Login as admin
        with client:
            client.post("/login", data={"password": "admin123"})
            
            # Update user successfully
            response = client.post(f"/update_user/{user.id}", data={
                "name": "Updated User Name",
                "email": "updated@gmail.com",
                "employment_type": "Part-time",
                "schedule_start": "08:00",
                "schedule_end": "17:00",
                "role": "admin"
            })
            
            assert response.status_code == 303
            
            # Verify user was updated
            db_session.refresh(user)
            assert user.name == "Updated User Name"
            assert user.email == "updated@gmail.com"
            assert user.employment_type == "Part-time"
            assert user.schedule_start == "08:00"
            assert user.schedule_end == "17:00"
            assert user.role == "admin"

class TestAnalytics:
    """Test analytics functionality for dummy accounts"""
    
    def test_analytics_includes_all_users(self, client, db_session):
        """Test that analytics includes all users regardless of data"""
        # Create test users with different data
        user1 = User(name="Active User", email="active@gmail.com", staff_code="111111")
        user2 = User(name="Inactive User", email="inactive@gmail.com", staff_code="222222")
        user3 = User(name="Test User", email="test@gmail.com", staff_code="333333")
        
        db_session.add_all([user1, user2, user3])
        db_session.commit()
        
        # Add some attendance data for user1 only
        attendance = Attendance(
            user_id=user1.id,
            status="Present",
            timestamp=datetime.now()
        )
        db_session.add(attendance)
        db_session.commit()
        
        # Test analytics endpoint
        response = client.get("/analytics")
        assert response.status_code == 200
        
        data = response.json()
        assert "stats" in data
        assert len(data["stats"]) == 3  # Should include all 3 users
        
        # Find each user in stats
        user_stats = {stat["id"]: stat for stat in data["stats"]}
        assert user1.id in user_stats
        assert user2.id in user_stats
        assert user3.id in user_stats
        
        # Check that users with no attendance show zeros
        assert user_stats[user2.id]["present"] == 0
        assert user_stats[user2.id]["tardy"] == 0
        assert user_stats[user2.id]["absent"] == 0
    
    def test_analytics_engine_handles_empty_data(self, client, db_session):
        """Test that analytics engine handles users with no attendance data"""
        # Create test user with no attendance
        user = User(name="No Data User", email="nodata@gmail.com", staff_code="444444")
        db_session.add(user)
        db_session.commit()
        
        # Test analytics engine directly
        analytics = AnalyticsEngine(db_session)
        
        # Test weekly trends
        trends = analytics.get_weekly_trends()
        assert "labels" in trends
        assert len(trends["labels"]) == 7  # 7 days of week
        
        # Test monthly trends
        monthly = analytics.get_monthly_trends()
        assert "labels" in monthly
        
        # Test status distribution
        distribution = analytics.get_status_distribution()
        assert "labels" in distribution
        assert "data" in distribution
    
    def test_analytics_with_mixed_data_types(self, client, db_session):
        """Test analytics with mixed employment types and data"""
        # Create users with different employment types
        users = [
            User(name="Full-time User", email="ft@gmail.com", staff_code="555555", employment_type="Full-time"),
            User(name="Part-time User", email="pt@gmail.com", staff_code="666666", employment_type="Part-time"),
            User(name="Contractor", email="contractor@gmail.com", staff_code="777777", employment_type="Contractor")
        ]
        db_session.add_all(users)
        db_session.commit()
        
        # Add mixed attendance data
        for i, user in enumerate(users):
            for day in range(5):  # 5 days of data
                status = ["Present", "Late", "Absent"][i % 3]
                attendance = Attendance(
                    user_id=user.id,
                    status=status,
                    timestamp=datetime.now() - timedelta(days=day)
                )
                db_session.add(attendance)
        db_session.commit()
        
        # Test analytics with employment type filter
        response = client.get("/analytics")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["stats"]) == 3
        
        # Verify employment types are preserved
        user_stats = {stat["id"]: stat for stat in data["stats"]}
        assert user_stats[users[0].id]["employment_type"] == "Full-time"
        assert user_stats[users[1].id]["employment_type"] == "Part-time"
        assert user_stats[users[2].id]["employment_type"] == "Contractor"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])