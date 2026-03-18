#!/usr/bin/env python3
"""
Simple tests for CVIAAR application fixes

Tests cover:
1. Standby mode functionality
2. Profile editing functionality  
3. Analytics endpoint behavior
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from modules.models import User, Attendance
from modules.analytics_engine import AnalyticsEngine

def test_standby_elements():
    """Test that standby elements are properly defined in HTML"""
    print("Testing standby mode elements...")
    
    # Read the index.html file
    with open('templates/index.html', 'r') as f:
        html_content = f.read()
    
    # Check for required elements
    checks = [
        ('standby-overlay element', 'id="standby-overlay"' in html_content),
        ('standby-names element', 'id="standby-names"' in html_content),
        ('standby-title element', 'class="standby-title"' in html_content),
        ('standby CSS styles', '.standby-overlay' in html_content),
        ('JavaScript functions', 'function markActivity()' in html_content),
        ('Event listeners', "addEventListener('mousemove'" in html_content),
        ('Research members array', 'researchMembers = [' in html_content)
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {check_name}: {status}")
        if not result:
            all_passed = False
    
    return all_passed

def test_profile_edit_elements():
    """Test that profile editing elements exist"""
    print("\nTesting profile editing elements...")
    
    # Check admin.html for edit buttons
    with open('templates/admin.html', 'r') as f:
        admin_html = f.read()
    
    # Check for edit_user.html template
    try:
        with open('templates/edit_user.html', 'r') as f:
            edit_html = f.read()
        edit_template_exists = True
    except FileNotFoundError:
        edit_template_exists = False
    
    checks = [
        ('Edit button in admin', '/edit_user/' in admin_html),
        ('Edit user template exists', edit_template_exists),
        ('Form fields in template', 'name="name"' in edit_html if edit_template_exists else False),
        ('Gmail validation', 'gmail.com' in edit_html if edit_template_exists else False),
        ('Update form action', '/update_user/' in edit_html if edit_template_exists else False)
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {check_name}: {status}")
        if not result:
            all_passed = False
    
    return all_passed

def test_analytics_engine():
    """Test analytics engine with mock data"""
    print("\nTesting analytics engine...")
    
    # Create mock data
    users = [
        {"id": 1, "name": "Test User 1", "employment_type": "Full-time"},
        {"id": 2, "name": "Test User 2", "employment_type": "Part-time"},
        {"id": 3, "name": "Test User 3", "employment_type": "Contractor"}
    ]
    
    # Test analytics engine methods
    try:
        # Test with empty data
        analytics = AnalyticsEngine(None)  # Mock database session
        
        # Test weekly trends with empty data
        trends = analytics.get_weekly_trends()
        weekly_ok = 'labels' in trends and len(trends['labels']) == 7
        
        # Test monthly trends with empty data
        monthly = analytics.get_monthly_trends()
        monthly_ok = 'labels' in monthly
        
        # Test status distribution with empty data
        distribution = analytics.get_status_distribution()
        distribution_ok = 'labels' in distribution and 'data' in distribution
        
        # Test risk prediction with empty data
        risks = analytics.predict_risk_users()
        risks_ok = isinstance(risks, list)
        
        checks = [
            ('Weekly trends with empty data', weekly_ok),
            ('Monthly trends with empty data', monthly_ok),
            ('Status distribution with empty data', distribution_ok),
            ('Risk prediction with empty data', risks_ok)
        ]
        
        all_passed = True
        for check_name, result in checks:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {check_name}: {status}")
            if not result:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"  ✗ FAIL: Analytics engine test failed: {e}")
        return False

def test_auto_absence_feature():
    """Test auto-absence marking feature"""
    print("\nTesting auto-absence marking feature...")
    
    # Check sync_engine.py for auto-absence logic
    with open('modules/sync_engine.py', 'r') as f:
        sync_content = f.read()
    
    checks = [
        ('Auto-absence function exists', '_auto_mark_absent_for_date' in sync_content),
        ('Date checking logic', 'yesterday' in sync_content or 'target_date' in sync_content),
        ('Absent status creation', 'status="Absent"' in sync_content),
        ('Schedule start time usage', 'schedule_start' in sync_content)
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {check_name}: {status}")
        if not result:
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("CVIAAR Application Fixes Test Suite")
    print("=" * 40)
    
    tests = [
        ("Standby Mode", test_standby_elements),
        ("Profile Editing", test_profile_edit_elements),
        ("Analytics Engine", test_analytics_engine),
        ("Auto-Absence Feature", test_auto_absence_feature)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n✗ FAIL: {test_name} test failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())