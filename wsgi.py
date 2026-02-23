import sys
import os

# Add your project directory to the sys.path
project_home = os.path.expanduser('~/mysite')
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

from app import app as application
