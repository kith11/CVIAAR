import sys
import os

# Add your project directory to the sys.path
project_home = os.path.expanduser('~/mysite')
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Set the environment variable for the database if needed, though app.py handles basedir
# os.environ['DATABASE_URL'] = 'sqlite:////home/yourusername/mysite/data/attendance.db'

from app import app as application
