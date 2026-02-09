import os
import shutil
import sys

def prompt_yes_no(question):
    while True:
        reply = input(question + " (y/n): ").lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

def prepare_donation():
    # Ensure we are in the project root
    # This script is in scripts/ so we assume cwd is root or we check.
    # If run via "python scripts/prepare_donation.py", cwd is likely root.
    # If run via "cd scripts; python prepare_donation.py", cwd is scripts.
    
    # Let's verify and adjust path if needed.
    # We expect 'data' folder to be in the current working directory or parent.
    
    if os.path.exists('data'):
        base_dir = '.'
    elif os.path.exists('../data'):
        base_dir = '..'
    else:
        print("[ERR] Could not find 'data' directory. Please run this script from the project root.")
        sys.exit(1)

    print("WARNING: This script will DELETE ALL USER DATA to prepare for donation.")
    print("This includes:")
    print("1. All attendance records (Database)")
    print("2. All registered faces (Images)")
    print("3. The trained recognition model")
    print("4. It will NOT delete the configuration or code.")
    
    if not prompt_yes_no("Are you sure you want to proceed?"):
        print("Operation cancelled.")
        sys.exit(0)

    # 1. Delete Database
    db_path = os.path.join(base_dir, 'data', 'attendance.db')
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"[OK] Deleted {db_path}")
        except Exception as e:
            print(f"[ERR] Failed to delete {db_path}: {e}")
    else:
        print(f"[SKIP] {db_path} not found.")

    # 2. Delete Faces
    faces_dir = os.path.join(base_dir, 'data', 'faces')
    if os.path.exists(faces_dir):
        try:
            # Delete all subdirectories but keep the faces dir
            for item in os.listdir(faces_dir):
                item_path = os.path.join(faces_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"[OK] Cleared contents of {faces_dir}")
        except Exception as e:
            print(f"[ERR] Failed to clear {faces_dir}: {e}")
    else:
        os.makedirs(faces_dir, exist_ok=True)
        print(f"[OK] Created empty {faces_dir}")

    # 3. Delete Model
    model_path = os.path.join(base_dir, 'data', 'lbph_model.yml')
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            print(f"[OK] Deleted {model_path}")
        except Exception as e:
            print(f"[ERR] Failed to delete {model_path}: {e}")
    else:
        print(f"[SKIP] {model_path} not found.")

    print("\n=== Donation Preparation Complete ===")
    print("The system is now clean. The database will be recreated automatically on the next run.")
    print("Remember to check your .env file if you want to change secrets before donating.")

if __name__ == "__main__":
    prepare_donation()
