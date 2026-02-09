import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.face_engine import FaceEngine

def main():
    print("Initializing FaceEngine...")
    # Initialize with default paths (relative to project root)
    # We need to be careful about CWD. 
    # The script will be run from project root ideally.
    
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(basedir, 'data', 'lbph_model.yml')
    faces_dir = os.path.join(basedir, 'data', 'faces')
    
    engine = FaceEngine(model_path=model_path, faces_dir=faces_dir)
    
    print("Starting training with new preprocessing (Resize + EqualizeHist)...")
    if engine.train_model():
        print("Training successful! New model saved.")
    else:
        print("Training failed or no faces found.")

if __name__ == "__main__":
    main()
