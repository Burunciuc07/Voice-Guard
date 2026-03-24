import os
import shutil
try:
    import kagglehub
except ImportError:
    print("kagglehub is not installed. Please run 'pip install kagglehub' first.")
    exit(1)

def setup_dataset():
    print("Downloading 'abdallamohamed312/in-the-wild-audio-deepfake' from Kaggle...")
    try:
        # Download the dataset
        path = kagglehub.dataset_download("mohammedabdeldayem/the-fake-or-real-dataset")
        print(f"Downloaded to {path}")
        
        # Create destination directories
        base_dest = "dataset"
        real_dest = os.path.join(base_dest, "REAL")
        fake_dest = os.path.join(base_dest, "FAKE")
        
        print("Clearing old dataset folders...")
        if os.path.exists(base_dest):
            shutil.rmtree(base_dest)
            
        os.makedirs(real_dest, exist_ok=True)
        os.makedirs(fake_dest, exist_ok=True)
        
        print(f"Copying files to {base_dest}...")
        
        moved_real = 0
        moved_fake = 0
        
        # Locate REAL and FAKE folders in the downloaded dataset
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.wav'):
                    src = os.path.join(root, file)
                    
                    # Path heuristics to determine label (inspecting only the parent directory name, not the full path)
                    parent_dir = os.path.basename(root).lower()
                    
                    if 'real' in parent_dir or 'authentic' in parent_dir:
                        shutil.copy2(src, os.path.join(real_dest, file))
                        moved_real += 1
                    elif 'fake' in parent_dir or 'synthetic' in parent_dir or 'spoof' in parent_dir:
                        shutil.copy2(src, os.path.join(fake_dest, file))
                        moved_fake += 1
                        
        print(f"Dataset setup is complete! Downloaded {moved_real} REAL samples and {moved_fake} FAKE samples.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you are authenticated with Kaggle if required.")
        print("You can set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")

if __name__ == "__main__":
    setup_dataset()
