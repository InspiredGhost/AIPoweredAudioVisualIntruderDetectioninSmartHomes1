import subprocess
import sys
import os

# Helper to run a script and check for errors
def run_script(script, args=None):
    cmd = [sys.executable, script]
    if args:
        cmd += args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Script {script} failed with exit code {result.returncode}")

if __name__ == "__main__":
    # 1. Generate processed_data.csv
    run_script("create_processed_data_csv.py")

    # 2. Extract features from videos
    run_script("feature_extraction.py", ["--csv", "processed_data.csv", "--output", "features.npz"])

    # 3. Train the model
    run_script("train.py")

    # 4. Run inference
    run_script("inference.py")

    print("Pipeline completed successfully.")

