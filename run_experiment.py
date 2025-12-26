import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n>>> STARTING: {script_name}")
    print("=" * 50)
    
    # We use sys.executable to make sure it uses the same python that is running this script
    result = subprocess.run([sys.executable, f"scripts/{script_name}"])
    
    if result.returncode != 0:
        print(f"\n!!! ERROR: {script_name} crashed or was stopped.")
        print("Stopping experiment to prevent wasted time.")
        sys.exit(1)
    
    print("=" * 50)
    print(f">>> FINISHED: {script_name}\n")

def main():
    start_time = time.time()
    print("--- AUTOMATED TWIN EXPERIMENT STARTED ---\n")

    # Step 1: Train Standard (Twin A)
    run_script("run_standard_cifar.py")

    # Step 2: Train Biomimetic (Twin B)
    run_script("run_biomimetic_cifar.py")

    # Step 3: Compare Results
    run_script("evaluate_robustness.py")

    total_time = (time.time() - start_time) / 3600
    print(f"\nALL DONE! The experiment took {round(total_time, 2)} hours.")
    print("Check the results table above.")

if __name__ == "__main__":
    main()