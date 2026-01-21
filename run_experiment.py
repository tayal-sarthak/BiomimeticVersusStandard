import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n>>> STARTING: {script_name}")
    print("=" * 50)
    
    ## using sys.executable to run with same python
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

    ## step 1: training standard
    run_script("run_standard_cifar.py")

    ## step 2: training biomimetic
    run_script("run_biomimetic_cifar.py")

    ## step 3: compare results
    run_script("evaluate_robustness.py")

    total_time = (time.time() - start_time) / 3600
    print(f"\nALL DONE! The experiment took {round(total_time, 2)} hours.")
    print("Check the results table above.")

if __name__ == "__main__":
    main()