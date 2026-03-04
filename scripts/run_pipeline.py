"""
run_pipeline.py
Orchestrates the full Kickstarter ML pipeline:
  Step 1 - Data Engineering (ingestion → Parquet)
  Step 2 - ML Training (LR, RF, GBT + CrossValidator)
Run from workspace root:
    python scripts/run_pipeline.py
"""

import subprocess
import sys
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(label: str, script_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        check=False
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[FAILED] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[OK] {label} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    run_step("Data Engineering", os.path.join(BASE_DIR, "scripts", "data_engineering.py"))
    run_step("ML Training Pipeline", os.path.join(BASE_DIR, "main.py"))
    print("\nPipeline completed successfully.")
