#!/usr/bin/env python3
# File: run_value_pipeline.py

import subprocess
import logging
import time

logging.basicConfig(
    filename="pipeline_value.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

scripts = [
    "compute_forward_returns.py",
    "compute_sp500_outperformance.py",
    "compute_ai_value_scores.py",
]

def run():
    logger.info("💰 Starting value pipeline...")
    start = time.time()

    for script in scripts:
        print(f"\n💰 Running {script} ...")
        result = subprocess.run(["python", script])
        if result.returncode != 0:
            logger.error(f"{script} failed.")
            print(f"❌ {script} failed.")
            exit(1)

    print(f"\n✅ Value pipeline complete in {time.time() - start:.2f} sec")

if __name__ == "__main__":
    run()
