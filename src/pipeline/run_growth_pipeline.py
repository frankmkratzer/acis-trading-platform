#!/usr/bin/env python3
# File: run_growth_pipeline.py

import subprocess
import logging
import time

logging.basicConfig(
    filename="pipeline_growth.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

scripts = [
    "compute_forward_returns.py",
    "train_ai_growth_model.py",
    "score_ai_growth_model.py",
    "run_rank_growth_stocks.py"
]

def run():
    logger.info("📈 Starting growth pipeline...")
    start = time.time()

    for script in scripts:
        print(f"\n📈 Running {script} ...")
        result = subprocess.run(["python", f"src/training/{script}"])
        if result.returncode != 0:
            logger.error(f"{script} failed.")
            print(f"❌ {script} failed.")
            exit(1)

    print(f"\n✅ Growth pipeline complete in {time.time() - start:.2f} sec")

if __name__ == "__main__":
    run()