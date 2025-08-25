#!/usr/bin/env python3
# File: run_rank_growth_stocks.py
# Purpose: Run model training and scoring for growth strategy

import subprocess
import logging
import time
from datetime import datetime

logging.basicConfig(
    filename="rank_growth_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

scripts = [
    "train_ai_growth_model.py",
    "score_ai_growth_model.py",
    "compute_ai_growth_score_labels.py",
    "populate_ai_growth_scores.py"
]

def run_pipeline():
    logger.info("🚀 Starting growth model pipeline run...")
    start_time = time.time()

    for script in scripts:
        logger.info(f"▶️ Running: {script}")
        print(f"\n🚀 Running {script} ...")

        step_start = time.time()
        result = subprocess.run(["python", script])
        elapsed = time.time() - step_start

        if result.returncode != 0:
            logger.error(f"❌ Script {script} failed. Aborting pipeline.")
            print(f"❌ {script} failed. Check logs.")
            exit(1)

        logger.info(f"✅ Finished {script} in {elapsed:.2f} sec")
        print(f"✅ Finished {script} ({elapsed:.2f} sec)")

    total_time = time.time() - start_time
    logger.info(f"🎉 Growth pipeline complete in {total_time:.2f} seconds")
    print(f"\n🎉 Growth pipeline completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    run_pipeline()
