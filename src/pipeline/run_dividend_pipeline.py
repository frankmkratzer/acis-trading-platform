#!/usr/bin/env python3
# File: run_dividend_pipeline.py

import subprocess
import logging
import time

logging.basicConfig(
    filename="pipeline_dividend.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

scripts = [
    "compute_dividend_growth_scores.py",
    "compute_ai_dividend_scores.py",
    "run_rank_dividend_stocks.py"
]

def run():
    logger.info("💰 Starting dividend pipeline...")
    start = time.time()

    for script in scripts:
        print(f"\n💰 Running {script} ...")
        result = subprocess.run(["python", f"src/scoring/{script}"])
        if result.returncode != 0:
            logger.error(f"{script} failed.")
            print(f"❌ {script} failed.")
            exit(1)

    print(f"\n✅ Dividend pipeline complete in {time.time() - start:.2f} sec")

if __name__ == "__main__":
    run()