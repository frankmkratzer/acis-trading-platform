#!/usr/bin/env python3
# File: run_all_strategies.py

import subprocess

for script in [
    "run_value_pipeline.py",
    "run_growth_pipeline.py",
    "run_dividend_pipeline.py",
]:
    print(f"\n🚀 Running strategy pipeline: {script}")
    result = subprocess.run(["python", script])
    if result.returncode != 0:
        print(f"❌ {script} failed.")
        exit(1)

print("\n🎉 All strategy pipelines complete.")
