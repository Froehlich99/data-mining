"""Download the MEBeauty and SCUT-FBP5500 datasets (pinned commits)."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    {
        "url": "https://github.com/fbplab/MEBeauty-database.git",
        "commit": "7b849562ee92d99d34d56afa2ebe85e5075f9b20",
        "target": "MEBeauty-database-main",
    },
    {
        "url": "https://github.com/HCIILAB/SCUT-FBP5500-Database-Release.git",
        "commit": "bff34ad298ae80e8f9a3e15bbd7290a32b620446",
        "target": "SCUT-FBP5500_v2",
    },
]


def main():
    for ds in DATASETS:
        dest = PROJECT_ROOT / ds["target"]
        if dest.exists():
            print(f"Skipping {ds['target']} (already exists)")
            continue

        print(f"Cloning {ds['target']} ...")
        subprocess.run(
            ["git", "clone", "--single-branch", ds["url"], str(dest)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", ds["commit"]],
            check=True,
        )
        print(f"  Pinned to {ds['commit'][:12]}")

    print("Done.")


if __name__ == "__main__":
    main()
