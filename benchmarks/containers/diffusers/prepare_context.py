"""Create a small allowlisted Docker build context; never send repository data/secrets."""

import argparse
import shutil
from pathlib import Path

from sls_bench.image_diffusers import REQUIREMENTS

parser = argparse.ArgumentParser()
parser.add_argument("destination", type=Path)
args = parser.parse_args()
source = Path(__file__).resolve().parent
args.destination.mkdir(parents=True, exist_ok=False)
for name in ("Dockerfile", "bootstrap.sh", "verify_runtime.py"):
    shutil.copyfile(source / name, args.destination / name)
(args.destination / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n")
(args.destination / ".dockerignore").write_text(
    "*\n!Dockerfile\n!requirements.txt\n!verify_runtime.py\n!bootstrap.sh\n"
)
print(args.destination)
