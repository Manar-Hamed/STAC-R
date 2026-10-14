"""Shortcut for ``python -m stacr train``."""

import sys
from stacr.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
