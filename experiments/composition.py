from __future__ import annotations

import _bootstrap  # noqa: F401  (puts the package root on the import path)

from comparators.galore_adapter import galore_available
from comparators.subtrack_adapter import subtrack_available


def main() -> None:
    print({"galore_available": galore_available(), "subtrack_available": subtrack_available()})


if __name__ == "__main__":
    main()
