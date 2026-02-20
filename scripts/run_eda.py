from __future__ import annotations

from pathlib import Path

import pandas as pd

from alz_ras.config import DEFAULT_CONFIG
from alz_ras.eda import build_eda_artifacts


def main() -> None:
    frame = pd.read_parquet(DEFAULT_CONFIG.data.processed_data_path)
    report = build_eda_artifacts(frame, Path("artifacts/eda"))
    print(report)


if __name__ == "__main__":
    main()
