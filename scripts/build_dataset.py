from __future__ import annotations

from alz_ras.config import DEFAULT_CONFIG
from alz_ras.data_pipeline import build_dataset, save_dataset


def main() -> None:
    frame = build_dataset(DEFAULT_CONFIG.data)
    save_dataset(frame, DEFAULT_CONFIG.data)
    print(f"Saved dataset with {len(frame)} rows to {DEFAULT_CONFIG.data.processed_data_path}")


if __name__ == "__main__":
    main()
