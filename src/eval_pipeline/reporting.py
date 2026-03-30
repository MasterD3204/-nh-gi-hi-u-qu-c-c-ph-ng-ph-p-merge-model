from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import ensure_directory, json_ready, slugify, write_json


class RunReportWriter:
    def __init__(
        self,
        output_root: str | Path,
        run_name: str | None,
        request_payload: dict[str, Any],
        environment_payload: dict[str, Any],
        registry_payload: dict[str, Any],
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = slugify(run_name or "evaluation")
        self.run_dir = ensure_directory(Path(output_root) / f"{timestamp}_{suffix}")
        self.pairs_dir = ensure_directory(self.run_dir / "pairs")
        self.summary_rows: list[dict[str, Any]] = []

        write_json(self.run_dir / "manifest.json", request_payload)
        write_json(self.run_dir / "environment.json", environment_payload)
        write_json(self.run_dir / "registry_snapshot.json", registry_payload)

    def write_pair(
        self,
        model_name: str,
        dataset_name: str,
        pair_payload: dict[str, Any],
        sample_results: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        pair_slug = f"{slugify(model_name)}__{slugify(dataset_name)}"
        pair_dir = ensure_directory(self.pairs_dir / pair_slug)

        write_json(pair_dir / "pair_manifest.json", pair_payload)
        write_json(pair_dir / "metrics.json", summary)

        samples_path = pair_dir / "samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as handle:
            for sample in sample_results:
                handle.write(json.dumps(json_ready(sample), ensure_ascii=False))
                handle.write("\n")

        summary_row = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            **summary,
            "pair_dir": str(pair_dir),
        }
        self.summary_rows.append(json_ready(summary_row))

    def finalize(self) -> dict[str, Any]:
        overall_summary = {
            "num_pairs": len(self.summary_rows),
            "pairs": self.summary_rows,
        }
        write_json(self.run_dir / "overall_summary.json", overall_summary)
        self._write_summary_csv(self.run_dir / "summary.csv")
        return {
            "run_dir": str(self.run_dir),
            "overall_summary": overall_summary,
        }

    def _write_summary_csv(self, path: Path) -> None:
        if not self.summary_rows:
            return
        fieldnames: list[str] = []
        for row in self.summary_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.summary_rows)

