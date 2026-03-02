from __future__ import annotations

from typing import Any, Dict, List


def summarize_layers(summary_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = list(summary_data.get("records", []))
    by_layer: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        if not name:
            continue
        energy = float(rec.get("energy", 0.0))
        spec = float(rec.get("spec_mean_amp", 0.0))
        bucket = by_layer.setdefault(
            name,
            {
                "name": name,
                "energy_sum": 0.0,
                "spec_sum": 0.0,
                "count": 0,
                "shape": rec.get("shape"),
            },
        )
        bucket["energy_sum"] += energy
        bucket["spec_sum"] += spec
        bucket["count"] += 1

    rows: List[Dict[str, Any]] = []
    for name, v in by_layer.items():
        cnt = max(1, int(v["count"]))
        rows.append(
            {
                "name": name,
                "energy": float(v["energy_sum"]) / cnt,
                "spec_mean_amp": float(v["spec_sum"]) / cnt,
                "count": int(v["count"]),
                "shape": v.get("shape"),
            }
        )
    rows.sort(key=lambda x: x["energy"], reverse=True)
    return rows
