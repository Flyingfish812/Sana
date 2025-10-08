# debuggers/migrate_ckpt.py
import torch, json
from pathlib import Path
from datetime import datetime

def migrate(old_path: str, new_path: str):
    try:
        ckpt = torch.load(old_path, map_location="cpu", weights_only=True)
    except Exception:
        import torch.serialization as ts
        try:
            from lightning_fabric.utilities.data import AttributeDict
            ts.add_safe_globals([AttributeDict])
        except Exception:
            pass
        ckpt = torch.load(old_path, map_location="cpu", weights_only=False)

    sd = ckpt.get("state_dict", ckpt)
    meta = {"migrated_from": old_path, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    torch.save({"state_dict": sd, "_meta": meta}, new_path)
    Path(new_path + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# 用法：
# migrate("runs/epd_smoketest_unet/model_last.pt", "runs/epd_smoketest_unet/model_last_clean.pt")
