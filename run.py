# Full Test
import yaml
cfg = yaml.safe_load(open("examples/train_configs/train_vit_base_nc.yaml", "r"))

from backend.train.runner import run_training
model, artefacts = run_training(cfg)
print(artefacts)
