# L3 Usage Specification

## Scope

L3 is an analysis layer built on top of L2 outputs. It does not train models and does not own data preprocessing.

## Mandatory Rules

1. L3 must not access L1 artifacts directly.
2. L3 must prioritize loading L2.5 frozen features (`freeze/manifest.json` + `freeze/layers/*.npz`).
3. If frozen features are missing, L3 may fallback to online L2 feature extraction (`mode="online"`).
4. L3 analysis must never trigger retraining.
5. Every analysis result must record the exact `run_name` used.

## Canonical API

Use the unified entrypoint below:

```python
from backend2.l2 import ArtifactManager, load_l2_features_or_fallback

manager = ArtifactManager(
    artifacts_dir="artifacts",
    dataset_id="h5_full",
    exp_name="baseline_unet",
    run_name="run_h5_full_20260228_120000",
)

result = load_l2_features_or_fallback(manager, split="test")
print(result["mode"])      # "freeze" or "online"
print(result["pair_nt"].shape)
print(list(result["layers"].keys()))
```

## Return Contract

`load_l2_features_or_fallback(...)` returns:

- `mode`: `"freeze"` (loaded from disk) or `"online"` (forward fallback)
- `layers`: `{layer_name: np.ndarray}` where each value is `[N, C, H, W]`
- `pair_nt`: `np.ndarray` with shape `[N, 2]`

## Operational Notes

- Frozen feature order strictly follows L2 test pairs order.
- Fallback online extraction runs forward only and does not save `preds_test.npz`.
- Freeze mode currently supports `split="test"` only.
