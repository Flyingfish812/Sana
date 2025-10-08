## dataio/

### 目标

将任意原始数据集（h5 / nc / mat / 未来自定义格式）**规范化为统一 5D 数组** `[N,T,H,W,C]`，并在**不改其它模块**的情况下，通过配置即可完成：

1. 读取与标准化 → 2) 采样拼片 → 3) 变换增强 → 4) 适配模型输入 → 5) DataLoader → 6) 批次与元信息落盘。

### 目录结构（dataio）

```
backend/dataio/
  api.py                 # 统一入口：build_dataset / build_dataloader / run / summarise
  __init__.py            # 侧效导入，完成 readers/samplers/adapters 的注册
  registry.py            # 轻量注册表（reader/transform/sampler/adapter）
  schema.py              # DataMeta / ArraySample / AdapterOutput 数据契约
  utils/typing.py        # 类型别名
  readers/               # 数据读取：输出统一的 [N,T,H,W,C] + DataMeta
    base.py              # BaseReader 抽象类
    h5.py                # H5Reader（支持多键聚合到 C 维）
    nc.py                # NCReader（xarray；可选 u/v→涡量）
    mat.py               # MatReader（兼容 v7 / v7.3；NOAA 风格重塑）
  sampling/              # 样本规范与采样器
    base.py              # SampleSpec / ListSampler
    __init__.py          # build_sampler_from_config（最小工厂）
    registry_hooks.py    # 注入 "static" / "multi_frame" 两个采样器
  transforms/            # 逐样本变换（作用于 ArraySample）
    base.py              # Transform 协议
    compose.py           # Compose 串联
    normalize.py         # Normalizer / NormalizeTransform / InverseNormalizeTransform
    coords.py            # AddCoordsTransform（[-1,1] 网格坐标）
    time_encoding.py     # AddTimeEncodingTransform（简单正弦时间编码）
    to_tensor.py         # ToTensorTransform（可选）
    __init__.py
  dataset/
    unified.py           # UnifiedDataset（Array5D + SampleSpec 列表 + Transforms）
    adapters.py          # static2d / timecond2d 适配器（输出 x/y/cond）
    collate.py           # make_collate(adapter_fn) → DataLoader 可用的 collate
  cache/torch_cache.py   # CacheManager：array5d / batch_*.pt / meta.json / summary.json
  io/exporters.py        # dump_prep_outputs()：统一导出入口
```

### 数据流（从 config 到可训练）

```
Reader([N,T,H,W,C]) → Sampler(SampleSpec...) → UnifiedDataset(ArraySample) 
→ Transforms(可选：Normalize/Coords/TimeEnc/ToTensor)
→ Adapter(static2d|timecond2d) → Collate → DataLoader
→ Export(cache): array5d.npy / batch_*.pt / meta.json / summary.json
```

### 扩展契约

* **新增数据集**：实现 `BaseReader` 子类，`probe()` / `read_array5d()` 输出严格的 `[N,T,H,W,C]` 与 `DataMeta`；在文件底部用 `@register_reader("your_kind")` 注册即可。其它模块**无需改动**。
* **新增采样**：实现函数返回 `ListSampler(SampleSpec...)`，并 `@register_sampler("name")`。
* **新增变换**：实现 `Transform` 协议（`__call__(ArraySample)->ArraySample`），在 `api._build_transforms()` 中按配置启用。
* **切换适配器**：在配置里 `adapter.kind` 选择（或新增并 `@register_adapter`）。

### 典型配置片段

```yaml
reader:
  kind: h5
  path: ./datasets/foo.h5
  dataset: data
  times_key: times
  fill_value: 0.0

sampler:
  kind: multi_frame
  history: 4
  t_stride: 1
  n: 0

transforms:
  normalize: { method: zscore }
  add_coords: true
  add_time_encoding: true
  to_tensor: false

adapter:
  kind: timecond2d

batch_size: 4
shuffle: true

output:
  out_dir: ./prep_out/foo
  save_array5d: true       # 保存 Reader 统一后的 5D 原始数组
  save_batches: true       # 保存 collate 后的批次（含变换与适配）
  num_batch_dump: 8
  overwrite: true
```

### 入口用法

* 代码内调用：

```python
import backend.dataio as _   # 确保注册生效
from backend.dataio.api import run

dataset, dataloader, summary = run(config)
```

* 命令行：

```
python scripts/run_prep.py --config examples/configs/h5_static.yaml
```