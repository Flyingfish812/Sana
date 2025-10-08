# backend/dataio/__init__.py
# 通过导入副作用完成注册（读者/采样器/适配器等）
from .dataset.adapters import *           # 注册 adapters: static2d, timecond2d
from .sampling.registry_hooks import *    # 注册 samplers: static, multi_frame
from .readers.h5 import *                 # 注册 reader: h5
from .readers.nc import *                 # 注册 reader: nc
from .readers.mat import *                # 注册 reader: mat
