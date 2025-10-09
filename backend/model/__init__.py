# backend/model/__init__.py
# 先导入内置组件，触发 @register
from .encoders import unet_base as _enc_unet_base  # noqa: F401
from .encoders import vit as _enc_vit  # noqa: F401
from .propagators import identity as _prop_identity  # noqa: F401
from .propagators import vit as _prop_vit  # noqa: F401
from .decoders import unet_decoder as _dec_unet  # noqa: F401
from .decoders import vit as _dec_vit  # noqa: F401
from .heads import pixel_head as _head_pixel  # noqa: F401

from .epd_system import EPDSystem
from .factory import build_component, register
