"""Uni-SuperMimic G1 deployment runtime (contract unisupermimic_v1).

Self-contained NumPy/ONNXRuntime implementation of the Uni-SuperMimic
policy I/O contract, derived only from the Uni-SuperMimic training side:

* obs frame (99)  = [q - q0 - offset(0), dq, body gyro, projected gravity,
                     previous raw action, relative reference rotation 6D]
* obs history     = 10 frames x 99, oldest -> newest, frame-major flatten (990)
* goal            = precomputed window row f of the clip table, 10 x 73 = (730,)
* action target   = q0 + action_scale * a   (episode joint offset = 0)
* safety          = training termination (tilt vs reference) + velocity /
                    action magnitude / finite guards; abort enters DAMP
"""

from .contract import DeployAssets, Clip, load_assets, load_clip
from .core import DeployCore, SafetyCfg

__all__ = ["Clip", "DeployAssets", "DeployCore", "SafetyCfg", "load_assets", "load_clip"]
