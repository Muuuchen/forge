import torch
import torch.nn as nn
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class CompileConfig:
    cache_dir: str = "./.triton_cache"
    max_autotune: bool = True
    compile_mode: str = "max-autotune"  # "default", "reduce-overhead", "max-autotune"
    backend: str = "inductor"
    gemm_backends: str = "TRITON"
    conv_backends: str = "TRITON"
    use_cutlass: bool = False
    verbose: bool = True # 是否开启详细日志
    auto_profile: bool = False  # 是否自动进行性能分析
    warmup_runs: int = 3
    profile_runs: int = 10