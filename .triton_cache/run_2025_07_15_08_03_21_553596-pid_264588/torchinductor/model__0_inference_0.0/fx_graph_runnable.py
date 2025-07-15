
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.verbose = True
torch._dynamo.config.debug_dir_root = './.triton_cache'
torch._inductor.config.debug = True
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm_backends = 'TRITON'
torch._inductor.config.max_autotune_conv_backends = 'TRITON'
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.1+cu126
# torch cuda version: 12.6
# torch git version: e2d141dbde55c2a4370fac5165b0561b6af4798b


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Wed_Jan_15_19:20:09_PST_2025 
# Cuda compilation tools, release 12.8, V12.8.61 
# Build cuda_12.8.r12.8/compiler.35404655_0 

# GPU Hardware Info: 
# NVIDIA H100 PCIe : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        permute = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        return (addmm,)
        
def load_args(reader):
    buf0 = reader.storage(None, 200)
    reader.tensor(buf0, (5, 10), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 20)
    reader.tensor(buf1, (5,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1280)
    reader.tensor(buf2, (32, 10), is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)