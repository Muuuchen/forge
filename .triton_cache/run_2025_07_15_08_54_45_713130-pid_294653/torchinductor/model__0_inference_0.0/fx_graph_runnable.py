
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        permute = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        relu = torch.ops.aten.relu.default(addmm);  addmm = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(relu, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [1], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(relu, rsqrt);  relu = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg5_1, mul_1, permute_1);  arg5_1 = mul_1 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 200, device=device(type='cuda', index=0))
    reader.tensor(buf0, (5, 10), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 20, device=device(type='cuda', index=0))
    reader.tensor(buf1, (5,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 10), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 20, device=device(type='cuda', index=0))
    reader.tensor(buf3, (5,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2, 5), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2,), is_leaf=True)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)