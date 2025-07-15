# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (5, 10), (10, 1))
    assert_size_stride(arg1_1, (5, ), (1, ))
    assert_size_stride(arg2_1, (32, 10), (10, 1))
    buf0 = empty_strided_cpu((32, 5), (5, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(reinterpret_tensor(arg1_1, (32, 5), (0, 1), 0), arg2_1, reinterpret_tensor(arg0_1, (10, 5), (1, 10), 0), alpha=1, beta=1, out=buf0)
    del arg0_1
    del arg1_1
    del arg2_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((5, 10), (10, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((5, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, 10), (10, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
