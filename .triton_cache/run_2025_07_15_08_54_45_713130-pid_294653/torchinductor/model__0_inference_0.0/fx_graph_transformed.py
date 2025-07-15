class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[5, 10]", arg1_1: "f32[5]", arg2_1: "f32[32, 10]", arg3_1: "f32[5]", arg4_1: "f32[2, 5]", arg5_1: "f32[2]"):
         # File: /root/TorchForge/main.py:15 in forward, code: x = self.linear(x)
        permute: "f32[10, 5]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[32, 5]" = torch.ops.aten.mm.default(arg2_1, permute);  arg2_1 = permute = None
        add_tensor: "f32[32, 5]" = torch.ops.aten.add.Tensor(mm_default, arg1_1);  mm_default = arg1_1 = None
        
         # File: /root/TorchForge/main.py:16 in forward, code: x = self.relu(x)
        relu: "f32[32, 5]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        
         # File: /root/workspace/.venv/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_1: "f32[32, 5]" = torch.ops.aten.pow.Tensor_Scalar(relu, 2)
        mean: "f32[32, 1]" = torch.ops.aten.mean.dim(pow_1, [1], True);  pow_1 = None
        add: "f32[32, 1]" = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt: "f32[32, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 5]" = torch.ops.aten.mul.Tensor(relu, rsqrt);  relu = rsqrt = None
        mul_1: "f32[32, 5]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        
         # File: /root/TorchForge/main.py:18 in forward, code: x = self.linear2(x)
        permute_1: "f32[5, 2]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm_1: "f32[32, 2]" = torch.ops.aten.addmm.default(arg5_1, mul_1, permute_1);  arg5_1 = mul_1 = permute_1 = None
        return (addmm_1,)
        