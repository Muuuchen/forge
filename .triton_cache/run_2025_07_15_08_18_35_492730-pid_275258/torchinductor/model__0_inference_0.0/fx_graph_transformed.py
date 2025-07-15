class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[5, 10]", arg1_1: "f32[5]", arg2_1: "f32[32, 10]"):
         # File: /root/TorchForge/main.py:12 in forward, code: return self.linear(x)
        permute: "f32[10, 5]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "f32[32, 5]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        return (addmm,)
        