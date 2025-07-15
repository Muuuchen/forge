from torchforge import no_bubble
import torch
from torch import nn

@no_bubble
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.rmsnorm = nn.RMSNorm(5)
        self.linear2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.rmsnorm(x)
        x = self.linear2(x)
        return x

# 使用模型
model = MyModel().to(device='cuda')
x = torch.randn(32, 10).to(device='cuda')
output = model.bubble_free(x)

# 获取编译产物
artifacts = model.get_compilation_artifacts()
print(f"总共生成了 {artifacts['total_count']} 个文件")

print("生成的 Triton kernel 文件:", model.get_triton_kernels_path())