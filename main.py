from torchforge import no_bubble
from torchforge import process_triton_file
import torch
from torch import nn
import os
os.environ['PDL_FLAG'] = '0'

@no_bubble
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.rmsnorm = nn.RMSNorm(128)
        self.linear2 = nn.Linear(128, 256)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.rmsnorm(x)
        x = self.linear2(x)
        return x

# 使用模型
model = MyModel().to(device='cuda')
x = torch.randn(256, 256).to(device='cuda')
output = model.bubble_free(x)

# 获取编译产物
artifacts = model.get_compilation_artifacts()
print(f"总共生成了 {artifacts['total_count']} 个文件")
src_path = str( model.get_triton_kernels_path()[0])

print("生成的 Triton kernel 文件:", src_path)

_, output_path =process_triton_file(src_path)
# print(process_triton_file(str(model.get_triton_kernels_path()[0])))
print(output_path)
print(src_path)

os.environ['PDL_FLAG'] = '0'
print("执行原来的的文件")
exec(open(src_path).read())

os.environ['PDL_FLAG'] = '1'
print("开始执行生成的文件")
exec(open(output_path).read())



