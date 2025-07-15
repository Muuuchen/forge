from torchforge import no_bubble
import torch
from torch import nn

@no_bubble
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

# 使用模型
model = MyModel()
x = torch.randn(32, 10)
output = model.bubble_free(x)

# 获取编译产物
artifacts = model.get_compilation_artifacts()
print(f"总共生成了 {artifacts['total_count']} 个文件")

# 获取 Triton kernels
triton_kernels = model.get_triton_kernels()
for i, kernel in enumerate(triton_kernels):
    print(f"Kernel {i}: {kernel['name']} ({kernel['size']} bytes)")

# 获取 C++ 代码
cpp_files = model.get_cpp_code()
for cpp_file in cpp_files:
    print(f"C++ file: {cpp_file['relative_path']}")

# 打印第一个 kernel 的代码
model.print_kernel_code(0)

# 保存所有产物信息
model.save_artifacts_info("my_model_artifacts.json")
