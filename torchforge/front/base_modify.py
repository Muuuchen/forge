import ast
import re
import os

def modify_triton_kernels_for_pdl(file_path):
    """
    对源码中的triton kernel进行PDL优化修改
    
    Args:
        file_path (str): 源代码文件路径
    
    Returns:
        str: 修改后的代码字符串
    """
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    
    # 查找所有triton kernel定义
    kernel_pattern = r"async_compile\.triton\([^,]+,\s*'''(.*?)'''"
    
    def modify_kernel_code(match):
        kernel_code = match.group(1)
        
        # 解析kernel代码，找到函数定义
        try:
            # 添加PDL函数定义到kernel代码开头
            modified_kernel = add_pdl_functions_to_kernel(kernel_code)
            
            # 在首个tl.load之前添加gdc_wait()
            modified_kernel = add_gdc_wait_before_first_load(modified_kernel)
            
            # 在epilogue部分添加gdc_launch_dependents()
            modified_kernel = add_gdc_launch_dependents_to_epilogue(modified_kernel)
            
            return match.group(0).replace(kernel_code, modified_kernel)
            
        except Exception as e:
            print(f"Warning: Failed to modify kernel: {e}")
            return match.group(0)
    
    # 应用修改
    modified_content = re.sub(kernel_pattern, modify_kernel_code, content, flags=re.DOTALL)
    
    print("Modified code:")
    print("=" * 80)
    print(modified_content)
    print("=" * 80)
    
    return modified_content
def add_pdl_functions_to_kernel(kernel_code):
        """在kernel代码最前面添加PDL函数定义"""
        pdl_code = '''
from triton.language import core
@core.extern
def gdc_wait(_builder=None):
    core.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=core.int32, is_pure=False, pack=1,_builder=_builder)

@core.extern
def gdc_launch_dependents(_builder=None):
    core.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [], dtype=core.int32,is_pure=False, pack=1, _builder=_builder)

    '''
        return pdl_code + kernel_code

def add_gdc_wait_before_first_load(kernel_code):
    """在首个tl.load之前添加gdc_wait()"""
    lines = kernel_code.split('\n')
    
    # 找到函数定义开始
    func_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def ') and '(' in line:
            func_start = i
            break
    
    if func_start == -1:
        return kernel_code
    
    # 找到首个tl.load
    first_load_idx = -1
    for i in range(func_start, len(lines)):
        if 'tl.load(' in lines[i]:
            first_load_idx = i
            break
    
    if first_load_idx == -1:
        return kernel_code
    
    # 在首个tl.load之前添加gdc_wait()
    indent = get_line_indent(lines[first_load_idx])
    gdc_wait_line = f"{indent}gdc_wait()"
    lines.insert(first_load_idx, gdc_wait_line)
    
    return '\n'.join(lines)

def add_gdc_launch_dependents_to_epilogue(kernel_code):
    """在epilogue部分（tl.store之前）添加gdc_launch_dependents()"""
    lines = kernel_code.split('\n')
    
    # 找到最后一个tl.store的位置
    last_store_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if 'tl.store(' in lines[i]:
            last_store_idx = i
            break
    
    if last_store_idx == -1:
        return kernel_code
    
    # 在最后一个tl.store之前添加gdc_launch_dependents()
    indent = get_line_indent(lines[last_store_idx])
    gdc_launch_line = f"{indent}gdc_launch_dependents()"
    lines.insert(last_store_idx, gdc_launch_line)
    
    return '\n'.join(lines)

def get_line_indent(line):
    """获取行的缩进"""
    return line[:len(line) - len(line.lstrip())]

# 使用AST方法的替代实现
class TritonKernelModifier(ast.NodeTransformer):
    """使用AST方式修改triton kernel"""
    
    def __init__(self):
        self.first_load_found = False
        self.gdc_wait_added = False
    
    def visit_Call(self, node):
        # 检查是否是tl.load调用
        if (isinstance(node.func, ast.Attribute) and 
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'tl' and 
            node.func.attr == 'load'):
            
            if not self.first_load_found:
                self.first_load_found = True
                # 创建gdc_wait()调用节点
                gdc_wait_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id='gdc_wait', ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    )
                )
                return [gdc_wait_call, node]
        
        # 检查是否是tl.store调用
        if (isinstance(node.func, ast.Attribute) and 
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'tl' and 
            node.func.attr == 'store'):
            
            # 创建gdc_launch_dependents()调用节点
            gdc_launch_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='gdc_launch_dependents', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )
            )
            return [gdc_launch_call, node]
        
        return self.generic_visit(node)

def modify_kernel_with_ast(kernel_code):
    """使用AST方法修改kernel代码"""
    try:
        # 解析kernel代码
        tree = ast.parse(kernel_code)
        
        # 应用修改
        modifier = TritonKernelModifier()
        modified_tree = modifier.visit(tree)
        
        # 转换回代码
        import astor
        return astor.to_source(modified_tree)
    except Exception as e:
        print(f"AST modification failed: {e}")
        return kernel_code

# 主函数
def process_triton_file(file_path):
    """
    处理包含triton kernel的文件
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        str: 修改后的代码
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    modified_code = modify_triton_kernels_for_pdl(file_path)
    
    # 可选：保存修改后的代码到新文件
    output_path = file_path.replace('.py', '_modified.py')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(modified_code)
    
    print(f"Modified code saved to: {output_path}")
    
    return modified_code, output_path

# 使用示例
if __name__ == "__main__":
    # 使用示例
    file_path = "your_triton_code.py"  # 替换为实际文件路径
    
    try:
        result = process_triton_file(file_path)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error processing file: {e}")