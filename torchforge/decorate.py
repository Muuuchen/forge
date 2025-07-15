from dataclasses import dataclass
from .config import CompileConfig
from typing import Optional,Dict,Any,List
import torch
from .cachemonitor import CacheMonitor
import torch.nn as nn
from pathlib import Path
import json
class TorchCompilerDecorator:
    def __init__(self, config: Optional[CompileConfig] = None):
        self.config = config or CompileConfig()
        self.cache_monitor =CacheMonitor(self.config.cache_dir)
        self.cache_monitor.clear_cache()  # 每次初始化都清理cache
        self._setup_environment()

    def _setup_environment(self):
        """ set up the environment for torch compiler """
        torch._dynamo.reset()
        torch._inductor.codecache.FxGraphCache.clear()
         
        torch._inductor.config.max_autotune = self.config.max_autotune
        torch._inductor.config.max_autotune_gemm_backends = self.config.gemm_backends
        torch._inductor.config.max_autotune_conv_backends = self.config.conv_backends
        torch._inductor.config.debug = True
        torch._inductor.config.trace.enabled = True
        torch._dynamo.config.verbose = self.config.verbose
        torch._dynamo.config.debug_dir_root = self.config.cache_dir

        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        if self.config.verbose:
            print(f"🔧 Torch Inductor Environment Setted!")
            print(f"📁 Cache file Dir: {self.config.cache_dir}")
    def _compile_model(self,model: nn.Module)->nn.Module:
        self.cache_monitor.start_monitoring()
        compiled_model = torch.compile(
            model,
            backend=self.config.backend,
            mode=self.config.compile_mode,
        )
        if self.config.verbose:
            print(f"✅ Model has been Compiled (mode: {self.config.compile_mode})")
        
        return compiled_model
    
    def _get_compilation_artifacts(self) -> Dict[str, Any]:
        """获取编译过程中生成的文件和信息"""
        return self.cache_monitor.get_generated_files()
    
    def read_ir(self, ir_type: str = "ir_pre_fusion") -> Optional[str]:
        """读取指定类型的IR文件内容"""
        return self.cache_monitor.read_ir_file(ir_type)

    def rewrite_ir(self, ir_type: str = "ir_pre_fusion", rewrite_fn=None) -> bool:
        """对指定类型的IR文件内容进行改写"""
        return self.cache_monitor.rewrite_ir_file(ir_type, rewrite_fn)

_default_decorator = TorchCompilerDecorator()

def no_bubble(cls_or_config=None):
    def decorator(cls):
        if not issubclass(cls, nn.Module):
            raise TypeError("The decorated class must be a subclass of nn.Module")
        
        if isinstance(cls_or_config, CompileConfig):
            decorator_instance = TorchCompilerDecorator(cls_or_config)
        else:
            decorator_instance = _default_decorator
        
        original_init = cls.__init__
        original_forward = cls.forward
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._needs_compilation = True
            self._decorator_instance = decorator_instance
            self._compilation_artifacts = None
            self._compiled_model = None
            
        def new_forward(self, *args, **kwargs):
            if getattr(self, '_needs_compilation', False):
                if self._decorator_instance.config.verbose:
                    print(f"🚀 first invoke, Compiling...")
                self._compiled_model = self._decorator_instance._compile_model(self)
                if args:
                    try:
                        with torch.no_grad():
                            _ = self._compiled_model.forward(*args, **kwargs)
                    except Exception as e:
                        if self._decorator_instance.config.verbose:
                            print(f"⚠️ Compiling time error: {e}")
                self._compilation_artifacts = self._decorator_instance._get_compilation_artifacts()

                self._needs_compilation = False
                if self._decorator_instance.config.verbose:
                    print(f"🎉 Compiled done!")
                    print(f"📦 Gen files: {self._compilation_artifacts['total_count']} 个")
                    self._print_artifacts_summary()


        def _print_artifacts_summary(self):
            """打印编译产物摘要"""
            artifacts = self._compilation_artifacts
            if not artifacts:
                return
                
            print(f"   🔹 Triton Kernels: {len(artifacts['output_triton_kernels'])}")
            print(f"   🔹 Pre fusion IR: {len(artifacts['ir_pre_fusion'])}")
            print(f"   🔹 Post fusion IR: {len(artifacts['ir_post_fusion'])}")
        
        def get_compilation_artifacts(self) -> Dict[str, Any]:
            """获取编译产物"""
            return getattr(self, '_compilation_artifacts', {})
        
        def get_triton_kernels_path(self) -> List[Dict[str, Any]]:
            """获取生成的 Triton kernel 文件"""
            artifacts = self.get_compilation_artifacts()
            path = artifacts.get('output_triton_kernels', [])
            return path
        def get_post_ir_path(self) -> List[Dict[str, Any]]:
            """获取生成的 Triton kernel 文件"""
            artifacts = self.get_compilation_artifacts()
            path = artifacts.get('ir_post_fusion', [])
            return path
        def get_pre_ir_path(self) -> List[Dict[str, Any]]:
            """获取生成的 Triton kernel 文件"""
            artifacts = self.get_compilation_artifacts()
            path = artifacts.get('ir_pre_fusion', [])
            return path
        
        
        def get_cpp_code(self) -> List[Dict[str, Any]]:
            """获取生成的 C++ 代码文件"""
            artifacts = self.get_compilation_artifacts()
            return artifacts.get('cpp_code', [])
        
        def save_artifacts_info(self, output_file: str = "compilation_artifacts.json"):
            """保存编译产物信息到文件"""
            artifacts = self.get_compilation_artifacts()
            if artifacts:
                cache_dir = Path(self._decorator_instance.config.cache_dir)
                output_path = cache_dir / output_file
                with open(output_path, 'w') as f:
                    json.dump(artifacts, f, indent=2, default=str)
                print(f"📄 编译产物信息已保存到: {output_path}")
            else:
                print("⚠️ 没有编译产物信息可保存")
        
        def force_recompile(self):
            """强制重新编译"""
            self._needs_compilation = True
            self._compilation_artifacts = None
            if hasattr(self, '_decorator_instance') and self._decorator_instance.config.verbose:
                print("🔄 标记为需要重新编译")
        
   
        cls.__init__ = new_init
        cls._print_artifacts_summary = _print_artifacts_summary
        cls.get_compilation_artifacts = get_compilation_artifacts
        cls.get_triton_kernels_path = get_triton_kernels_path
        cls.get_post_ir_path = get_post_ir_path
        cls.get_pre_ir_path = get_pre_ir_path
        cls.get_cpp_code = get_cpp_code
        cls.save_artifacts_info = save_artifacts_info
        cls.force_recompile = force_recompile
        cls.bubble_free = new_forward
        return cls
    
    if cls_or_config is None or isinstance(cls_or_config, CompileConfig):
        return decorator
    else:
        return decorator(cls_or_config)



