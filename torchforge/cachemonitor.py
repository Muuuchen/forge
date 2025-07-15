from pathlib import Path
import time
from typing import Dict, Any, Optional
class CacheMonitor:
    def __init__(self, cache_dir:str):
        self.cache_dir = Path(cache_dir)
        self.initial_files = set()
        self.monitoring = False

    def start_monitoring(self):
        """Start monitoring the cache directory for changes."""
        self.initial_files = self._get_all_files()
        self.monitoring = True

    def _get_all_files(self):
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory {self.cache_dir} does not exist.")
        return set(self.cache_dir.rglob("*"))
    def get_generated_files(self):
        """Get the files that have been generated since monitoring started."""
        if not self.monitoring:
            raise RuntimeError("Monitoring has not been started.")
        current_files = self._get_all_files()
        generated_files = current_files - self.initial_files
        
        artifacts = {
            "output_triton_kernels": [],
            "ir_pre_fusion": [],
            "ir_post_fusion": [],   
            "total_count": len(generated_files),  
        }
        for file_path in generated_files:
            if file_path.isfile():
                file_info = self._analyze_file(file_path)
                artifacts["file_details"].append(file_info)
                
                if file_info["type"] == "output_triton_kernels":
                    artifacts["output_triton_kernels"].append(file_path)
                elif file_info["type"] == "ir_pre_fusion":
                    artifacts["ir_pre_fusion"].append(file_path)
                elif file_info["type"] == "ir_post_fusion":
                    artifacts["ir_post_fusion"].append(file_path)
                else :
                    artifacts["other_files"].append(file_path)
        return artifacts
    
    def _analyze_file(self, file_path: Path) ->Dict[Any, str]:
        file_info = {
            "path" : str(file_path),
            "size": file_path.stat().st_size,
            "name":file_path.name,
            "type": "unknown"
        }
        
        if file_path.name == "output_code.py":
            file_info["type"] = "output_triton_kernels"
        elif "pre_fusion" in file_path.name:
            file_info["type"] = "ir_pre_fusion"
        elif "post_fusion" in file_path.name:
            file_info["type"] = "ir_post_fusion"
        elif file_path.suffix in [".so", ".o"]:
            file_info["type"] = "binary"
        return file_info
    def _safe_read_text(self,file_path:Path, max_chars:int = None) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore ') as f:
                content = f.read(max_chars) if max_chars else f.read()
                return content
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
                
    def stop_monitoring(self):
        """Stop monitoring the cache directory."""
        self.monitoring = False

    def clear_cache(self):
        """删除cache目录下的所有文件和子目录"""
        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for sub in item.rglob("*"):
                        if sub.is_file():
                            sub.unlink()
                        elif sub.is_dir():
                            sub.rmdir()
                    item.rmdir()

    def read_ir_file(self, ir_type: str = "ir_pre_fusion") -> Optional[str]:
        """读取指定类型的IR文件内容（如 pre_fusion/post_fusion）"""
        ir_files = list(self.cache_dir.glob(f"*{ir_type}*"))
        if ir_files:
            return self._safe_read_text(ir_files[0])
        return None

    def rewrite_ir_file(self, ir_type: str = "ir_pre_fusion", rewrite_fn=None) -> bool:
        """对指定类型的IR文件内容进行改写，rewrite_fn为内容处理函数"""
        ir_files = list(self.cache_dir.glob(f"*{ir_type}*"))
        if ir_files and rewrite_fn:
            content = self._safe_read_text(ir_files[0])
            new_content = rewrite_fn(content)
            with open(ir_files[0], 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False




import torch
import torch.nn as nn
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class CompileConfig:
    cache_dir: str = "./.triton_cache"
    max_autotune: bool = True
    compile_mode: str = "max-autotune"  # "default", "reduce-overhead", "max-autotune"
    backend: str = "inductor"
    gemm_backends: str = "TRITON"
    conv_backends: str = "TRITON"
    use_cutlass: bool = True
    verbose: bool = True # 是否开启详细日志
    auto_profile: bool = False  # 是否自动进行性能分析
    warmup_runs: int = 3
    profile_runs: int = 10