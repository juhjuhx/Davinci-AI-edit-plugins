import os
import sys
import logging
from typing import Optional, Any

logger = logging.getLogger("smart_aroll.utils")


def clean_path(p: str) -> str:
    for ch in ('\u200b', '\u200e', '\u200f', '\u202a', '\u202b', '\u202c', '\u202d', '\u202e'):
        p = p.replace(ch, '')
    return p.strip()


def fix_dll_path():
    if sys.platform != "win32":
        return
    dirs = []
    for env_var in ("LLAMA_CPP_PATH", "FFMPEG_PATH"):
        p = os.environ.get(env_var, "")
        if p and os.path.isdir(p):
            dirs.append(p)
    conda = os.environ.get("CONDA_PREFIX", "")
    if conda:
        dirs.extend([
            os.path.join(conda, "lib", "site-packages", "llama_cpp", "lib"),
            os.path.join(conda, "Library", "bin"),
            os.path.join(conda, "DLLs"),
            os.path.join(conda, "bin"),
        ])
    for ver in ("v12.4", "v12.3", "v12.2", "v12.1", "v12.0", "v11.8"):
        dirs.append(rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}\bin")
    for ver in ("v9.0", "v8.9", "v8.8"):
        dirs.append(rf"C:\Program Files\NVIDIA\CUDNN\{ver}\bin")
    dirs.append(r"C:\Windows\System32")
    seen = set()
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        real = os.path.realpath(d)
        if real in seen:
            continue
        seen.add(real)
        try:
            os.add_dll_directory(d)
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


class LazyImport:
    def __init__(self, import_path: str, name: str = None):
        self._import_path = import_path
        self._name = name or import_path.split(".")[-1]
        self._module = None
        self._error = None

    def load(self) -> Optional[Any]:
        if self._module is not None:
            return self._module
        try:
            import importlib
            self._module = importlib.import_module(self._import_path)
            return self._module
        except ImportError as e:
            self._error = str(e)
            return None

    @property
    def is_available(self) -> bool:
        return self.load() is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def get_attr(self, attr_name: str, default=None):
        mod = self.load()
        if mod is None:
            return default
        return getattr(mod, attr_name, default)


def lazy_import(import_path: str, name: str = None) -> LazyImport:
    return LazyImport(import_path, name)
