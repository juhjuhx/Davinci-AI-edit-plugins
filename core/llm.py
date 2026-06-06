import logging
import os
from typing import Any, Dict, List, Optional

from .utils import fix_dll_path, lazy_import

logger = logging.getLogger("smart_aroll.llm")

fix_dll_path()

_llama = lazy_import("llama_cpp", "Llama")


PROMPT_TEMPLATE = """You are a video editing assistant. Decide if this Chinese speech segment should be CUT.
Answer "KEEP" or "CUT" with a brief reason.

Rules:
- Filler words (um, ah, er, that, then) should CUT
- Repeated words should CUT
- Complete sentences should KEEP
- Self-corrections should KEEP the corrected part

Segment: "{text}"

Answer:"""


class LLMAnalyzer:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        max_tokens: int = 256,
        temperature: float = 0.05,
        use_flash_attention: bool = False,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_flash_attention = use_flash_attention
        self._llm = None
        self._enabled = False
        self._load_error = None

    def load(self) -> bool:
        Llama = _llama.get_attr("Llama")  # noqa: N806
        if Llama is None:
            self._load_error = "llama-cpp-python not installed"
            return False

        if not os.path.isfile(self.model_path):
            self._load_error = f"Model not found: {self.model_path}"
            logger.error(self._load_error)
            return False

        try:
            logger.info(f"Loading LLM: {self.model_path}")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                use_mmap=True,
                use_mlock=False,
                flash_attn=self.use_flash_attention,
                verbose=False,
            )
            self._enabled = True
            logger.info("LLM loaded")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"LLM load failed: {e}")
            self._llm = None
            return False

    def unload(self):
        if self._llm is not None:
            try:
                del self._llm
            except Exception:
                pass
            self._llm = None
            self._enabled = False
            import gc

            gc.collect()

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._llm is not None

    def analyze_segment(self, text: str) -> Dict[str, Any]:
        if not self.is_ready:
            return {"should_cut": False, "reason": "LLM not ready", "verified": False}

        try:
            prompt = PROMPT_TEMPLATE.format(text=text[:200])
            response = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["\n\n", "Segment:", "Answer:"],
            )
            text_out = response["choices"][0]["text"].strip()
            should_cut = "CUT" in text_out.upper()
            reason = text_out[:100]
            return {"should_cut": should_cut, "reason": reason, "verified": True}
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"should_cut": False, "reason": f"Error: {e}", "verified": False}

    def batch_analyze(self, texts: List[str], callback=None) -> List[Dict[str, Any]]:
        results = []
        for i, text in enumerate(texts):
            r = self.analyze_segment(text)
            results.append(r)
            if callback:
                callback(i + 1, len(texts))
        return results


_global_llm: Optional[LLMAnalyzer] = None


def get_llm(config=None) -> LLMAnalyzer:
    global _global_llm
    if _global_llm is None:
        if config and config.llm.get("enabled", False):
            _global_llm = LLMAnalyzer(
                model_path=config.llm.get("model_path"),
                n_ctx=config.llm.get("n_ctx", 2048),
                n_threads=config.llm.get("n_threads", 4),
                n_gpu_layers=config.llm.get("n_gpu_layers", 0),
                max_tokens=config.llm.get("max_tokens", 256),
                temperature=config.llm.get("temperature", 0.05),
                use_flash_attention=config.llm.get("use_flash_attention", False),
            )
            _global_llm.load()
        else:
            _global_llm = LLMAnalyzer(model_path="")
    return _global_llm


def reset_llm():
    global _global_llm
    if _global_llm is not None:
        _global_llm.unload()
    _global_llm = None
