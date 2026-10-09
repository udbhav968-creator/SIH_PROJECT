"""
Language-model providers behind one call: complete(system, user, images=None) -> {"text", "provider", "model"}.

Chosen automatically (ROAD_SHIELD_LLM=anthropic|ollama|none forces one):
    anthropic   ANTHROPIC_API_KEY is set. Model ROAD_SHIELD_LLM_MODEL (default claude-sonnet-5-5).
                Messages API over HTTPS with the standard library: no SDK needed.
    ollama      an Ollama server answers at OLLAMA_HOST (default http://127.0.0.1:11434), free and local.
                Model ROAD_SHIELD_OLLAMA_MODEL (default llama3.2:3b; a vision model such as llava for photos).
    none        no model: callers fall back to extractive answers and templates, and say so.

The API key is read from the environment only, never from a request, a file in the repository or a log.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CLAUDE = "claude-sonnet-5-5"
DEFAULT_OLLAMA = "llama3.2:3b"
TIMEOUT_S = 60


class LLMError(RuntimeError):
    pass


def _post(url, payload, headers, timeout=TIMEOUT_S):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"HTTP {e.code} from {url.split('/')[2]}: {body}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"cannot reach {url.split('/')[2]}: {e}")


def _sniff_media_type(b64):
    head = base64.b64decode(b64[:32] + "=" * (-len(b64[:32]) % 4))
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:3] == b"GIF":
        return "image/gif"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return None


def _as_supported_image(b64):
    """(media type, base64) the API accepts: JPEG, PNG, GIF and WebP pass through; anything else is re-encoded as JPEG."""
    mt = _sniff_media_type(b64)
    if mt:
        return mt, b64
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "image/jpeg", base64.b64encode(buf.getvalue()).decode()


class Anthropic:
    name = "anthropic"

    def __init__(self, api_key=None, model=None, url=None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ROAD_SHIELD_LLM_MODEL") or DEFAULT_CLAUDE
        self.url = url or os.environ.get("ROAD_SHIELD_ANTHROPIC_URL") or ANTHROPIC_URL
        self.vision = True

    @property
    def available(self):
        return bool(self.api_key)

    def complete(self, system, user, images=None, max_tokens=900, temperature=0.2):
        content = []
        for b in images or []:
            mt, data = _as_supported_image(b)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        content.append({"type": "text", "text": user})
        t0 = time.time()
        r = _post(self.url, {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                             "system": system, "messages": [{"role": "user", "content": content}]},
                  {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION})
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        if not text and r.get("stop_reason") == "refusal":
            raise LLMError("the model declined to answer")
        return {"text": text, "provider": self.name, "model": r.get("model", self.model),
                "ms": round((time.time() - t0) * 1000), "usage": r.get("usage")}


class Ollama:
    name = "ollama"

    def __init__(self, host=None, model=None):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.model = model or os.environ.get("ROAD_SHIELD_OLLAMA_MODEL") or DEFAULT_OLLAMA
        self._ok = None
        self.vision = any(k in self.model for k in ("llava", "vision", "moondream", "bakllava", "qwen2.5vl", "gemma3"))

    @property
    def available(self):
        if self._ok is None or (time.time() - self._ok[1]) > 60:
            try:
                with urllib.request.urlopen(self.host + "/api/tags", timeout=0.6) as r:
                    names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
                # 'llama3.2:3b' must be pulled as exactly that; a bare name means ':latest'
                want = self.model if ":" in self.model else self.model + ":latest"
                ok = want in names
            except Exception:
                ok = False
            self._ok = (ok, time.time())
        return self._ok[0]

    def complete(self, system, user, images=None, max_tokens=900, temperature=0.2):
        msg = {"role": "user", "content": user}
        if images:
            msg["images"] = list(images)
        t0 = time.time()
        r = _post(self.host + "/api/chat", {"model": self.model, "stream": False,
                                            "options": {"temperature": temperature, "num_predict": max_tokens},
                                            "messages": [{"role": "system", "content": system}, msg]},
                  {}, timeout=180)
        return {"text": (r.get("message") or {}).get("content", ""), "provider": self.name, "model": self.model,
                "ms": round((time.time() - t0) * 1000)}


class NoModel:
    name = "none"
    model = None
    vision = False
    available = True

    def complete(self, *a, **kw):
        raise LLMError("no language model is configured (set ANTHROPIC_API_KEY, or run Ollama)")


_CACHE = {}


def _provider(cls):
    """One instance per provider and configuration, so Ollama's availability check is cached (60 s)."""
    key = (cls.__name__, os.environ.get("ANTHROPIC_API_KEY", "")[-6:], os.environ.get("ROAD_SHIELD_LLM_MODEL"),
           os.environ.get("ROAD_SHIELD_ANTHROPIC_URL"), os.environ.get("OLLAMA_HOST"),
           os.environ.get("ROAD_SHIELD_OLLAMA_MODEL"))
    if key not in _CACHE:
        if len(_CACHE) > 16:
            _CACHE.clear()
        _CACHE[key] = cls()
    return _CACHE[key]


def pick(require_vision=False):
    forced = os.environ.get("ROAD_SHIELD_LLM", "").lower()
    order = {"anthropic": [Anthropic], "ollama": [Ollama], "none": []}.get(forced, [Anthropic, Ollama])
    for cls in order:
        p = _provider(cls)
        if p.available and (p.vision or not require_vision):
            return p
    return NoModel()


def describe():
    p = pick()
    return {"provider": p.name, "model": p.model,
            "vision": bool(getattr(p, "vision", False)) and p.name != "none",
            "how_to_enable": None if p.name != "none" else
            "set ANTHROPIC_API_KEY (Claude) or install Ollama and `ollama pull llama3.2:3b`, then restart the engine"}
