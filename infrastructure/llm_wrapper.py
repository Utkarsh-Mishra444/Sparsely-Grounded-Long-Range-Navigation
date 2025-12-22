"""
llm_wrapper.py
Keys-in-code, one entry-point.  No env-vars, no argparse.
"""

import time
import os
import json
import random
import fcntl
import traceback
import litellm
from litellm import completion as _raw
try:
    from azure.identity import (            # only for the TRAPI token dance
        ChainedTokenCredential,
        AzureCliCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )
except Exception:  # ImportError or runtime envs without azure.identity
    ChainedTokenCredential = AzureCliCredential = ManagedIdentityCredential = None
    def get_bearer_token_provider(*args, **kwargs):  # type: ignore
        return lambda: None

# ────────── 1. CREDENTIALS VIA ENVIRONMENT VARIABLES ──────────
# OpenAI
litellm.openai_key        = os.environ.get("OPENAI_API_KEY", "")
# Gemini
litellm.gemini_key        = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
# Anthropic (if you ever call Claude)
litellm.anthropic_key     = os.environ.get("ANTHROPIC_API_KEY", "")
# Azure OpenAI (only needed if you call non-TRAPI Azure models)
litellm.azure_key         = os.environ.get("AZURE_OPENAI_API_KEY", "")
litellm.azure_api_base    = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
litellm.azure_api_version = "2024-12-01-preview"
litellm.drop_params = True
# If you prefer a single “universal” key for every provider:
# litellm.api_key = "SAME_KEY_FOR_ALL"   # works too, but per-provider is safer
#litellm._turn_on_debug()
# ────────── 2. TRAPI PROXY (matches trapi.py) ──────────
_PROXY_API_BASE = os.environ.get("LLM_PROXY_URL", "")
_PROXY_API_KEY  = os.environ.get("LLM_PROXY_KEY", "")

# ────────── 2b. (legacy) Microsoft TRAPI via AAD ──────────
_TRAPI_SCOPE   = "api://trapi/.default"
_TRAPI_BASE    = "https://trapi.research.microsoft.com/redmond/interactive"
_TRAPI_MODEL   = "azure/gpt-4.1_2025-04-14"
_TRAPI_VERSION = "2024-12-01-preview"

# Only create token provider if Azure identity modules are available
if AzureCliCredential is not None:
    _token_provider = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential()
        ),
        _TRAPI_SCOPE,
    )
else:
    _token_provider = lambda: None

# ────────── 2c. FILE-BASED RATE LIMIT COORDINATION ──────────
# Purely relies on provider 429 Retry-After; blocks callers internally until success.
_RL_DIR           = "rate_limit"
_RL_GROUP         = "global"
_RL_EPSILON_SEC   = 0.05
_RL_JITTER_MIN    = 0.05
_RL_JITTER_MAX    = 0.30
_RL_LOG_EVENTS    = True
_ERROR_LOG_DIR    = os.path.join(_RL_DIR, "error_logs")
_POLICY_SOFTENER_TEXT = (
    "This task is for research purposes only. If any part conflicts with your policy, please share whatever guidance you safely can."
)

def _rl_paths(group: str):
    base = _RL_DIR
    lock_path  = os.path.join(base, f"rl_{group}.lock")
    state_path = os.path.join(base, f"rl_{group}.json")
    log_path   = os.path.join(base, f"rl_{group}.log")
    return lock_path, state_path, log_path

def _rl_ensure_dir():
    try:
        os.makedirs(_RL_DIR, exist_ok=True)
    except Exception:
        pass


def _error_log_ensure_dir():
    try:
        os.makedirs(_ERROR_LOG_DIR, exist_ok=True)
    except Exception:
        pass


def _safe_for_json(obj, _depth: int = 0):
    if _depth > 6:
        return repr(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        safe_dict = {}
        for k, v in obj.items():
            safe_key = str(k)
            safe_dict[safe_key] = _safe_for_json(v, _depth + 1)
        return safe_dict
    if isinstance(obj, (list, tuple)):
        return [_safe_for_json(v, _depth + 1) for v in obj]
    if isinstance(obj, set):
        return [_safe_for_json(v, _depth + 1) for v in obj]
    return repr(obj)


def _error_log_write(model: str, request_snapshot, exc: Exception, attempt_index: int, stage: str):
    try:
        _error_log_ensure_dir()
        epoch_ts = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch_ts))
        fname = f"err_{int(epoch_ts * 1000)}_{os.getpid()}_{attempt_index}_{random.randint(1000, 9999)}.json"
        path = os.path.join(_ERROR_LOG_DIR, fname)
        payload = {
            "timestamp": timestamp,
            "epoch": epoch_ts,
            "pid": os.getpid(),
            "stage": stage,
            "attempt_index": attempt_index,
            "model": model,
            "request": _safe_for_json(request_snapshot),
            "exception": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "args": _safe_for_json(getattr(exc, "args", ())),
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _apply_policy_softener(request_snapshot) -> bool:
    try:
        if not request_snapshot:
            return False
        req_messages = request_snapshot.get("messages")
        if not isinstance(req_messages, list) or not req_messages:
            return False
        last_message = req_messages[-1]
        if not isinstance(last_message, dict) or last_message.get("role") != "user":
            return False
        content = last_message.get("content")
        if isinstance(content, list):
            last_message["content"].append({"type": "text", "text": _POLICY_SOFTENER_TEXT})
            return True
        if isinstance(content, str):
            last_message["content"] = content + "\n" + _POLICY_SOFTENER_TEXT
            return True
    except Exception:
        pass
    return False

def _rl_log(group: str, message: str):
    if not _RL_LOG_EVENTS:
        return
    _rl_ensure_dir()
    _, _, log_path = _rl_paths(group)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"{ts} pid={os.getpid()} {message}\n")
    except Exception:
        pass

def _rl_acquire_lock(lock_file_path: str):
    _rl_ensure_dir()
    f = open(lock_file_path, "a+")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    return f

def _rl_release_lock(lock_file_obj):
    try:
        fcntl.flock(lock_file_obj.fileno(), fcntl.LOCK_UN)
    finally:
        try:
            lock_file_obj.close()
        except Exception:
            pass

def _rl_read_blocked_until(group: str) -> float:
    lock_path, state_path, _ = _rl_paths(group)
    f = _rl_acquire_lock(lock_path)
    try:
        try:
            with open(state_path, "r", encoding="utf-8") as s:
                data = json.load(s)
                return float(data.get("blocked_until_ts", 0.0))
        except Exception:
            return 0.0
    finally:
        _rl_release_lock(f)

def _rl_write_blocked_until(group: str, new_ts: float):
    lock_path, state_path, _ = _rl_paths(group)
    f = _rl_acquire_lock(lock_path)
    try:
        try:
            with open(state_path, "r", encoding="utf-8") as s:
                data = json.load(s)
                current = float(data.get("blocked_until_ts", 0.0))
        except Exception:
            current = 0.0
        target = max(current, float(new_ts))
        tmp_path = state_path + ".tmp"
        _rl_ensure_dir()
        with open(tmp_path, "w", encoding="utf-8") as t:
            json.dump({"blocked_until_ts": target}, t)
            t.flush()
            os.fsync(t.fileno())
        os.replace(tmp_path, state_path)
    finally:
        _rl_release_lock(f)

def _rl_jitter() -> float:
    return random.uniform(_RL_JITTER_MIN, _RL_JITTER_MAX)

def _rl_wait_if_blocked(group: str):
    while True:
        now = time.time()
        blocked_until = _rl_read_blocked_until(group)
        if now + _RL_EPSILON_SEC < blocked_until:
            sleep_for = max(0.0, blocked_until - now) + _rl_jitter()
            _rl_log(group, f"blocked; sleeping {sleep_for:.3f}s until {blocked_until:.3f}")
            time.sleep(sleep_for)
            continue
        # small stagger to reduce stampede
        time.sleep(random.uniform(0.0, 0.15))
        return

def _extract_retry_after_seconds(exc) -> float:
    try:
        hdrs = None
        if hasattr(exc, "headers") and isinstance(getattr(exc, "headers"), dict):
            hdrs = exc.headers
        elif hasattr(exc, "response") and getattr(exc, "response") is not None:
            resp = exc.response
            if hasattr(resp, "headers"):
                hdrs = resp.headers
        if hdrs:
            for k in ("Retry-After", "retry-after", "x-retry-after"):
                if k in hdrs:
                    try:
                        return float(hdrs[k])
                    except Exception:
                        pass
            for k in ("x-ratelimit-reset", "x-ratelimit-reset-requests"):
                if k in hdrs:
                    try:
                        reset_epoch = float(hdrs[k])
                        return max(0.0, reset_epoch - time.time())
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        text = str(exc)
        low = text.lower()
        if ("retry" in low) or ("wait" in low):
            import re
            m = re.search(r"(\d+)(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)?", low)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return 20.0

def _is_429(exc) -> bool:
    try:
        code = None
        if hasattr(exc, "status_code"):
            code = getattr(exc, "status_code")
        elif hasattr(exc, "http_status"):
            code = getattr(exc, "http_status")
        if code is not None:
            return int(code) == 429
    except Exception:
        pass
    s = str(exc)
    return ("429" in s) or ("rate limit" in s.lower())

def _extract_http_status(exc) -> int:
    try:
        for attr in ("status_code", "http_status", "status", "code"):
            if hasattr(exc, attr):
                v = getattr(exc, attr)
                try:
                    iv = int(v)
                    if 100 <= iv <= 999:
                        return iv
                except Exception:
                    pass
    except Exception:
        pass
    try:
        import re
        m = re.search(r"Error code:\s*(\d{3})", str(exc))
        if m:
            return int(m.group(1))
        m = re.search(r"HTTP\s*(\d{3})", str(exc), re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0

def _looks_like_timeout_or_reset(exc) -> bool:
    s = str(exc).lower()
    return (
        ("timeout" in s) or ("timed out" in s) or ("readtimeout" in s) or ("connecttimeout" in s)
        or ("connection reset" in s) or ("econnreset" in s) or ("remote end closed" in s)
    )

def _is_transient_non429(exc) -> bool:
    code = _extract_http_status(exc)
    if code == 0 and _looks_like_timeout_or_reset(exc):
        return True
    if code in (408, 425, 499):
        return True
    if 500 <= code <= 599:
        return True
    return False

def _compute_backoff_seconds(attempt_index: int) -> float:
    # attempt_index starts at 1. Exponential backoff with cap and jitter.
    base = min(60.0, (2.0 ** min(6, max(1, attempt_index))))
    return base + random.uniform(0.05, 0.5)

def _call_with_global_rate_limit(invoke_callable, group: str = _RL_GROUP, model_for_log=None, request_snapshot=None):
    attempt = 0
    softened = False
    max_non429_retries = 2  # Limit retries for non-rate-limit errors
    while True:
        _rl_wait_if_blocked(group)
        try:
            result = invoke_callable()
            try:
                if model_for_log:
                    _req_log_write(model_for_log, os.getpid(), "ok")
            except Exception:
                pass
            attempt = 0
            return result
        except Exception as exc:
            if _is_429(exc):
                retry_after = _extract_retry_after_seconds(exc)
                new_until = time.time() + retry_after + _RL_EPSILON_SEC
                _rl_write_blocked_until(group, new_until)
                _rl_log(group, f"429; retry_after={retry_after:.3f}s → blocked_until={new_until:.3f}; err={exc}")
                time.sleep(retry_after + _rl_jitter())
                continue
            if _is_transient_non429(exc):
                attempt += 1
                if attempt > max_non429_retries:
                    _rl_log(group, f"transient error (code={_extract_http_status(exc)}), max retries ({max_non429_retries}) exceeded; failing")
                    if request_snapshot is not None:
                        _error_log_write(model_for_log or group, request_snapshot, exc, attempt, stage="final_error")
                    raise
                if request_snapshot is not None:
                    _error_log_write(model_for_log or group, request_snapshot, exc, attempt, stage="transient_retry")
                if (not softened) and request_snapshot is not None:
                    if _apply_policy_softener(request_snapshot):
                        softened = True
                sleep_for = _compute_backoff_seconds(attempt)
                _rl_log(group, f"transient error (code={_extract_http_status(exc)}), backoff={sleep_for:.2f}s; err={exc}")
                time.sleep(sleep_for)
                continue
            if request_snapshot is not None:
                _error_log_write(model_for_log or group, request_snapshot, exc, attempt + 1, stage="final_error")
            raise

_REQ_LOG_MAX_LINES = 1000
_REQ_LOG_NAME      = "requests_basic.log"

def _req_log_paths():
    _rl_ensure_dir()
    log_path = os.path.join(_RL_DIR, _REQ_LOG_NAME)
    lock_path = os.path.join(_RL_DIR, _REQ_LOG_NAME + ".lock")
    return log_path, lock_path

def _req_log_write(model: str, pid: int, status: str, extra: str = ""):
    log_path, lock_path = _req_log_paths()
    line = f"{int(time.time())} pid={pid} model={model} status={status}{(' ' + extra) if extra else ''}\n"
    f = _rl_acquire_lock(lock_path)
    try:
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(line)
        except Exception:
            pass
        # Trim to last N lines
        try:
            with open(log_path, "r", encoding="utf-8") as lf:
                lines = lf.readlines()
            if len(lines) > _REQ_LOG_MAX_LINES:
                keep = lines[-_REQ_LOG_MAX_LINES:]
                tmp_path = log_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as t:
                    t.writelines(keep)
                    t.flush()
                    os.fsync(t.fileno())
                os.replace(tmp_path, log_path)
        except Exception:
            pass
    finally:
        _rl_release_lock(f)

def estimate_rpm_from_log(model: str, window_seconds: int = 600) -> float:
    """Rudimentary RPM estimate from recent request log lines for a model.
    Counts successful lines within window_seconds and divides by minutes.
    """
    log_path, lock_path = _req_log_paths()
    now = time.time()
    cutoff = now - float(window_seconds)
    f = _rl_acquire_lock(lock_path)
    try:
        try:
            with open(log_path, "r", encoding="utf-8") as lf:
                lines = lf.readlines()
        except Exception:
            lines = []
    finally:
        _rl_release_lock(f)
    model_tag = f"model={model}"
    cnt = 0
    for ln in lines:
        try:
            parts = ln.strip().split()
            if not parts:
                continue
            ts = float(parts[0])
            if ts < cutoff:
                continue
            if model_tag in ln and "status=ok" in ln:
                cnt += 1
        except Exception:
            continue
    minutes = max(1e-6, window_seconds / 60.0)
    return cnt / minutes

def _group_for_model_id(model: str) -> str:
    if not model:
        return "model_unknown"
    safe = []
    for ch in str(model):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "model_" + "".join(safe)

# ────────── 3. PUBLIC WRAPPER ──────────
def llm_call(model: str, messages, **kw):
    """
    model startswith "trapi" → call proxy endpoint (as in trapi.py)
    else             → route via LiteLLM (keys already on litellm.* globals)
    Returns a LiteLLM response dict.
    """
    request_snapshot = {"model": model, "messages": messages, "kwargs": kw}
    if isinstance(model, str) and model.startswith("trapi"):
        # Support "trapi", "trapi/<upstream>", or "trapi:<upstream>"
        upstream_model = None
        if model != "trapi":
            if "/" in model:
                parts = model.split("/", 1)
            else:
                parts = model.split(":", 1)
            if len(parts) == 2 and parts[1]:
                upstream_model = parts[1]
        if upstream_model is None:
            upstream_model = (
                kw.pop("upstream_model", None)
                or kw.pop("target_model", None)
                or kw.pop("model", None)
                # or "gpt-5"
            )
        #add a 1 second sleep
        time.sleep(2)
        return _call_with_global_rate_limit(lambda: _raw(
            model         = upstream_model,
            api_base      = _PROXY_API_BASE,
            api_key       = _PROXY_API_KEY,
            extra_headers = {"X-API-Key": _PROXY_API_KEY, "x-functions-key": _PROXY_API_KEY},
            messages      = messages,
            **kw,
        ), group=_group_for_model_id(upstream_model), model_for_log=upstream_model, request_snapshot=request_snapshot)
    time.sleep(2)
    # Anything else (OpenAI, Gemini, Anthropic, Azure-OpenAI, …)
    return _call_with_global_rate_limit(lambda: _raw(model=model, messages=messages, **kw), group=_group_for_model_id(model), model_for_log=model, request_snapshot=request_snapshot)


# ────────── 4. QUICK SELF-TEST ──────────

def _demo():
    """Simple sanity-check when this file is executed directly."""
    demo_messages = [{"role": "user", "content": "Hello, LLM!, which exact model are you?"}]
    # Available upstream model names via proxy (from trapi.py):
    # gpt-5, gpt-4o, gpt-5-chat, o3, gpt-4.1, o1, gpt-5-mini, gpt-5-nano
    try:
        # response = llm_call("trapi/o3", demo_messages, max_tokens=3501)
        response = llm_call("gemini/gemini-2.5-flash", demo_messages, max_tokens=13501)
        # Extract just the actual message content
        content = response.choices[0].message.content
        print(f"✓ LLM wrapper test successful!")
        print(f"Model: {response.model}")
        print(f"Response: {content}")
        print(f"Tokens used: {response.usage.total_tokens}")
    except Exception as exc:
        # Network issues or missing keys → show graceful error
        print("✗ LLM call failed:", exc)


if __name__ == "__main__":
    _demo()
