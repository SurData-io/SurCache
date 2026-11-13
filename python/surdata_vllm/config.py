"""Startup checks for the direct vLLM path.

vLLM v0.33.0 (4ba787ec) cannot express per-block early free or a verified
partial-tail restore. Those settings fail closed.
"""

RECLAIM_REQUEST = "request_after_all_saves"
RECLAIM_PER_BLOCK = "per_block_after_safe_retirement"

VLLM_TAG = "v0.33.0"
VLLM_COMMIT = "4ba787ec9959e566d3b587a796dff07673324da5"
ROUTE = "B"


class ConfigError(ValueError):
    pass


def load_settings(raw: dict | None) -> dict:
    data = dict(raw or {})
    if data.get("partial_tail_restore"):
        raise ConfigError(
            "partial_tail_restore is refused: vLLM v0.33.0 only enables "
            "supports_partial_tail for an aligned Mamba group, and this "
            "connector has not verified tail restore"
        )
    reclaim = data.get("finished_block_reclaim", RECLAIM_REQUEST)
    if reclaim == RECLAIM_PER_BLOCK:
        raise ConfigError(
            "per-block reclaim is refused: request_finished delays the whole "
            "request until get_finished, so saved blocks are not freed ahead "
            "of the unsaved tail"
        )
    if reclaim != RECLAIM_REQUEST:
        raise ConfigError(
            "finished_block_reclaim must be request_after_all_saves; "
            "the whole request's GPU blocks stay until every D2H completes"
        )
    block_tokens = int(data.get("block_tokens", 16))
    bytes_per_token = int(data.get("bytes_per_token", 0))
    if block_tokens <= 0 or bytes_per_token <= 0:
        raise ConfigError("block_tokens and bytes_per_token must be positive")
    layout = str(data.get("layout_signature", "")).strip()
    if not layout:
        raise ConfigError("layout_signature is required")
    return {
        "route": ROUTE,
        "vllm_tag": VLLM_TAG,
        "vllm_commit": VLLM_COMMIT,
        "partial_tail_restore": False,
        "finished_block_reclaim": RECLAIM_REQUEST,
        "block_tokens": block_tokens,
        "bytes_per_token": bytes_per_token,
        "layout_signature": layout,
        "shard_id": int(data.get("shard_id", 0)),
        "model": str(data.get("model", "m")),
        "model_version": str(data.get("model_version", "v1")),
        "store_full_policy": normalize_store_full_policy(
            data.get("store_full_policy", "finish_drop")
        ),
    }


def normalize_store_full_policy(value) -> str:
    text = str(value or "finish_drop").strip().lower()
    if text not in ("finish_drop", "drop"):
        raise ConfigError("kv.store_full_policy must be finish_drop or drop")
    return text


def normalize_direct_save_mode(value) -> str:
    """``per_layer`` records one event per layer. ``end`` records one event after the forward."""
    text = str(value or "per_layer").strip().lower()
    if text not in ("per_layer", "end"):
        raise ConfigError("kv.direct_save_mode must be per_layer or end")
    return text


STORE_POOL_DEFAULT = 8 * 1024 * 1024 * 1024
PRELOAD_POOL_DEFAULT = 12 * 1024 * 1024 * 1024
EXTERNAL_KV_WORKER_DEFAULT = 128
EXTERNAL_KV_WORKER_MAX = 128


def _parse_byte_count(text: str):
    raw = text.strip().lower()
    if not raw:
        return None
    mult = 1
    if raw.endswith("g"):
        mult = 1024 ** 3
        raw = raw[:-1]
    elif raw.endswith("m"):
        mult = 1024 ** 2
        raw = raw[:-1]
    elif raw.endswith("k"):
        mult = 1024
        raw = raw[:-1]
    try:
        value = int(float(raw) * mult)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _cfg_text(lib, key: str) -> str:
    """Read one surfs_cfg string. Registry miss falls through to the JSON walk.

    ``surfs_cfg_get_string`` only returns a value when the key is registered
    and the JSON value is a string. The deployed SDK has no registry entry
    for the pool sizes, so a missing id uses ``surfs_cfg_debug_get_string``,
    which walks the same loaded document. A missing symbol keeps the caller
    on ``default``.
    """
    import ctypes

    lib.surfs_cfg_get_id_by_key.argtypes = [ctypes.c_char_p]
    lib.surfs_cfg_get_id_by_key.restype = ctypes.c_int
    lib.surfs_cfg_get_string.argtypes = [ctypes.c_int]
    lib.surfs_cfg_get_string.restype = ctypes.c_void_p
    cfg_id = int(lib.surfs_cfg_get_id_by_key(str(key).encode()))
    if cfg_id > 0:
        ptr = lib.surfs_cfg_get_string(cfg_id)
        if ptr:
            raw = ctypes.cast(ptr, ctypes.c_char_p).value
            text = raw.decode() if raw else ""
            if text:
                return text
    debug = getattr(lib, "surfs_cfg_debug_get_string", None)
    if debug is None:
        return ""
    try:
        debug.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        debug.restype = ctypes.c_char_p
        raw = debug(str(key).encode(), b"")
    except (AttributeError, OSError):
        return ""
    if not raw:
        return ""
    return raw.decode()


def load_pool_bytes(key: str, default: int, explicit=None) -> int:
    """Read a pool size through surfs_cfg. Missing key uses ``default``.

    An explicit argument is for tests. Values may be a byte count or a
    ``G`` / ``M`` / ``K`` suffix. This function does not consult the environment.
    Defaults stay 8GiB store and 12GiB preload. A bench config that writes
    the key as a JSON string overrides that default. The direct bench writes
    those same two sizes and does not raise them.
    """
    if explicit is not None:
        parsed = _parse_byte_count(str(explicit))
        if parsed is None:
            raise ConfigError("%s must be a positive byte count" % key)
        return parsed
    try:
        import ctypes
        import os

        lib = ctypes.CDLL(os.environ.get("SURDATA_SDK_LIB", "libsurdata_sdk.so"))
    except OSError:
        return int(default)
    parsed = _parse_byte_count(_cfg_text(lib, key))
    if parsed is None:
        return int(default)
    return parsed


def _clamp_external_kv_workers(value) -> int:
    """Missing or <= 0 becomes 128. Anything above 128 is capped at 128."""
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = 0
    if workers <= 0:
        workers = EXTERNAL_KV_WORKER_DEFAULT
    if workers > EXTERNAL_KV_WORKER_MAX:
        workers = EXTERNAL_KV_WORKER_MAX
    return workers


def load_external_kv_worker_count(explicit=None) -> int:
    """Read ``kv.external_kv_worker_count`` through surfs_cfg.

    The SDK defaults this key to 128 and caps it at 128. A missing library,
    a missing key, or a value <= 0 uses 128, then the same cap applies.
    An explicit argument is for tests. This function does not consult the
    environment.
    """
    if explicit is not None:
        return _clamp_external_kv_workers(explicit)
    try:
        import ctypes
        import os

        lib = ctypes.CDLL(os.environ.get("SURDATA_SDK_LIB", "libsurdata_sdk.so"))
    except OSError:
        return EXTERNAL_KV_WORKER_DEFAULT
    get_id = getattr(lib, "surfs_cfg_get_id_by_key", None)
    get_int = getattr(lib, "surfs_cfg_get_int", None)
    if not callable(get_id) or not callable(get_int):
        return EXTERNAL_KV_WORKER_DEFAULT
    get_id.argtypes = [ctypes.c_char_p]
    get_id.restype = ctypes.c_int
    get_int.argtypes = [ctypes.c_int]
    get_int.restype = ctypes.c_int
    cfg_id = int(get_id(b"kv.external_kv_worker_count"))
    if cfg_id <= 0:
        return EXTERNAL_KV_WORKER_DEFAULT
    return _clamp_external_kv_workers(get_int(cfg_id))


def _clamp_phase_diag_ms(value) -> int:
    """0 keeps the summary off. A negative or unreadable value is also off."""
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return 0
    if ms < 0:
        return 0
    return ms


def load_direct_phase_diag_ms(explicit=None) -> int:
    """Read ``kv.direct_phase_diag_ms`` through surfs_cfg. Missing key is 0.

    Zero disables the direct phase summary. A positive value is the interval
    in milliseconds. An explicit argument is for tests. This function does
    not consult the environment.
    """
    if explicit is not None:
        return _clamp_phase_diag_ms(explicit)
    try:
        import ctypes
        import os

        lib = ctypes.CDLL(os.environ.get("SURDATA_SDK_LIB", "libsurdata_sdk.so"))
    except OSError:
        return 0
    get_id = getattr(lib, "surfs_cfg_get_id_by_key", None)
    get_int = getattr(lib, "surfs_cfg_get_int", None)
    if not callable(get_id) or not callable(get_int):
        return 0
    get_id.argtypes = [ctypes.c_char_p]
    get_id.restype = ctypes.c_int
    get_int.argtypes = [ctypes.c_int]
    get_int.restype = ctypes.c_int
    cfg_id = int(get_id(b"kv.direct_phase_diag_ms"))
    if cfg_id <= 0:
        return 0
    return _clamp_phase_diag_ms(get_int(cfg_id))


def load_store_full_policy(explicit=None) -> str:
    """Read ``kv.store_full_policy`` through surfs_cfg. Missing key defaults to finish_drop.

    An explicit argument is for tests. This function does not consult the
    environment. A missing SDK library uses the same default.
    """
    if explicit is not None:
        return normalize_store_full_policy(explicit)
    try:
        import ctypes
        import os

        lib = ctypes.CDLL(os.environ.get("SURDATA_SDK_LIB", "libsurdata_sdk.so"))
    except OSError:
        return "finish_drop"
    lib.surfs_cfg_get_id_by_key.argtypes = [ctypes.c_char_p]
    lib.surfs_cfg_get_id_by_key.restype = ctypes.c_int
    lib.surfs_cfg_get_string.argtypes = [ctypes.c_int]
    lib.surfs_cfg_get_string.restype = ctypes.c_void_p
    cfg_id = int(lib.surfs_cfg_get_id_by_key(b"kv.store_full_policy"))
    if cfg_id <= 0:
        return "finish_drop"
    ptr = lib.surfs_cfg_get_string(cfg_id)
    if not ptr:
        return "finish_drop"
    raw = ctypes.cast(ptr, ctypes.c_char_p).value
    text = raw.decode() if raw else ""
    if text == "":
        return "finish_drop"
    return normalize_store_full_policy(text)


def load_direct_save_mode(explicit=None) -> str:
    """Read ``kv.direct_save_mode`` through surfs_cfg. Missing key defaults to per_layer.

    An explicit argument is for tests. This function does not consult the
    environment. A missing SDK library uses the same default. ``unmounted``
    is not a value of this key: that arm does not load the connector.
    """
    if explicit is not None:
        return normalize_direct_save_mode(explicit)
    try:
        import ctypes
        import os

        lib = ctypes.CDLL(os.environ.get("SURDATA_SDK_LIB", "libsurdata_sdk.so"))
    except OSError:
        return "per_layer"
    lib.surfs_cfg_get_id_by_key.argtypes = [ctypes.c_char_p]
    lib.surfs_cfg_get_id_by_key.restype = ctypes.c_int
    lib.surfs_cfg_get_string.argtypes = [ctypes.c_int]
    lib.surfs_cfg_get_string.restype = ctypes.c_void_p
    cfg_id = int(lib.surfs_cfg_get_id_by_key(b"kv.direct_save_mode"))
    if cfg_id <= 0:
        return "per_layer"
    ptr = lib.surfs_cfg_get_string(cfg_id)
    if not ptr:
        return "per_layer"
    raw = ctypes.cast(ptr, ctypes.c_char_p).value
    text = raw.decode() if raw else ""
    if text == "":
        return "per_layer"
    return normalize_direct_save_mode(text)
