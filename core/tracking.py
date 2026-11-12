"""Khoi tao W&B dung chung; code goi tu tinh metric, dat seed va dong run."""

from collections.abc import Mapping
import json
import math
import os

import wandb

DEFAULT_ENTITY = "gr_ml_29_37_37"
DEFAULT_PROJECT = "ml-distribution-shift"
MAX_CONFIG_BYTES = 65536
_SECRET_KEYS = {
    "apikey", "wandbapikey", "token", "accesstoken", "refreshtoken",
    "clientsecret", "password", "secret", "authorization", "credentials",
    "credential", "privatekey", "secretkey", "awsaccesskeyid", "awssecretaccesskey",
}
_UNSUPPORTED_CONTROLS = {
    "id", "resume", "resume_from", "fork_from", "reinit", "dir", "settings",
    "base_url", "config", "wandb", "anonymous", "force", "save_code", "notes",
    "config_exclude_keys", "config_include_keys", "allow_val_change",
    "tensorboard", "sync_tensorboard", "monitor_gym",
}


def _metadata_copy(config):
    """Sao chep metadata JSON-safe, chan key nhay cam va config qua lon."""
    ancestors = set()
    nodes = 0

    def visit(value, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 10 or nodes > 10000:
            raise ValueError("config metadata exceeds the depth/node limit")
        if isinstance(value, (Mapping, list, tuple)):
            identity = id(value)
            if identity in ancestors:
                raise ValueError("config metadata must not contain cycles")
            ancestors.add(identity)
            try:
                if isinstance(value, Mapping):
                    result = {}
                    for key, child in value.items():
                        if not isinstance(key, str):
                            raise TypeError("config metadata keys must be strings")
                        normalized = "".join(c for c in key.casefold() if c.isalnum())
                        if normalized in _SECRET_KEYS:
                            raise ValueError("Credentials belong in SDK login or Secrets, never config")
                        result[key] = visit(child, depth + 1)
                    return result
                return [visit(child, depth + 1) for child in value]
            finally:
                ancestors.remove(identity)
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        raise TypeError("config metadata must contain only finite JSON-safe values")

    result = visit(config)
    try:
        encoded = json.dumps(result, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, UnicodeError):
        raise ValueError("config metadata must be valid UTF-8 JSON") from None
    if len(encoded) > MAX_CONFIG_BYTES:
        raise ValueError("config metadata must not exceed 64 KiB")
    return result


def init_run(config):
    """Tao mot global run tu config phang va tra ve W&B Run.

    Config uu tien hon environment, sau do moi den gia tri mac dinh.
    Entity/project mac dinh: gr_ml_29_37_37/ml-distribution-shift; mode online.
    Ten run: [experiment-]method-seedN; name duoc truyen se ghi de ten tu sinh.
    group/job_type khong bat buoc. Lab1 la mac dinh, lab2 can chon ro rang.
    Tag lab mau thuan bi tu choi. Config goc va RNG khong bi thay doi.

    Metadata can JSON-safe, sau toi da 10 muc va khong qua 64 KiB.
    Khong truyen API key, model, tensor hoac SDK options nhu id/resume/reinit.
    Dang nhap va endpoint do SDK quan ly ben ngoai ham nay.

    wandb.log va run.log cung ghi vao run duoc tra ve.
    Can goi run.finish() truoc khi init lai; khong tu retry, reuse hay dong run cu.
    Input sai gay TypeError/ValueError, run dang active gay RuntimeError.
    Loi SDK duoc truyen lai, khong tu chuyen sang offline."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping (for example, a dict)")
    if wandb.run is not None:
        raise RuntimeError("A W&B run is already active; call run.finish() before init_run()")
    settings = _metadata_copy(config)
    if _UNSUPPORTED_CONTROLS.intersection(settings):
        raise ValueError("Unsupported W&B control option; use the documented flat control keys")

    def text_setting(key, default=None, *, environment=True):
        fallback = os.environ.get("WANDB_" + key.upper(), default) if environment else default
        value = settings.get(key, fallback)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a non-empty string")
        return value

    project = text_setting("project", DEFAULT_PROJECT)
    entity = text_setting("entity", DEFAULT_ENTITY)
    for key, value in (("project", project), ("entity", entity)):
        if (value is None or value != value.strip() or len(value) > 128
                or any(char in value for char in "/\\#?%:") or any(c.isspace() for c in value)):
            raise ValueError(f"{key} must be a non-empty identifier without whitespace or reserved characters")
    mode = text_setting("mode", "online")
    if mode not in ("online", "offline", "disabled"):
        raise ValueError("mode must be online, offline, or disabled")
    seed = settings.get("seed")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    method = text_setting("method", "experiment", environment=False)
    experiment = text_setting("experiment", environment=False)
    if method is None:
        raise ValueError("method must be a non-empty string")
    lab = settings.get("lab", "lab1")
    if lab not in ("lab1", "lab2"):
        raise ValueError("lab must be lab1 or lab2")
    tags = settings.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ValueError("tags must be a list or tuple of non-empty strings")
    if any(tag.strip() in {"lab1", "lab2"} and tag != lab for tag in tags):
        raise ValueError("lab tags conflict with lab; select lab explicitly")
    tags = list(dict.fromkeys([lab, *tags]))
    prefix = f"{experiment}-" if experiment is not None else ""
    name = text_setting("name") or f"{prefix}{method}-seed{seed if seed is not None else 'unset'}"
    group = text_setting("group")
    job_type = text_setting("job_type")
    settings.update(project=project, entity=entity, mode=mode, name=name, tags=tags,
                    lab=lab, method=method, group=group, job_type=job_type)
    settings = _metadata_copy(settings)  # Kiem tra ca payload cuoi vi environment co the lam config lon hon.
    return wandb.init(project=project, entity=entity, name=name, mode=mode, tags=tags,
                      config=settings, group=group, job_type=job_type,
                      reinit="return_previous", resume="never" if mode == "online" else None)
