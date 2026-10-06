"""Shared conversion caches with compatibility manifests and reader leases."""

import ast
import atexit
import difflib
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shlex
import shutil
import struct
import tempfile
from contextvars import ContextVar
from pathlib import Path

logger = logging.getLogger(__name__)
MANIFEST = "miles_checkpoint_manifest.json"
_LEASES = ContextVar("checkpoint_reader_leases", default=None)

# Distributed checkpoints re-shard on load; execution topology is not a weight identity.
_EXECUTION_OPTIONS = {
    "--tensor-model-parallel-size",
    "--pipeline-model-parallel-size",
    "--context-parallel-size",
    "--expert-model-parallel-size",
    "--expert-tensor-parallel-size",
    "--decoder-first-pipeline-num-layers",
    "--decoder-last-pipeline-num-layers",
    "--max-workers",
    "--device",
}
_CONVERTERS = {
    "convert_hf_to_torch_dist.py": ("--hf-checkpoint", "--save"),
    "fp8_cast_bf16.py": ("--input-fp8-hf-path", "--output-bf16-hf-path"),
    "convert_kimi_int4_to_bf16.py": ("--model-dir", "--output-dir"),
    **{
        name: ("--model-dir", "--save-dir")
        for name in (
            "convert_hf_to_fp8.py",
            "convert_hf_to_mxfp8.py",
            "convert_hf_to_nvfp4.py",
            "convert_hf_to_int4_direct.py",
            "convert_mxfp4_to_bf16.py",
            "convert_mxfp4_to_fp8.py",
        )
    },
}


def enabled():
    return os.environ.get("MILES_CI_CHECKPOINT_CACHE") == "1"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def snapshot(path, *, weights_only=False):
    result = {}
    for file in sorted(path.rglob("*")):
        if not file.is_file() or ".cache" in file.relative_to(path).parts or file.name == MANIFEST:
            continue
        if (
            weights_only
            and file.suffix not in {".safetensors", ".bin", ".pt", ".py"}
            and file.name
            not in {
                "config.json",
                "hf_quant_config.json",
                "quantize_config.json",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            }
        ):
            continue
        stat = file.stat()
        entry = {"size": stat.st_size}
        if file.suffix in {".json", ".py", ".txt"} or file.name == ".metadata":
            entry["sha256"] = _digest(file.read_bytes())
        else:
            entry["mtime_ns"] = stat.st_mtime_ns
            if file.suffix == ".safetensors":
                with file.open("rb") as reader:
                    length = struct.unpack("<Q", reader.read(8))[0]
                    if length > 100_000_000:
                        raise ValueError(f"Invalid safetensors header: {file}")
                    header = reader.read(length)
                    if len(header) != length:
                        raise ValueError(f"Truncated safetensors header: {file}")
                    entry["header_sha256"] = _digest(header)
        result[str(file.relative_to(path))] = entry
    if not result:
        raise ValueError(f"Empty checkpoint: {path}")
    return result


def _options(argv):
    options = {}
    flag = None
    for token in argv:
        if token.startswith("--"):
            flag, sep, value = token.partition("=")
            options[flag] = [value] if sep else []
        else:
            if flag is None:
                raise ValueError(f"Expected conversion option, got {token!r}")
            options[flag].append(token)
    return options


def _source_hashes(paths, root):
    return {
        str(path.relative_to(root)): _digest(ast.dump(ast.parse(path.read_text()), include_attributes=False).encode())
        for path in sorted(set(paths))
    }


def _model_dependencies(paths, root):
    pending = list(paths)
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("miles_plugins.models."):
                target = root.joinpath(*node.module.split("."))
                if target.with_suffix(".py").is_file():
                    pending.append(target.with_suffix(".py"))
                elif target.is_dir():
                    pending.extend(target.rglob("*.py"))
    return found


def _local_converter_dependencies(paths, root):
    pending = list(paths)
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.replace(".", "/")
                for candidate in (root / f"{module}.py", root / "tools" / f"{module}.py"):
                    if candidate.is_file():
                        pending.append(candidate)
    return found


def _package_versions():
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")}
    # Different converter families use different optional GPU libraries.
    for name in ("transformer-engine", "triton", "compressed-tensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _conversion_code(tool, options, source, environment):
    root = Path(__file__).resolve().parents[4]
    paths = [tool]
    if tool.name != "convert_hf_to_torch_dist.py":
        paths = _local_converter_dependencies(paths, root)
        sources = {"miles": _source_hashes(paths, root)}
        if tool.name == "convert_hf_to_mxfp8.py":
            spec = importlib.util.find_spec("sglang")
            assert spec is not None, "SGLang is required for MXFP8 conversion"
            sglang = Path(next(iter(spec.submodule_search_locations)))
            sources["sglang"] = _source_hashes((sglang / "srt/layers/quantization").rglob("*.py"), sglang)
        return sources

    paths.extend(
        root / "miles/backends/megatron_utils" / name
        for name in ("arguments.py", "model_provider.py", "initialize.py", "fp32_param_utils.py")
    )
    config = json.loads((source / "config.json").read_text())
    model_types = {config["model_type"]}
    if "text_config" in config:
        model_types.add(config["text_config"]["model_type"])
    for file in (root / "miles_plugins/mbridge").glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_model":
                registered = ast.literal_eval(node.args[0])
                registered = {registered} if isinstance(registered, str) else set(registered)
                if model_types & registered:
                    paths.append(file)
                    model_types.add(file.stem)
                    break
    for flag in ("--spec", "--custom-model-provider-path"):
        for value in options.get(flag, []):
            if value.startswith("miles_plugins.models."):
                model_types.add(value.split(".")[2])
    for name in model_types:
        directory = root / "miles_plugins/models" / name
        if directory.is_dir():
            paths.extend(directory.rglob("*.py"))
        elif directory.with_suffix(".py").is_file():
            paths.append(directory.with_suffix(".py"))
    paths = _model_dependencies(paths, root)
    candidates = [Path(p) for p in environment.get("PYTHONPATH", "").split(os.pathsep) if p]
    candidates.append(Path(os.environ["MEGATRON_SOURCE_ROOT"]))
    megatron = next(path for path in candidates if (path / "megatron/training").is_dir())
    megatron_paths = []
    for directory in ("core/models", "core/transformer", "core/tensor_parallel", "core/dist_checkpointing"):
        megatron_paths.extend((megatron / "megatron" / directory).rglob("*.py"))
    megatron_paths.extend(
        megatron / "megatron/training" / name for name in ("arguments.py", "checkpointing.py", "training.py")
    )
    if not megatron_paths:
        raise ValueError(f"No Megatron conversion sources under {megatron}")
    bridge = importlib.util.find_spec("mbridge")
    if bridge is None:
        raise ValueError("mbridge is required to identify the checkpoint conversion recipe")
    bridge_root = Path(next(iter(bridge.submodule_search_locations)))
    return {
        "miles": _source_hashes(paths, root),
        "megatron": _source_hashes(megatron_paths, megatron),
        "mbridge": _source_hashes(bridge_root.rglob("*.py"), bridge_root),
    }


def _mismatch(path, recipe):
    manifest = path / MANIFEST
    if not path.exists() or not any(path.iterdir()):
        return "missing"
    if not manifest.is_file():
        return "existing checkpoint has no compatibility manifest"
    try:
        stored = json.loads(manifest.read_text())
        expected = {"recipe": recipe, "files": snapshot(path)}
    except (ValueError, struct.error) as error:
        return f"invalid checkpoint metadata: {error}"
    if stored == expected:
        return None
    diff = difflib.unified_diff(
        _json(stored).splitlines(), _json(expected).splitlines(), fromfile="cached", tofile="requested", lineterm=""
    )
    return "\n".join(diff)


def _leases():
    leases = _LEASES.get()
    if leases is None:
        leases = {}
        _LEASES.set(leases)
        atexit.register(_close_leases)
    return leases


def _close_leases():
    leases = _LEASES.get()
    if leases is not None:
        for lock in leases.values():
            lock.close()
        leases.clear()


def cached_checkpoint(path, recipe, build, *, adopt=None):
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError(f"Checkpoint cache must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    leases = _leases()
    if path in leases:
        mismatch = _mismatch(path, recipe)
        if mismatch is not None:
            raise RuntimeError(f"Checkpoint already in use by this test: {path}\n{mismatch}")
        return
    lock = path.with_name(path.name + ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_SH)
        mismatch = _mismatch(path, recipe)
        if mismatch is not None:
            allow = os.environ.get("MILES_CI_OVERWRITE_CHECKPOINT") == "1"
            if adopt is None:
                _require_overwrite(path, mismatch, allow)
            fcntl.flock(lock, fcntl.LOCK_UN)
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Another converter may have published while we waited for readers.
            mismatch = _mismatch(path, recipe)
            if mismatch is not None:
                if adopt is not None and not (path / MANIFEST).exists() and adopt(path):
                    (path / MANIFEST).write_text(_json({"recipe": recipe, "files": snapshot(path)}))
                else:
                    _require_overwrite(path, mismatch, allow)
                    _publish(path, recipe, build)
            fcntl.flock(lock, fcntl.LOCK_SH)
        logger.info("checkpoint cache verified: %s", path)
        leases[path] = lock
    except BaseException:
        lock.close()
        raise


def _require_overwrite(path, mismatch, allow):
    if mismatch != "missing" and not allow:
        raise RuntimeError(
            f"Incompatible checkpoint cache: {path}\n{mismatch}\n"
            "Refusing to overwrite. Add the PR label 'overwrite-checkpoint' to authorize rebuilding."
        )


def _check_complete(path, converter):
    if converter == "huggingface":
        assert (path / "config.json").is_file(), f"Missing HF model config: {path}"
    elif converter == "convert_hf_to_torch_dist.py":
        assert (path / "latest_checkpointed_iteration.txt").read_text().strip() == "release"
        assert (path / "release/.metadata").is_file(), f"Missing distributed checkpoint metadata: {path}"
    else:
        index = json.loads((path / "model.safetensors.index.json").read_text())
        assert index["weight_map"], f"Empty converted checkpoint: {path}"
        for name in set(index["weight_map"].values()):
            assert (path / name).is_file(), f"Missing converted shard: {path / name}"


def _publish(path, recipe, build):
    with tempfile.TemporaryDirectory(prefix=f".{path.name}.building-", dir=path.parent) as temp:
        staging = Path(temp) / "checkpoint"
        build(staging)
        _check_complete(staging, recipe["converter"])
        manifest = {"recipe": recipe, "files": snapshot(staging)}
        (staging / MANIFEST).write_text(_json(manifest))
        # Readers hold this lock through the test and cannot observe the replacement gap.
        if path.exists():
            shutil.rmtree(path)
        staging.rename(path)
    logger.info("checkpoint cache published: %s", path)


def _weight_options(converter, options, source_flag, destination_flag):
    weight_options = {
        flag: values
        for flag, values in options.items()
        if flag not in _EXECUTION_OPTIONS | {source_flag, destination_flag, "--overwrite"}
    }
    if converter == "convert_hf_to_torch_dist.py":
        # Without an explicit padded vocabulary, TP can change the global embedding shape.
        if "--padded-vocab-size" not in options:
            weight_options["--tensor-model-parallel-size"] = options.get("--tensor-model-parallel-size", ["1"])
        # set_default_megatron_args uses bf16 unless fp16 is requested.
        weight_options.pop("--bf16", None)
        weight_options.setdefault("--megatron-to-hf-mode", ["raw"])
    return weight_options


def run_conversion(cmd, execute):
    if not enabled():
        return execute(cmd)
    argv = shlex.split(cmd)
    if argv[:2] == ["hf", "download"]:
        if "--repo-type" in argv and argv[argv.index("--repo-type") + 1] == "dataset":
            return execute(cmd)
        options = _options(argv[3:])
        if options.get("--repo-type", ["model"])[0] == "model":
            unknown = options.keys() - {"--local-dir", "--repo-type", "--revision"}
            if unknown:
                raise ValueError(f"Unsupported cached HF download options: {sorted(unknown)}")
            return download_hf_checkpoint(
                argv[2], local_dir=options["--local-dir"][0], revision=options.get("--revision", [None])[0]
            )
    converters = [(i, Path(token)) for i, token in enumerate(argv) if Path(token).name in _CONVERTERS]
    if not converters:
        return execute(cmd)
    if len(converters) != 1 or any(token in {";", "&&", "||", "|"} for token in argv):
        raise ValueError(f"Checkpoint conversion must be a standalone command: {cmd}")
    index, tool = converters[0]
    tool = tool.resolve(strict=True)
    source_flag, destination_flag = _CONVERTERS[tool.name]
    options = _options(argv[index + 1 :])
    source = Path(options[source_flag][0]).resolve(strict=True)
    destination = Path(options[destination_flag][0]).absolute()
    resolved_destination = destination.resolve()
    if (
        source == resolved_destination
        or source in resolved_destination.parents
        or resolved_destination in source.parents
    ):
        raise ValueError("Checkpoint source and destination must not overlap")
    if any(flag in options for flag in ("--files", "--shard-rank", "--num-shards", "--finalize-only")):
        raise ValueError("CI checkpoint caches require a complete conversion, not a partial shard")
    environment = dict(os.environ)
    environment.update(token.split("=", 1) for token in argv[:index] if "=" in token and not token.startswith("--"))
    recipe = {
        "format": 1,
        "converter": tool.name,
        "source": snapshot(source, weights_only=True),
        "options": _weight_options(tool.name, options, source_flag, destination_flag),
        "code": _conversion_code(tool, options, source, environment),
        "packages": _package_versions(),
        "environment": {
            key: value
            for key, value in environment.items()
            if key.startswith(("NVTE_", "FLASHINFER_")) or key == "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH"
        },
    }

    def build(staging):
        replacement = []
        for flag, values in options.items():
            replacement.extend([flag, str(staging)] if flag == destination_flag else [flag, *values])
        prefix = []
        for token in argv[: index + 1]:
            if "=" in token and not token.startswith("--"):
                key, value = token.split("=", 1)
                prefix.append(key + "=" + shlex.quote(value))
            else:
                prefix.append(shlex.quote(token))
        execute(" ".join(prefix) + " " + shlex.join(replacement))
        if snapshot(source, weights_only=True) != recipe["source"]:
            raise RuntimeError(f"Source checkpoint changed during conversion: {source}")

    return cached_checkpoint(destination, recipe, build)


def download_hf_checkpoint(repo_id, *, local_dir, revision=None):
    # HF is optional for local launcher recording; load it only when downloading.
    from huggingface_hub import HfApi, snapshot_download

    if not enabled():
        return snapshot_download(repo_id, local_dir=local_dir, revision=revision)
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    identities = {file.rfilename: file.lfs.sha256 if file.lfs is not None else file.blob_id for file in info.siblings}
    recipe = {"format": 1, "converter": "huggingface", "repo_id": repo_id, "files": identities}

    def build(staging):
        _seed_hf_download(Path(local_dir), staging, identities)
        snapshot_download(repo_id, revision=info.sha, local_dir=str(staging))

    def adopt(path):
        # HF's ETag records verify legacy download provenance without re-reading all weight bytes.
        if not path.is_dir() or not any(path.iterdir()):
            return False
        try:
            inventory = snapshot(path)
        except (ValueError, struct.error):
            return False
        if set(inventory) != set(identities):
            return False
        for name, etag in identities.items():
            file = path / name
            metadata = path / ".cache/huggingface/download" / (name + ".metadata")
            if not file.is_file() or not metadata.is_file():
                return False
            lines = metadata.read_text().splitlines()
            if len(lines) != 3 or lines[1] != etag or file.stat().st_mtime > float(lines[2]):
                return False
        return path.is_dir()

    cached_checkpoint(Path(local_dir), recipe, build, adopt=adopt)
    return str(local_dir)


def _seed_hf_download(source, staging, identities):
    # Hub downloads replace files via rename; unchanged blobs can keep their existing inodes.
    for name, etag in identities.items():
        file = source / name
        metadata_name = Path(".cache/huggingface/download") / (name + ".metadata")
        metadata = source / metadata_name
        if not file.is_file() or not metadata.is_file():
            continue
        lines = metadata.read_text().splitlines()
        if len(lines) != 3 or lines[1] != etag or file.stat().st_mtime > float(lines[2]):
            continue
        (staging / name).parent.mkdir(parents=True, exist_ok=True)
        os.link(file, staging / name)
        (staging / metadata_name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata, staging / metadata_name)
