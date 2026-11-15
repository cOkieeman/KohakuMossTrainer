"""Native parameter schemas, configuration resolution and script snapshots."""

import ast
import functools
import json
import re
import tokenize
from pathlib import Path

from kohakuengine import introspect_schema

from kohakumoss.cli.config.starter import literal
from kohakumoss.cli.config.validate import config_problems, load_knobs, target_script
from kohakumoss.training.knobs import ALGOS, CHOICES
from kohakumoss.training.weights import component
from kohakumoss.weights import DIT, TEXT
from kohakumoss_data import DataConfig

from .configuration import ROOT, TrainSettings, cache_encoders, native_knobs
from .operations import catalog

TRAIN_SCRIPT = "scripts/train/moss.py"
BINDINGS = {
    "dataset_path": ["DATASETS"],
    "source": ["DATASETS"],
    "caption_format": ["DATASETS", "CAPTION"],
    "train_repeat": ["DATASETS"],
    "reg_enabled": ["DATASETS"],
    "reg_dataset_path": ["DATASETS"],
    "reg_source": ["DATASETS"],
    "reg_caption_format": ["DATASETS"],
    "reg_repeat": ["DATASETS"],
    "reg_loss_weight": ["DATASETS"],
    "transformer_path": ["TRANSFORMER"],
    "text_encoder_path": ["TEXT_ENCODER"],
    "vae_path": ["VAE"],
    "algorithm": ["ALGO", "DORA", "DORA_DETACH", "ALGO_KWARGS"],
    "rank": ["DIM"],
    "alpha": ["ALPHA"],
    "factor": ["ALGO_KWARGS"],
    "learning_rate": ["LR"],
    "steps": ["STEPS"],
    "resolution": ["TARGET_AREA", "SAMPLE_SIZE"],
    "max_tokens": ["MAX_TOKENS"],
    "batching": ["BATCHING"],
    "batch_size": ["BATCH_SIZE"],
    "precision": ["PRECISION"],
    "memory_profile": ["MEMORY_PROFILE"],
    "grad_acc": ["GRAD_ACC"],
    "warmup": ["SCHEDULE"],
    "save_every": ["SAVE_EVERY"],
    "sample_every": ["SAMPLE_EVERY"],
    "sample_prompt": ["SAMPLE_PROMPTS", "SAMPLE_EVERY"],
    "sample_negative": ["SAMPLE_NEGATIVE"],
    "sample_steps": ["SAMPLE_STEPS"],
    "sample_cfg": ["SAMPLE_CFG"],
    "sample_seed": ["SAMPLE_SEEDS"],
    "seed": ["SEED"],
    "text_drop": ["TEXT_DROP"],
    "caption_variants": ["CAPTION_VARIANTS"],
    "workers": ["NUM_WORKERS"],
    "cache_mode": ["LATENT_CACHE", "TEXT_CACHE"],
    "tune": ["TUNE"],
}
EXTRA_CHOICES = {
    "ALGO": ALGOS,
    "OPTIMIZER": (
        "adamw",
        "adamw16",
        "adamw8bit",
        "adamw8bit_kernel",
        "muon",
        "prodigy",
        "schedule_free",
    ),
    "RESOLUTION_MODE": ("resize", "bucket", "shifted_square", "square_crop"),
    "BATCHING": ("tokens", "tokens_ffd", "count"),
    "OVERLONG": ("drop", "truncate_text", "drop_tokens"),
    "TIMESTEP": ("moss", "uniform", "logit_normal"),
    "LOSS": ("mse", "huber", "l1", "masked_mse"),
    "WEIGHTING": ("none", "x0", "eps"),
    "PARALLEL": ("none", "ddp", "fsdp", "pipeline", "threaded"),
    "SAVE_FORMAT": ("comfy", "hf", "both", "none"),
    "CROP": ("center", "random"),
    "PRECISION": ("bf16", "fp16", "auto"),
    "LATENT_CACHE": ("off", "offline", "online"),
    "TEXT_CACHE": ("off", "offline", "online"),
    "TOKEN_BUDGET": ("estimate", "render", "exact"),
}


def json_value(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def configurable(value):
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and configurable(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(configurable(item) for item in value)
    return value is None or isinstance(value, (str, bool, int, float))


@functools.cache
def operation_catalog():
    return catalog(ROOT)


def get_operation(kind):
    try:
        return operation_catalog()[kind]
    except KeyError as error:
        raise ValueError(f"Unknown native operation: {kind}") from error


@functools.cache
def schema(script: str):
    path = ROOT / script
    source = path.read_text(encoding="utf-8")
    defaults = {
        key: value
        for key, value in introspect_schema(str(path)).defaults.items()
        if key.isupper() and not key.startswith("_") and configurable(value)
    }
    tokens = list(
        tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
    )
    comments = {
        token.start[0]: token.string.lstrip("# ")
        for token in tokens
        if token.type == tokenize.COMMENT
    }
    groups = {}
    group = "GENERAL"
    for number, line in enumerate(source.splitlines(), 1):
        found = re.match(r"#\s*=+\s*(.*?)\s*=+\s*$", line)
        if found:
            group = found.group(1)
        groups[number] = group
    fields = []
    for node in ast.parse(source).body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        )
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in defaults:
                continue
            name, value = target.id, defaults[target.id]
            help_text = " ".join(
                comments.get(line, "")
                for line in range(node.lineno, node.end_lineno + 1)
            ).strip()
            field = {
                "name": name,
                "default": json_value(value),
                "group": groups[node.lineno],
                "help": help_text,
            }
            if script == TRAIN_SCRIPT:
                field["choices"] = list(EXTRA_CHOICES.get(name, CHOICES.get(name, ())))
            fields.append(field)
    if script.startswith("scripts/data/") and "../train/moss.py" in source:
        owner = schema(TRAIN_SCRIPT)
        names = {name.upper() for name in DataConfig.__dataclass_fields__}
        own_names = {field["name"] for field in fields}
        fields.extend(
            field
            for field in owner["fields"]
            if field["name"] in names and field["name"] not in own_names
        )
        defaults = {
            **{key: value for key, value in owner["defaults"].items() if key in names},
            **defaults,
        }
    names = {field["name"] for field in fields}
    defaults = {key: value for key, value in defaults.items() if key in names}
    return {
        "script": script,
        "fields": fields,
        "defaults": json_value(defaults),
        "doc": ast.get_docstring(ast.parse(source)) or "",
    }


def train_values(settings, project_id="preview", job_id="manual-run"):
    return {
        **schema(TRAIN_SCRIPT)["defaults"],
        **native_knobs(settings, project_id, job_id),
    }


def tool_values(settings, kind, output: Path, project_id="preview"):
    operation = get_operation(kind)
    if operation.driver == "python":
        return {}
    defaults = schema(operation.script)["defaults"]
    inherited = train_values(settings, project_id)
    shared = {
        "WEIGHTS",
        "VARIANT",
        "TRANSFORMER",
        "TEXT_ENCODER",
        "VAE",
        "PRECISION",
        "FP16_FIX",
        "FP16_FIX_KWARGS",
        "TUNE",
        "TEXT_DEVICE",
        "TEXT_ENCODE",
        "VAE_ENCODE",
        "VAE_DTYPE",
        "MEMORY_PROFILE",
        "MEMORY_OVERRIDES",
        "STREAM",
        "STREAM_RESIDENT",
        "PINNED",
        "BASE_PRECISION",
        "FP8_RECIPE",
        "INT8_COMPUTE",
    }
    if operation.group == "data":
        shared.update(name.upper() for name in DataConfig.__dataclass_fields__)
    values = {
        key: inherited[key]
        for key in defaults
        if key in inherited
        and key in shared
        and (inherited[key] is not None or defaults[key] is None)
    }
    values = {**defaults, **values}
    if kind in ("fp8_quant", "text_quant"):
        values["PRECISION"] = defaults["PRECISION"]
    if kind in (
        "fp16_fix",
        "fp8_quant",
        "comfy_int8",
        "probe_activations",
        "probe_backward",
    ):
        values["WEIGHTS"] = component(inherited, DIT).as_posix()
    if kind in ("text_quant", "probe_embed"):
        values["WEIGHTS"] = component(inherited, TEXT).as_posix()
    if kind in ("probe_parity", "probe_trajectory", "probe_adapter_e2e"):
        values["WEIGHTS"] = defaults["WEIGHTS"]
    if "DIT" in values:
        values["DIT"] = inherited["TRANSFORMER"]
    if "TEXT_ENCODER_KWARGS" in values:
        encoders = cache_encoders(inherited)
        values["TEXT_ENCODER"] = encoders["TEXT_ENCODER"]
        values["TEXT_ENCODER_KWARGS"] = encoders["TEXT_ENCODER_KWARGS"]
    if "IMAGE_ENCODER_KWARGS" in values:
        values["IMAGE_ENCODER_KWARGS"] = cache_encoders(inherited)[
            "IMAGE_ENCODER_KWARGS"
        ]
    if kind in ("sample", "compare"):
        values.update(
            STEPS=settings.sample_steps,
            CFG=settings.sample_cfg,
            NEGATIVE=settings.sample_negative,
            SIZES=[[settings.resolution, settings.resolution]],
            SEEDS=[settings.sample_seed],
        )
        values["PROMPT" if kind == "sample" else "PROMPTS"] = (
            settings.sample_prompt
            if kind == "sample"
            else ([settings.sample_prompt] if settings.sample_prompt else [])
        )
    options = settings.tool_options.get(kind, {}).copy()
    if "OUT" in values:
        suffix = {
            "export": ".safetensors",
            "merge": ".safetensors",
            "fp16_fix": ".safetensors",
            "fp8_quant": ".safetensors",
            "comfy_int8": ".safetensors",
            "text_quant": ".safetensors",
            "heal_merge": ".safetensors",
            "heal_rescale": ".safetensors",
            "heal_calibrate": ".json",
            "probe_activations": ".json",
            "probe_backward": ".json",
            "probe_embed": ".pt",
        }.get(kind, "")
        values["OUT"] = str(
            output
            if not suffix
            or (kind in ("export", "merge") and options.get("FORMAT") == "diffusers")
            or (kind == "merge" and options.get("FORMAT") == "hf")
            else output / f"{kind}{suffix}"
        )
    if options.get("OUT") == "":
        options.pop("OUT")
    values.update(options)
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {kind} parameters: {sorted(unknown)}")
    return values


def render_script(script, values):
    lines = [f'"""Workbench snapshot for {script}."""', ""]
    defaults = introspect_schema(str(ROOT / script)).defaults if values else {}
    for name, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z_0-9]*", name):
            raise ValueError(f"Invalid parameter name: {name}")
        if isinstance(defaults.get(name), tuple) and isinstance(value, list):
            value = tuple(value)
        lines.append(f"{name} = {literal(value)}")
    return "\n".join(lines) + "\n"


def write_operation(settings, kind, folder, output, project_id="preview"):
    operation = get_operation(kind)
    values = tool_values(settings, kind, output, project_id)
    (folder / "operation.py").write_text(
        render_script(operation.script, values), encoding="utf-8"
    )
    (folder / "operation.json").write_text(
        json.dumps(
            {
                "script": operation.script,
                "driver": operation.driver,
                "values": values,
                "args": settings.tool_args.get(kind, []),
            }
        ),
        encoding="utf-8",
    )


def validate_snapshot(path: Path):
    try:
        script = target_script(path)
        if (
            script.startswith("scripts/data/")
            and "DATASETS" not in schema(script)["defaults"]
        ):
            return []
        return config_problems(path)
    except (
        OSError,
        SyntaxError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
        SystemExit,
    ) as error:
        return [str(error)]


def required_inputs(kind, values):
    names = {
        "export": ("ADAPTER",),
        "merge": ("ADAPTERS",),
        "compare": ("CHECKPOINTS", "PROMPTS"),
        "heal_train": ("RECORDS",),
        "heal_calibrate": ("RECORDS",),
        "heal_rescale": ("RECORDS",),
        "heal_merge": ("RUN",),
        "image_index": ("ROOT",),
        "probe_parity": ("WEIGHTS",),
        "probe_trajectory": ("WEIGHTS",),
        "probe_adapter_e2e": ("WEIGHTS",),
    }.get(kind, ())
    return [f"{name} must be provided" for name in names if not values.get(name)]


def import_native(path: Path):
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value
    if path.suffix.lower() != ".py":
        raise ValueError("Choose a .json or native .py configuration")
    values = json_value(load_knobs(path))
    return {"name": path.stem, "config": settings_from_knobs(values)}


def settings_from_knobs(values):
    """Workbench fields and lossless overrides from native training knobs."""
    settings = TrainSettings().model_dump()
    for field, names in BINDINGS.items():
        if len(names) != 1 or names[0] not in values:
            continue
        value, original = values[names[0]], settings[field]
        if isinstance(original, str) and isinstance(value, str):
            if field != "precision" or value in ("fp16", "bf16"):
                try:
                    TrainSettings.model_validate({**settings, field: value})
                except ValueError:
                    continue
                settings[field] = value
        elif isinstance(original, (float, int)) and isinstance(value, (float, int)):
            settings[field] = value
    settings["native_overrides"] = values
    return TrainSettings.model_validate(settings).model_dump()
