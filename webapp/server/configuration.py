"""Validated workbench settings and native trainer configuration snapshots."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kohakumoss.cli.config.starter import literal
from kohakumoss.cli.config.validate import script_defaults
from kohakumoss.training.weights import component
from kohakumoss.weights import TEXT, VAE

ROOT = Path(__file__).resolve().parents[2]
STATE_ROOT = ROOT / "internal" / "studio"
MODEL_ROOT = ROOT / "models"
KINDS = Literal["check", "cache", "train", "doctor"]
DATA_FIELDS = (
    "dataset_path",
    "source",
    "caption_format",
    "train_repeat",
    "reg_enabled",
    "reg_dataset_path",
    "reg_source",
    "reg_caption_format",
    "reg_repeat",
    "native_overrides",
)


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = ""
    source: Literal["folder", "kohya"] = "folder"
    caption_format: Literal["nl", "tags"] = "nl"
    train_repeat: float = Field(default=1, gt=0, allow_inf_nan=False)
    reg_enabled: bool = False
    reg_dataset_path: str = ""
    reg_source: Literal["folder", "kohya"] = "folder"
    reg_caption_format: Literal["nl", "tags"] = "tags"
    reg_repeat: float = Field(default=1, gt=0, allow_inf_nan=False)
    reg_loss_weight: float = Field(default=1, ge=0, allow_inf_nan=False)
    transformer_path: str = str(
        MODEL_ROOT / "diffusion_models" / "moss_image_heal_int8_convrot.safetensors"
    )
    text_encoder_path: str = str(
        MODEL_ROOT / "text_encoders" / "qwen3.5_9b_int8_convrot.safetensors"
    )
    vae_path: str = str(MODEL_ROOT / "vae" / "flux2-vae.safetensors")
    algorithm: Literal[
        "lora",
        "locon",
        "lokr",
        "loha",
        "dora",
        "oft",
        "boft",
        "glora",
        "ia3",
        "dylora",
        "tlora",
        "full",
    ] = "lora"
    rank: int = Field(default=32, gt=0)
    alpha: float = Field(default=32, gt=0, allow_inf_nan=False)
    factor: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    steps: int = Field(default=4000, gt=0)
    resolution: int = Field(default=1024, gt=0, multiple_of=16)
    max_tokens: int = Field(default=8192, gt=0)
    batching: Literal["tokens", "count"] = "tokens"
    batch_size: int = Field(default=1, gt=0)
    precision: Literal["bf16", "fp16"] = "bf16"
    memory_profile: str = "low"
    grad_acc: int = Field(default=1, gt=0)
    warmup: int = Field(default=100, ge=0)
    save_every: int = Field(default=500, ge=0)
    sample_every: int = Field(default=500, ge=0)
    sample_prompt: str = ""
    sample_negative: str = ""
    sample_steps: int = Field(default=25, gt=0)
    sample_cfg: float = Field(default=3, ge=0, allow_inf_nan=False)
    sample_seed: int = Field(default=42, ge=0)
    seed: int = Field(default=42, ge=0)
    text_drop: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    caption_variants: int = Field(default=1, gt=0)
    workers: int = Field(default=2, ge=0)
    cache_mode: Literal["offline", "online", "off"] = "offline"
    tune: Literal["table", "off"] = "table"
    native_overrides: dict = Field(default_factory=dict)
    tool_options: dict[str, dict] = Field(default_factory=dict)
    tool_args: dict[str, list[str]] = Field(default_factory=dict)


class ProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="New training", min_length=1, max_length=120)
    config: TrainSettings = Field(default_factory=TrainSettings)

    @model_validator(mode="after")
    def strip_name(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Project name must contain text")
        return self


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str


def validate_paths(settings: TrainSettings, kind: str) -> None:
    if kind not in ("check", "cache", "train"):
        return
    data = Path(settings.dataset_path).expanduser()
    if "DATASETS" in settings.native_overrides:
        if not settings.native_overrides["DATASETS"]:
            raise ValueError("DATASETS must contain a dataset")
    elif not settings.dataset_path.strip() or not data.is_dir():
        raise ValueError("Choose an existing dataset directory")
    if settings.reg_enabled and "DATASETS" not in settings.native_overrides:
        regularization = Path(settings.reg_dataset_path).expanduser()
        if not settings.reg_dataset_path.strip() or not regularization.is_dir():
            raise ValueError("Choose an existing regularization directory")
    if kind == "check":
        return
    fields = ["text_encoder_path", "vae_path"]
    if kind == "train":
        fields.append("transformer_path")
    for field in fields:
        native_key = {
            "text_encoder_path": "TEXT_ENCODER",
            "vae_path": "VAE",
            "transformer_path": "TRANSFORMER",
        }[field]
        if native_key in settings.native_overrides:
            continue
        path = Path(getattr(settings, field)).expanduser()
        if not path.exists():
            raise ValueError(f"{field} does not exist: {path}")


def caption_config(caption_format: str) -> dict:
    caption = {"template": caption_format}
    if caption_format == "tags":
        caption["shuffle"] = False
    return caption


def dataset_entry(path, source, caption_format, repeat, loss_weight=1.0):
    dataset = {
        "source": source,
        "path": (Path(path).expanduser().resolve().as_posix() if path.strip() else ""),
        "txt_as": caption_format,
        "caption": caption_config(caption_format),
        "weight": repeat,
        "loss_weight": loss_weight,
    }
    if source == "kohya":
        dataset.update(repeat_prefix="first", include_root=False)
    return dataset


def dataset_entries(settings: TrainSettings) -> list[dict]:
    if "DATASETS" in settings.native_overrides:
        return settings.native_overrides["DATASETS"]
    entries = [
        dataset_entry(
            settings.dataset_path,
            settings.source,
            settings.caption_format,
            settings.train_repeat,
        )
    ]
    if settings.reg_enabled:
        entries.append(
            dataset_entry(
                settings.reg_dataset_path,
                settings.reg_source,
                settings.reg_caption_format,
                settings.reg_repeat,
                settings.reg_loss_weight,
            )
        )
    return entries


def native_knobs(settings: TrainSettings, project_id: str, job_id: str) -> dict:
    algo = "lora" if settings.algorithm == "dora" else settings.algorithm
    knobs = {
        "WEIGHTS": MODEL_ROOT.as_posix(),
        "TRANSFORMER": Path(settings.transformer_path).as_posix(),
        "TEXT_ENCODER": Path(settings.text_encoder_path).as_posix(),
        "VAE": Path(settings.vae_path).as_posix(),
        "BASE_PRECISION": "auto",
        "DATASETS": dataset_entries(settings),
        "CAPTION": caption_config(settings.caption_format),
        "ALGO": algo,
        "DIM": settings.rank,
        "ALPHA": settings.alpha,
        "DORA": settings.algorithm == "dora",
        "DORA_DETACH": settings.algorithm == "dora",
        "LR": settings.learning_rate,
        "STEPS": settings.steps,
        "TARGET_AREA": settings.resolution**2,
        "MAX_TOKENS": settings.max_tokens,
        "BATCHING": settings.batching,
        "BATCH_SIZE": settings.batch_size,
        "PRECISION": settings.precision,
        "FP16_FIX": "none",
        "MEMORY_PROFILE": settings.memory_profile,
        "TEXT_ENCODE": {"memory": "ring", "ahead": 4},
        "VAE_ENCODE": {"memory": "offload", "tile": True},
        "GRAD_ACC": settings.grad_acc,
        "SCHEDULE": {
            "mode": "cosine",
            "warmup": settings.warmup,
            "end": -1,
            "min_value": 0.1,
        },
        "SAVE_EVERY": settings.save_every,
        "KEEP_LAST": 0,
        "SAMPLE_EVERY": settings.sample_every if settings.sample_prompt.strip() else 0,
        "SAMPLE_PROMPTS": (
            [settings.sample_prompt] if settings.sample_prompt.strip() else []
        ),
        "SAMPLE_NEGATIVE": settings.sample_negative,
        "SAMPLE_STEPS": settings.sample_steps,
        "SAMPLE_CFG": settings.sample_cfg,
        "SAMPLE_SIZE": (settings.resolution, settings.resolution),
        "SAMPLE_SEEDS": [settings.sample_seed],
        "SAMPLE_AT_START": False,
        "SEED": settings.seed,
        "TEXT_DROP": settings.text_drop,
        "NUM_WORKERS": settings.workers,
        "CAPTION_VARIANTS": settings.caption_variants,
        "LATENT_CACHE": settings.cache_mode,
        "TEXT_CACHE": settings.cache_mode,
        "CACHE_DIR": (STATE_ROOT / "cache" / project_id).as_posix(),
        "OUTPUT_DIR": (ROOT / "runs" / "studio" / project_id).as_posix(),
        "RUN_NAME": job_id,
        "SAVE_FORMAT": "both",
        "LOG_EVERY": 1,
        "METRICS_EVERY": 1,
        "TUNE": settings.tune,
    }
    if settings.algorithm == "lokr":
        knobs["ALGO_KWARGS"] = {"factor": settings.factor}
    knobs.update(settings.native_overrides)
    return knobs


def render_train(settings: TrainSettings, project_id: str, job_id: str) -> str:
    base = (ROOT / "configs" / "moss" / "_base" / "base_lora.py").as_posix()
    lines = [
        '"""Workbench snapshot for scripts/train/moss.py."""',
        "",
        "from kohakuengine import use_config",
        "",
        f"use_config({literal(base)})",
        "",
    ]
    defaults = script_defaults(ROOT, "scripts/train/moss.py")
    for key, value in native_knobs(settings, project_id, job_id).items():
        if isinstance(defaults.get(key), tuple) and isinstance(value, list):
            value = tuple(value)
        lines.append(f"{key} = {literal(value)}")
    return "\n".join(lines) + "\n"


def cache_encoders(knobs: dict) -> dict:
    devices = knobs.get("ENCODE_DEVICES") or knobs.get("DEVICES") or (0,)
    selected = devices[0]
    device = f"cuda:{selected}" if isinstance(selected, int) else str(selected)
    return {
        "TEXT_ENCODER": "kohakumoss.encoders.text_encoder",
        "TEXT_ENCODER_KWARGS": {
            "path": component(knobs, TEXT).as_posix(),
            "device": knobs.get("TEXT_DEVICE") or device,
            "plan": knobs.get("TEXT_ENCODE") or {},
        },
        "IMAGE_ENCODER": "kohakumoss.encoders.vae_encoder",
        "IMAGE_ENCODER_KWARGS": {
            "path": component(knobs, VAE).as_posix(),
            "device": device,
            "dtype": knobs.get("VAE_DTYPE", "auto"),
            "mode": knobs.get("LATENT_DRAW", "mode"),
            "plan": knobs.get("VAE_ENCODE") or {},
        },
    }


def write_snapshot(settings: TrainSettings, project_id: str, job_id: str) -> Path:
    folder = STATE_ROOT / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=False)
    train = folder / "train.py"
    train.write_text(render_train(settings, project_id, job_id), encoding="utf-8")
    data_lines = [
        '"""Workbench data preparation for scripts/data/inspect_data.py."""',
        "",
        "from kohakuengine import use_config",
        "",
        f"use_config({literal(train.as_posix())})",
        "",
    ]
    knobs = native_knobs(settings, project_id, job_id)
    data_lines.extend(
        f"{key} = {literal(value)}" for key, value in cache_encoders(knobs).items()
    )
    (folder / "data.py").write_text("\n".join(data_lines) + "\n", encoding="utf-8")
    return folder
