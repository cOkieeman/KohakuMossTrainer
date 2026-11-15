"""Native script operations exposed by the local workbench."""

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Operation:
    id: str
    label: str
    group: str
    script: str
    driver: str = "kogine"
    linux: bool = False


OPERATIONS = {
    item.id: item
    for item in (
        Operation(
            "sample", "Standalone sampling", "inference", "scripts/sample/sample.py"
        ),
        Operation(
            "compare",
            "Checkpoint comparison grid",
            "inference",
            "scripts/sample/compare_grid.py",
        ),
        Operation("export", "Checkpoint export", "export", "scripts/export/export.py"),
        Operation("merge", "Merge adapters", "export", "scripts/export/merge.py"),
        Operation(
            "fp16_fix", "FP16 checkpoint fix", "export", "scripts/export/fp16_fix.py"
        ),
        Operation(
            "fp8_quant", "FP8 / MXFP8 export", "export", "scripts/export/fp8_quant.py"
        ),
        Operation(
            "comfy_int8",
            "INT8 ConvRot export",
            "export",
            "scripts/export/comfy_int8.py",
        ),
        Operation(
            "text_quant",
            "Text encoder quantization",
            "export",
            "scripts/export/text_quant.py",
        ),
        Operation(
            "catalog_build",
            "Build dataset catalogs",
            "data",
            "scripts/data/catalog_build.py",
        ),
        Operation(
            "catalog_cache",
            "Catalog cache management",
            "data",
            "scripts/data/catalog_cache.py",
        ),
        Operation("cache_text", "Text cache", "data", "scripts/data/cache_text.py"),
        Operation(
            "cache_latents", "Latent cache", "data", "scripts/data/cache_latents.py"
        ),
        Operation(
            "cache_merge", "Merge cache shards", "data", "scripts/data/cache_merge.py"
        ),
        Operation(
            "image_index",
            "Image index",
            "data",
            "scripts/data/image_index.py",
        ),
        Operation("heal_cache", "Heal cache", "heal", "scripts/heal/cache.py"),
        Operation(
            "heal_calibrate", "Heal calibration", "heal", "scripts/heal/calibrate.py"
        ),
        Operation("heal_rescale", "Heal rescale", "heal", "scripts/heal/rescale.py"),
        Operation("heal_merge", "Heal merge", "heal", "scripts/heal/merge.py"),
        Operation("heal_train", "Heal training", "heal", "scripts/heal/heal.py"),
        Operation("tune", "Kernel tuning", "system", "scripts/bench/tune/tune.py"),
        Operation(
            "check_comm",
            "Distributed communication check",
            "distributed",
            "scripts/dist/check_comm.py",
            linux=True,
        ),
        Operation(
            "scaling_report",
            "Distributed scaling report",
            "distributed",
            "scripts/dist/scaling_report.py",
        ),
    )
}


def catalog(root: Path) -> dict[str, Operation]:
    result = OPERATIONS.copy()
    for group in ("bench", "probe"):
        for path in sorted((root / "scripts" / group).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if any(item.script == relative for item in result.values()):
                continue
            source = path.read_text(encoding="utf-8")
            if (
                '__name__ == "__main__"' not in source
                and "__name__ == '__main__'" not in source
            ):
                continue
            tree = ast.parse(source)
            doc = ast.get_docstring(tree) or path.stem
            key = "_".join(path.relative_to(root / "scripts").with_suffix("").parts)
            driver = (
                "kogine"
                if "kogine" in doc or "from kohakuengine" in source
                else "python"
            )
            if "ArgumentParser" in source:
                driver = "python"
            result[key] = Operation(key, doc.splitlines()[0], group, relative, driver)
    return result
