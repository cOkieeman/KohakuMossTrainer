# KohakuMoss Studio

KohakuMoss Studio is the local browser workbench included in this fork. It calls the
existing trainer, data preparation, inference and export scripts. The interface uses
Chinese labels and offers dark and light appearances.

## Install on Windows

Install the trainer and GPU dependencies following the [installation guide](install.md).
Use the repository's `.venv`, with KohakuMoss installed in editable mode so that native
commands use this fork's training entry point. Use a Node.js version supported by
Vite 6, such as Node.js 22 LTS, with npm to build the frontend.

Run these commands from the repository root in PowerShell.

```powershell
uv pip install --python .venv\Scripts\python.exe -e .
uv pip install --python .venv\Scripts\python.exe -r webapp\requirements.txt
Push-Location webapp\frontend
npm ci
npm run build
Pop-Location
.venv\Scripts\python.exe webapp\launch.py
```

The launcher reuses an existing Studio service or starts one on an available loopback
port from 8776 to 8785. It opens the resulting address in the default browser. To print
the address without opening a browser, pass `--no-browser`.

## Prepare a project

1. Create a project and select its training images and captions.
2. Enable regularization and select the project's regularization subset when needed.
   The workbench shows effective sample counts and repeat coefficients for both groups.
3. Choose compatible DiT, text-encoder and VAE files. Paths are configured per project.
   The initial model locations are under the repository's `models/` directory.
4. Set the adapter, resolution, precision, batch size, training steps and save intervals.
5. Check the data, build caches and submit training. With offline caches selected,
   submitting training prepares the required caches automatically.

The full parameter editor exposes the native training knobs. Native overrides take
precedence over the basic form. Editing a basic field clears its corresponding override.
The native-tools page provides configuration and execution controls for the scripts
available in this checkout. Linux-only operations retain the trainer's platform limits.

## Monitor and recover runs

Run history shows workbench jobs and external launches. Selecting a run opens its
actual logs, numeric metrics, saved checkpoints, exports and generated images. Complete
checkpoints can be selected for resume or initialization. Resubmitting a task uses its
saved configuration snapshot.

Training started through this fork's normal native entry point publishes a lifecycle
record automatically, including when Studio is closed. Studio discovers those records
when it starts and while it runs. This also covers custom output directories. Existing
training directories under `runs/` are imported with an unknown historical outcome
when they have no lifecycle record.

External runs belonging to a known project are associated with it. Other runs create
an imported project. Cancelling an external run checks its recorded process identity
before terminating its process tree. GPU work submitted from Studio waits while a
discovered external GPU task is active.

## Local state

Project settings, job snapshots, caches, lifecycle records and server logs are written
under `internal/studio/`. Training outputs normally use `runs/studio/`. The repository's
ignore rules keep these generated files local. Per-run log and metric snapshots preserve
earlier execution records when an output directory is reused.

The server binds to `127.0.0.1`. The workbench can browse local files, load native Python
configurations and launch trainer scripts with the current user's permissions.

## Checks

The workbench's model-free checks run separately from GPU training.

```powershell
uv pip install --python .venv\Scripts\python.exe pytest httpx
.venv\Scripts\python.exe -m pytest webapp\tests -q
Push-Location webapp\frontend
npm run build
Pop-Location
```

Desktop layouts have been inspected at 1920 by 1080, 2560 by 1440 and 2504 by 1308.
Real-model verification includes short FP16 MOSS-Image-heal LoRA training with an
official BF16 Qwen3.5 text encoder, fixed batch two, regularization, checkpoint export
and discovery of an external native training process across a Studio service restart.
Long-run stability and the quality of trained adapters require project-specific runs.
