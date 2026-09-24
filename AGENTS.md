# Repository Guidelines

## Project Structure & Module Organization

This is a Python desktop application. `main.py` starts the GUI; shared event and interface logic lives in `application_core.py` and `interface_core.py`. Sensor integrations are grouped under `sensors/` (`camera/`, `radar/`, and `gps/`). Recording, playback, and visualization code lives in `processing/`; QR camera calibration lives in `calibration/`. Automated tests are in `tests/`, while `content/` contains reference documents. Runtime recordings and snapshots are data outputs, not source code.

## Build, Test, and Development Commands

- `python3 main.py` — run the application from the repository root; the host also needs its native GStreamer runtime and plugins.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q` — run the full test suite. Disabling plugin autoload avoids unrelated system pytest plugins.
- `python3 -m pytest -q tests/camera/test_pipeline_policy.py` — run one focused test module.

There is no separate build step or declared formatter/linter configuration. Dependencies are listed in `requirements.txt`.

## Coding Style & Naming Conventions

Follow the surrounding Python style: four spaces for indentation, descriptive `snake_case` module/function/variable names, and `PascalCase` classes. Keep hardware, recording, and calibration behavior in their existing modules. Prefer small changes that preserve established recording schemas and worker communication contracts.

## Testing Guidelines

Tests use `pytest` and are organized by behavior. Name files `test_<behavior>.py` and test functions `test_<expected_behavior>`. Add or update focused tests for behavior changes, then run the full suite with the command above. The suite uses mocks and synthetic data; it does not prove operation with real radar, DVR, camera, or display hardware.

## Commit & Pull Request Guidelines

Recent commits generally use short imperative subjects, for example `Improve calibration evidence validation and timing diagnostics`. Keep commit titles concise and action-oriented. Pull requests should explain the change and its motivation, link a related issue when one exists, list relevant test results, and include screenshots for visible GUI changes. Note any hardware or deployment assumptions reviewers need to check.

## Configuration & Data Safety

Device addresses and DVR credentials are currently embedded in source. Do not add real credentials to examples or publish deployment-specific values. Avoid committing generated recordings, snapshots, or analysis output unless they are explicitly needed as test fixtures.
