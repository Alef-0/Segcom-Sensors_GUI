# Repository Guidelines

## Project Structure & Module Organization

This is a Python desktop application. The GUI at the root is structured into three dedicated layers plus an entry point:
- `main.py` — starts the application, parses CLI arguments via `argparse`, and initializes background multiprocessing workers.
- `menu_layout.py` — declarative layout generation (FreeSimpleGUI frames, tabs, buttons, and inputs). Pure layout without event logic.
- `menu_controls.py` — window and widget controller (`MenuControls`). Manages UI state transitions, enables/disables controls, updates labels, and validates/filters inputs.
- `menu_functions.py` — application event dispatcher and IPC bridge. Bundles worker pipes (`WorkerPipes`), dispatches GUI events, drains status queues, and handles worker process lifecycle and shutdown.

Sensor integrations are grouped under `sensors/` (`camera/`, `radar/`, and `gps/`). Recording, playback, and visualization code lives in `processing/`; QR camera calibration lives in `calibration/`. Automated tests are in `tests/`, while `content/` contains reference documents. Runtime recordings and snapshots are data outputs, not source code.

## Build, Test, and Development Commands

- `python3 main.py` — run the application from the repository root; the host also needs its native GStreamer runtime and plugins.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q` — run the full test suite. Disabling plugin autoload avoids unrelated system pytest plugins.
- `python3 -m pytest -q tests/camera/test_pipeline_policy.py` — run one focused test module.

There is no separate build step or declared formatter/linter configuration. Dependencies are listed in `requirements.txt`.

## Coding Style & Naming Conventions

- Follow standard Python style: four spaces for indentation, descriptive `snake_case` function/variable names, and `PascalCase` classes.
- Avoid abusive vertical space: do not split method arguments, simple tuples, or small UI rows onto single-item lines if they fit comfortably within standard line width (88-100 characters). Keep layouts and calls compact and readable.
- Descriptive naming without unnecessary pseudo-private prefixes: use clear public function and method names (`run_event_loop`, `handle_gui_event`, `start_recording`) rather than prefixing standard internal functions with leading underscores (`_`), unless strictly implementing private class encapsulation.
- Bundle repetitive parameters: group worker IPC channels into dedicated data structures (such as `WorkerPipes`) rather than threading 5+ separate pipe arguments across multiple function signatures.
- Keep docstrings concise (max 4 lines) with tight whitespace. Place variable explanations inline beside variables in the function scope when needed.

## Testing Guidelines

Tests use `pytest` and are organized by behavior. Name files `test_<behavior>.py` and test functions `test_<expected_behavior>`. Add or update focused tests for behavior changes, then run the full suite with the command above. The suite uses mocks and synthetic data; it does not prove operation with real radar, DVR, camera, or display hardware.

## Commit & Pull Request Guidelines

Recent commits generally use short imperative subjects, for example `Improve calibration evidence validation and timing diagnostics`. Keep commit titles concise and action-oriented. Pull requests should explain the change and its motivation, link a related issue when one exists, list relevant test results, and include screenshots for visible GUI changes. Note any hardware or deployment assumptions reviewers need to check.

## Configuration & Data Safety

Device addresses and DVR credentials are currently embedded in source. Do not add real credentials to examples or publish deployment-specific values. Avoid committing generated recordings, snapshots, or analysis output unless they are explicitly needed as test fixtures.

# Additional things
- Any README file consolidate inside of the SUMMARY.MD