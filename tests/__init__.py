"""Test package initialization with test-environment GUI shims."""

import sys


if "FreeSimpleGUI" not in sys.modules:
    try:
        import FreeSimpleGUI  # noqa: F401
    except ImportError:
        class _DummyElement:
            def __init__(self, *args, key=None, default=None, **kwargs):
                self.Key = key
                self.InitialState = default
                self.ButtonText = (
                    args[0]
                    if args and isinstance(args[0], str)
                    else kwargs.get("button_text", "")
                )
                self.Rows = kwargs.get("layout") or (
                    args[0] if args and isinstance(args[0], list) else None
                )
                self.kwargs = kwargs

            def __getattr__(self, _name):
                return None

        class _DummySG:
            def __getattr__(self, name):
                if name in (
                    "Text", "Button", "Checkbox", "Radio", "Combo",
                    "Column", "Frame", "Push", "VSep", "HSep",
                    "FolderBrowse", "Input", "Window",
                ):
                    return _DummyElement
                return lambda *args, **kwargs: None

        sys.modules["FreeSimpleGUI"] = _DummySG()
