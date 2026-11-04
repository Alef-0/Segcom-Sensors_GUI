"""Test package initialization with test-environment GUI shims."""

import sys


if "FreeSimpleGUI" not in sys.modules:
    try:
        import FreeSimpleGUI  # noqa: F401
    except ImportError:
        class _DummyTk:
            def eval(self, *_args, **_kwargs):
                return "popdown"

            def call(self, *_args, **_kwargs):
                return None

        class _DummyWidget:
            def __init__(self):
                self.tk = _DummyTk()

            def configure(self, *_args, **_kwargs):
                return None

        class _DummyElement:
            ElementType = "Element"

            def __init__(self, *args, key=None, default=None, **kwargs):
                self.Key = key
                self.InitialState = default
                self.Title = (
                    args[0]
                    if args and isinstance(args[0], str)
                    else kwargs.get("title", "")
                )
                self.ButtonText = (
                    args[0]
                    if args and isinstance(args[0], str)
                    else kwargs.get("button_text", "")
                )
                rows = kwargs.get("layout")
                if rows is None:
                    if args and isinstance(args[0], list):
                        rows = args[0]
                    elif len(args) > 1 and isinstance(args[1], list):
                        rows = args[1]
                self.Rows = rows
                self.Widget = _DummyWidget()
                self.Value = args[0] if args else None
                self.kwargs = kwargs

            def update(self, value=None, **kwargs):
                if value is not None:
                    self.Value = value
                self.kwargs.update(kwargs)

            def erase(self):
                return None

            def draw_image(self, **_kwargs):
                return None

            def __getattr__(self, _name):
                return None

        def _walk_dummy_elements(rows):
            if not rows:
                return
            for row in rows:
                items = row if isinstance(row, (list, tuple)) else (row,)
                for item in items:
                    if isinstance(item, _DummyElement):
                        yield item
                        if item.Rows:
                            yield from _walk_dummy_elements(item.Rows)

        class _DummyWindow(_DummyElement):
            ElementType = "Window"

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._elements = list(_walk_dummy_elements(self.Rows))
                self._by_key = {
                    el.Key: el for el in self._elements if el.Key is not None
                }

            def element_list(self):
                return list(self._elements)

            def __getitem__(self, key):
                if key not in self._by_key:
                    self._by_key[key] = _DummyElement(key=key)
                return self._by_key[key]

            def read(self, timeout=None):
                return "__TIMEOUT__", {
                    k: el.InitialState for k, el in self._by_key.items()
                }

            def close(self):
                return None

            @staticmethod
            def read_all_windows(timeout=None):
                return None, "__TIMEOUT__", {}

        _WIDGET_CLASSES = {
            name: type(name, (_DummyElement,), {"ElementType": name})
            for name in (
                "Text", "Button", "Checkbox", "Radio", "Combo", "Slider",
                "Column", "Frame", "Push", "VSep", "HSep",
                "HorizontalSeparator", "FolderBrowse", "Input",
                "Tab", "TabGroup", "Graph", "Listbox",
            )
        }
        _WIDGET_CLASSES["Window"] = _DummyWindow

        class _DummySG:
            WIN_CLOSED = "WIN_CLOSED"
            WINDOW_CLOSED = "WINDOW_CLOSED"
            TITLE_LOCATION_TOP = "top"

            def __getattr__(self, name):
                if name in _WIDGET_CLASSES:
                    return _WIDGET_CLASSES[name]
                return lambda *args, **kwargs: None

        sys.modules["FreeSimpleGUI"] = _DummySG()
