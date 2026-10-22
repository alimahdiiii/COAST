"""
Renders the API of this package as Markdown, so that the docstrings appear with
code highlighting rather than as plain text.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

def show_api(obj: Any, title: Optional[str] = None) -> None:
    """Render an object's docstring nicely in a Jupyter notebook.

    Parameters
    ----------
    obj:
        Function/class/module to document.
    title:
        Optional custom title.

    Notes
    -----
    Requires Jupyter/IPython. In a regular Python script, it falls back to `print()`.
    """
    doc = inspect.getdoc(obj) or "(No docstring found.)"
    try:
        sig = str(inspect.signature(obj))
    except Exception:
        sig = None

    name = getattr(obj, '__name__', 'object')
    header = f"### {title or name}\n"
    if sig:
        header += f"**Signature:** `{name}{sig}`\n\n"

    md = header + "```text\n" + doc + "\n```\n"

    try:
        from IPython.display import Markdown, display  # type: ignore
        display(Markdown(md))
    except Exception:
        print(md)
