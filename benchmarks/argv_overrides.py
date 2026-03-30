"""
Parse optional config files and --key=value overrides from sys.argv into a namespace.

Replaces the former repo-root ``configurator.py`` pattern: pass ``path/to/overrides.py``
or ``--foo=bar`` after the script name.
"""
from __future__ import annotations

import sys
from ast import literal_eval
from pathlib import Path


def apply_argv_overrides(namespace: dict) -> None:
    for arg in sys.argv[1:]:
        if "=" not in arg:
            assert not arg.startswith("--")
            config_file = arg
            print(f"Overriding config with {config_file}:")
            text = Path(config_file).read_text(encoding="utf-8")
            print(text)
            exec(compile(text, config_file, "exec"), namespace)
        else:
            assert arg.startswith("--")
            key, val = arg.split("=", 1)
            key = key[2:]
            if key not in namespace:
                raise ValueError(f"Unknown config key: {key}")
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            if type(attempt) is not type(namespace[key]):
                raise TypeError(
                    f"Override for {key!r}: got {type(attempt).__name__}, "
                    f"expected {type(namespace[key]).__name__} "
                    f"(default {namespace[key]!r})"
                )
            print(f"Overriding: {key} = {attempt}")
            namespace[key] = attempt
