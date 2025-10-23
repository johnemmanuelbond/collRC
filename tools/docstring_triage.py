"""Simple triage script for docstrings

Performs safe, automated fixes:
- add a minimal module docstring if missing
- replace instances of ":py::" with ":py:"
- ensure there's a blank line after ".. math::" directives

This is intentionally conservative and only touches whitespace/roles/module docstrings.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'

math_re = re.compile(r"(\n|\A)(\s*)\.\. math::\n(?!\n)")
pyrole_re = re.compile(r":py::")

changed_files = []

for p in sorted(SRC.rglob('*.py')):
    txt = p.read_text()
    orig = txt
    # add module docstring if missing
    # find first non-shebang, non-coding comment line
    if txt.lstrip().startswith('#!'):
        # skip shebang
        pass
    # parse for existing module docstring: look for triple-quoted string at top
    # match optional shebang/comments then a triple-quoted string (''' or """)
    m = re.match(r"\s*(?:#.*\n)*(?:'''|\"\"\")", txt)
    if not m:
        # Insert a minimal module docstring after any shebang or coding header
        insert_at = 0
        # if there's a coding cookie on first two lines, preserve it
        lines = txt.splitlines(True)
        if len(lines) >= 1 and lines[0].startswith('#!'):
            insert_at = len(lines[0])
            if len(lines) >= 2 and 'coding' in lines[1]:
                insert_at += len(lines[1])
        module_name = p.relative_to(ROOT)
        doc = f'"""Module {module_name}."""\n\n'
        txt = txt[:insert_at] + doc + txt[insert_at:]
    # replace accidental double-colon py roles
    txt = pyrole_re.sub(':py:', txt)
    # ensure blank line after .. math::
    txt = math_re.sub(lambda m: f"{m.group(1)}{m.group(2)}.. math::\n\n", txt)

    if txt != orig:
        p.write_text(txt)
        changed_files.append(str(p.relative_to(ROOT)))

print('Triage applied to:', changed_files)
