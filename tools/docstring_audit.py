"""Docstring audit tool

Scans Python files under src/ and reports:
- missing module/class/function docstrings
- very short / unfinished docstrings
- TODO/FIXME/WIP markers
- param documentation mismatches vs signature (Sphinx :param style)
- return documentation mismatches (:return/:rtype)
- likely Sphinx role mistakes like ":py::"

Produces docstring_report.json and docstring_report.md in repository root.
"""

import ast
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_GLOB = ROOT / 'src' / '**' / '*.py'

ISSUES = []

PARAM_RE = re.compile(r":param\s+(?P<name>\w+)\s*:")
RETURN_RE = re.compile(r":return\b|:rtype\b")
PYROLE_RE = re.compile(r":py::")
TODO_RE = re.compile(r"\b(TODO|FIXME|WIP)\b", re.I)


def analyze_docstring(node, docstring, kind, fullname, lineno, sig_params=None):
    issues = []
    if docstring is None:
        issues.append({'type':'missing_docstring','msg':f'Missing {kind} docstring','lineno':lineno})
        return issues
    # short/unhelpful
    if len(docstring.strip()) < 20:
        issues.append({'type':'short_docstring','msg':'Docstring very short/uninformative','lineno':lineno})
    # TODO markers
    if TODO_RE.search(docstring):
        issues.append({'type':'todo_marker','msg':'Contains TODO/FIXME/WIP marker','lineno':lineno})
    # sphinx py role mistakes
    if PYROLE_RE.search(docstring):
        issues.append({'type':'sphinx_role','msg':'Found ":py::" sequence — looks like a double-colon typo (use ":py:class:`...`")','lineno':lineno})
    # params
    doc_params = PARAM_RE.findall(docstring)
    if sig_params is not None:
        # compare
        sig_set = [p for p in sig_params if p not in ('self','cls','*','**')]
        for p in sig_set:
            if p not in doc_params:
                issues.append({'type':'param_missing','msg':f'Parameter "{p}" missing from docstring','lineno':lineno})
        for dp in doc_params:
            if dp not in sig_set:
                issues.append({'type':'param_extra','msg':f'Docstring documents parameter "{dp}" which is not in signature','lineno':lineno})
    # return
    has_return_doc = bool(RETURN_RE.search(docstring))
    has_return_stmt = False
    if isinstance(node, ast.FunctionDef):
        for n in ast.walk(node):
            if isinstance(n, ast.Return):
                if n.value is not None:
                    has_return_stmt = True
                    break
    if has_return_doc and not has_return_stmt:
        issues.append({'type':'return_mismatch','msg':'Docstring documents return value but function has no return with value','lineno':lineno})
    if has_return_stmt and not has_return_doc:
        issues.append({'type':'return_missing','msg':'Function returns a value but docstring lacks :return/:rtype','lineno':lineno})
    # math directive rough check: ``.. math::`` should be followed by blank line; flag if used inline
    if '.. math::' in docstring and '\n\n' not in docstring.split('.. math::',1)[1][:40]:
        issues.append({'type':'math_format','msg':'".. math::" found but no blank line after it — sphinx requires block separation','lineno':lineno})
    return issues


def analyze_file(path):
    text = path.read_text()
    tree = ast.parse(text)
    attach_parents(tree)
    module_doc = ast.get_docstring(tree)
    file_issues = []
    if module_doc is None:
        file_issues.append({'type':'missing_module_docstring','msg':'Missing module docstring','lineno':1})
    else:
        file_issues.extend(analyze_docstring(tree, module_doc, 'module', path.stem, 1, sig_params=None))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            fullname = f"{path.stem}.{node.name}"
            lineno = node.lineno
            file_issues.extend(analyze_docstring(node, doc, 'class', fullname, lineno))
            # check __init__ params vs class docstring? skip
            # analyze methods
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    sig_params = [a.arg for a in n.args.args]
                    doc = ast.get_docstring(n)
                    fullname = f"{path.stem}.{node.name}.{n.name}"
                    file_issues.extend(analyze_docstring(n, doc, 'method', fullname, n.lineno, sig_params=sig_params))
        elif isinstance(node, ast.FunctionDef):
            # top-level functions
            if isinstance(getattr(node,'parent',None), ast.ClassDef):
                continue
            sig_params = [a.arg for a in node.args.args]
            doc = ast.get_docstring(node)
            fullname = f"{path.stem}.{node.name}"
            file_issues.extend(analyze_docstring(node, doc, 'function', fullname, node.lineno, sig_params=sig_params))
    return file_issues

# attach parents to nodes for easy checks
def attach_parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def main():
    report = {}
    files = sorted([Path(p) for p in glob.glob(str(SRC_GLOB), recursive=True)])
    total_issues = 0
    for f in files:
        try:
            text = f.read_text()
            tree = ast.parse(text)
            attach_parents(tree)
            issues = analyze_file(f)
            if issues:
                report[str(f.relative_to(ROOT))] = issues
                total_issues += len(issues)
        except Exception as e:
            report[str(f.relative_to(ROOT))] = [{'type':'parse_error','msg':str(e),'lineno':0}]
            total_issues += 1
    out_json = ROOT / 'docstring_report.json'
    out_md = ROOT / 'docstring_report.md'
    with open(out_json,'w') as fh:
        json.dump(report, fh, indent=2)
    # write simple markdown summary
    with open(out_md,'w') as fh:
        fh.write('# Docstring audit report\n\n')
        fh.write(f'Found {len(report)} files with issues. Total flagged items: {total_issues}\\n\\n')
        for fn, issues in report.items():
            fh.write(f'## {fn}\n')
            for it in issues:
                fh.write(f'- [line {it.get("lineno", "?")}] {it.get("type")} : {it.get("msg")}\n')
            fh.write('\n')
    print(f'Wrote {out_json} and {out_md}')

if __name__ == '__main__':
    main()
