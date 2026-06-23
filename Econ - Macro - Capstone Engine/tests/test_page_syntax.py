import ast
import importlib
import os
import sys
import traceback


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PAGES_DIR = os.path.join(ROOT, "pages")
STREAMLIT_DEPENDENT = {"config.theme", "master_app"}


def module_path(module):
    direct = os.path.join(ROOT, *module.split(".")) + ".py"
    if os.path.exists(direct):
        return direct
    package = os.path.join(ROOT, *module.split("."), "__init__.py")
    if os.path.exists(package):
        return package
    return None


def source_symbols(module):
    path = module_path(module)
    if path is None:
        raise ImportError(f"No source file for {module}")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def validate_import_targets(filename, tree):
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if "streamlit" in mod or mod == "__future__":
                continue
            try:
                if mod in STREAMLIT_DEPENDENT or mod.startswith("pages."):
                    symbols = source_symbols(mod)
                    for alias in node.names or []:
                        if alias.name != "*" and alias.name not in symbols:
                            errors.append(f"{filename}: {mod}.{alias.name} does not exist")
                else:
                    imported = importlib.import_module(mod)
                    for alias in node.names or []:
                        if alias.name != "*" and not hasattr(imported, alias.name):
                            errors.append(f"{filename}: {mod}.{alias.name} does not exist")
            except Exception as exc:
                errors.append(f"{filename}: Cannot import {mod}: {exc}\n{traceback.format_exc()}")
        elif isinstance(node, ast.Import):
            for alias in node.names or []:
                mod = alias.name
                if "streamlit" in mod:
                    continue
                try:
                    importlib.import_module(mod)
                except Exception as exc:
                    errors.append(f"{filename}: Cannot import {mod}: {exc}\n{traceback.format_exc()}")
    return errors


def check_file(path, require_render):
    filename = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        print(f"FAIL  {filename}: SyntaxError at line {exc.lineno}: {exc.msg}")
        return [f"{filename}: syntax error"]

    errors = []
    if require_render:
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if "render" not in functions:
            msg = f"{filename}: No render() function found"
            print(f"WARN  {msg}")
            errors.append(msg)
        else:
            print(f"PASS  {filename}: render() exists")
    else:
        print(f"PASS  {filename}: Parses without SyntaxError")

    import_errors = validate_import_targets(filename, tree)
    for err in import_errors:
        print(f"FAIL  {err}")
    errors.extend(import_errors)
    return errors


def main():
    failures = []
    expected_pages = ["empirical.py", "solow.py", "vfi_models.py", "dsge_fiscal.py"]
    for filename in expected_pages:
        failures.extend(check_file(os.path.join(PAGES_DIR, filename), require_render=True))
    failures.extend(check_file(os.path.join(ROOT, "master_app.py"), require_render=False))
    if failures:
        print(f"FAIL  test_page_syntax: {len(failures)} issue(s)")
        sys.exit(1)
    print("PASS  test_page_syntax: all checks passed")


if __name__ == "__main__":
    main()
