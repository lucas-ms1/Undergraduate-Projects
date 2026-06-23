import ast
import importlib
import importlib.util
import os
import py_compile
import sys
import traceback


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SKIP_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".venv", "venv"}
PURE_DIRS = ("solow", "vfi", "dsge_engine", "empirical")


def iter_py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def relpath(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def module_name_from_path(path):
    rel = os.path.relpath(path, ROOT)
    no_ext = os.path.splitext(rel)[0]
    parts = no_ext.split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def project_module_path(module):
    direct = os.path.join(ROOT, *module.split(".")) + ".py"
    if os.path.exists(direct):
        return direct
    package = os.path.join(ROOT, *module.split("."), "__init__.py")
    if os.path.exists(package):
        return package
    return None


def source_symbols_for_module(module):
    path = project_module_path(module)
    if path is None:
        raise ImportError(f"No project source file found for {module}")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    symbols = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                symbols.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbols.add(alias.asname or alias.name.split(".")[0])
    return symbols


def streamlit_dependent_modules():
    modules = {"master_app", "config.theme"}
    pages_dir = os.path.join(ROOT, "pages")
    if os.path.isdir(pages_dir):
        for filename in os.listdir(pages_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                modules.add("pages." + filename[:-3])
    return modules


STREAMLIT_MODULES = streamlit_dependent_modules()


def validate_import_node(node, owner, errors):
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.name
            if name == "streamlit" or name.startswith("streamlit."):
                continue
            try:
                importlib.import_module(name)
            except Exception:
                errors.append(
                    f"{owner}:{node.lineno}: cannot import {name}\n"
                    + traceback.format_exc()
                )
        return

    if not isinstance(node, ast.ImportFrom) or not node.module:
        return

    module = node.module
    if module == "__future__" or module == "streamlit" or module.startswith("streamlit."):
        return

    names = [alias.name for alias in node.names if alias.name != "*"]
    try:
        if module in STREAMLIT_MODULES:
            symbols = source_symbols_for_module(module)
            for name in names:
                if name not in symbols:
                    errors.append(f"{owner}:{node.lineno}: {module}.{name} not found in source")
            return

        imported = importlib.import_module(module)
        for name in names:
            if not hasattr(imported, name):
                errors.append(f"{owner}:{node.lineno}: {module}.{name} does not exist")
    except Exception:
        errors.append(
            f"{owner}:{node.lineno}: cannot validate import target {module}\n"
            + traceback.format_exc()
        )


def syntax_check(files):
    errors = []
    for path in files:
        try:
            py_compile.compile(path, doraise=True)
        except Exception:
            errors.append(f"{relpath(path)}\n{traceback.format_exc()}")
    if errors:
        print(f"FAIL  Syntax check ({len(files)} files checked)")
        for err in errors:
            print(err)
    else:
        print(f"PASS  Syntax check ({len(files)} files checked)")
    return errors


def pure_logic_import_check(files):
    errors = []
    for path in files:
        rel = relpath(path)
        if not rel.startswith(PURE_DIRS):
            continue
        module = module_name_from_path(path)
        if not module:
            continue
        try:
            importlib.import_module(module)
        except Exception:
            errors.append(f"{rel} as {module}\n{traceback.format_exc()}")
    if errors:
        print("FAIL  Pure-logic imports (solow/, vfi/, dsge_engine/, empirical/)")
        for err in errors:
            print(err)
    else:
        print("PASS  Pure-logic imports (solow/, vfi/, dsge_engine/, empirical/)")
    return errors


def streamlit_target_check():
    errors = []
    targets = []
    for module in sorted(STREAMLIT_MODULES):
        path = project_module_path(module)
        if path is not None:
            targets.append((module, path))

    for module, path in targets:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            validate_import_node(node, module, errors)

    if errors:
        print("FAIL  Streamlit page import targets")
        for err in errors:
            print(err)
    else:
        print("PASS  Streamlit page import targets")
    return errors


def prefix_check():
    errors = []
    forbidden = {
        "vfi": {"config", "models", "simulation", "solvers", "utils"},
        "dsge_engine": {"config", "dsge", "simulation", "solvers", "policy", "utils"},
    }

    for package, roots in forbidden.items():
        base = os.path.join(ROOT, package)
        for dirpath, _, filenames in os.walk(base):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        root = node.module.split(".")[0]
                        if root in roots:
                            errors.append(f"{relpath(path)}:{node.lineno}: from {node.module} import ...")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            root = alias.name.split(".")[0]
                            if root in roots:
                                errors.append(f"{relpath(path)}:{node.lineno}: import {alias.name}")

    if errors:
        print("FAIL  Project-local import prefixes")
        for err in errors:
            print(err)
    else:
        print("PASS  Project-local import prefixes")
    return errors


def main():
    files = list(iter_py_files())
    failures = []
    failures.extend(syntax_check(files))
    failures.extend(pure_logic_import_check(files))
    failures.extend(streamlit_target_check())
    failures.extend(prefix_check())
    if failures:
        print(f"FAIL  test_imports: {len(failures)} issue(s)")
        sys.exit(1)
    print("PASS  test_imports: all checks passed")


if __name__ == "__main__":
    main()
