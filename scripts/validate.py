#!/usr/bin/env python3
"""
Production validation script for DeepFake Detector.

This script validates the repository without requiring all dependencies installed.
Run this before deploying to production.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def validate_python_syntax(directory: Path) -> Tuple[bool, List[str]]:
    """Validate Python syntax for all files in directory."""
    errors = []
    python_files = list(directory.rglob("*.py"))

    for file_path in python_files:
        if "__pycache__" in str(file_path):
            continue

        try:
            with open(file_path) as f:
                ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            errors.append(f"{file_path}: {e}")

    return len(errors) == 0, errors


def check_imports(directory: Path) -> Tuple[bool, List[str]]:
    """Check for common import issues."""
    issues = []
    python_files = list(directory.rglob("*.py"))

    for file_path in python_files:
        if "__pycache__" in str(file_path):
            continue

        try:
            with open(file_path) as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))

            # Check for relative imports without package
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module is None and node.level > 0:
                        # Relative import - ensure we're in a package
                        if not (file_path.parent / "__init__.py").exists():
                            issues.append(
                                f"{file_path}: Relative import in non-package"
                            )
        except Exception as e:
            issues.append(f"{file_path}: {e}")

    return len(issues) == 0, issues


def validate_test_structure(test_dir: Path) -> Tuple[bool, List[str]]:
    """Validate test file structure."""
    issues = []

    if not test_dir.exists():
        return False, ["Test directory does not exist"]

    # Check for conftest.py
    if not (test_dir / "conftest.py").exists():
        issues.append("Missing conftest.py")

    # Check test files
    test_files = list(test_dir.glob("test_*.py"))
    if len(test_files) == 0:
        issues.append("No test files found")

    # Validate test file structure
    for test_file in test_files:
        try:
            with open(test_file) as f:
                content = f.read()
                tree = ast.parse(content)

            # Check for test functions
            test_functions = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ]

            if len(test_functions) == 0:
                issues.append(f"{test_file.name}: No test functions found")
        except Exception as e:
            issues.append(f"{test_file.name}: {e}")

    return len(issues) == 0, issues


def validate_package_structure(src_dir: Path) -> Tuple[bool, List[str]]:
    """Validate package structure."""
    issues = []

    if not src_dir.exists():
        return False, ["Source directory does not exist"]

    # Check for __init__.py in package
    package_dirs = [d for d in src_dir.rglob("*") if d.is_dir() and not d.name.startswith(".")]

    for pkg_dir in package_dirs:
        if "__pycache__" in str(pkg_dir):
            continue

        # Check if it contains Python files
        py_files = list(pkg_dir.glob("*.py"))
        if py_files and not (pkg_dir / "__init__.py").exists():
            # Check if any .py file is not __init__
            non_init_files = [f for f in py_files if f.name != "__init__.py"]
            if non_init_files:
                issues.append(f"{pkg_dir}: Missing __init__.py in package")

    return len(issues) == 0, issues


def check_config_files() -> Tuple[bool, List[str]]:
    """Check configuration files."""
    issues = []
    root = Path.cwd()

    # Check pyproject.toml
    if not (root / "pyproject.toml").exists():
        issues.append("Missing pyproject.toml")
    else:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # Try parsing manually
                try:
                    with open(root / "pyproject.toml") as f:
                        content = f.read()
                        # Basic check
                        if "[project]" not in content:
                            issues.append("pyproject.toml: Missing [project] section")
                except Exception as e:
                    issues.append(f"pyproject.toml: {e}")

    # Check .pre-commit-config.yaml
    if not (root / ".pre-commit-config.yaml").exists():
        issues.append("Missing .pre-commit-config.yaml")
    else:
        try:
            import yaml
            with open(root / ".pre-commit-config.yaml") as f:
                yaml.safe_load(f)
        except ImportError:
            # Can't validate without PyYAML
            pass
        except Exception as e:
            issues.append(f".pre-commit-config.yaml: {e}")

    return len(issues) == 0, issues


def main():
    """Run all validations."""
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Production Validation Script{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    root = Path.cwd()
    all_passed = True

    # 1. Validate Python syntax
    print(f"{YELLOW}[1/5] Validating Python syntax...{RESET}")
    src_ok, src_errors = validate_python_syntax(root / "src")
    test_ok, test_errors = validate_python_syntax(root / "tests")

    if src_ok and test_ok:
        print(f"{GREEN}✓ All Python files have valid syntax{RESET}")
    else:
        print(f"{RED}✗ Syntax errors found:{RESET}")
        for error in src_errors + test_errors:
            print(f"  {RED}- {error}{RESET}")
        all_passed = False
    print()

    # 2. Check imports
    print(f"{YELLOW}[2/5] Checking imports...{RESET}")
    imports_ok, import_issues = check_imports(root / "src")

    if imports_ok:
        print(f"{GREEN}✓ Import structure is valid{RESET}")
    else:
        print(f"{RED}✗ Import issues found:{RESET}")
        for issue in import_issues:
            print(f"  {RED}- {issue}{RESET}")
        all_passed = False
    print()

    # 3. Validate package structure
    print(f"{YELLOW}[3/5] Validating package structure...{RESET}")
    pkg_ok, pkg_issues = validate_package_structure(root / "src")

    if pkg_ok:
        print(f"{GREEN}✓ Package structure is valid{RESET}")
    else:
        print(f"{RED}✗ Package issues found:{RESET}")
        for issue in pkg_issues:
            print(f"  {RED}- {issue}{RESET}")
        all_passed = False
    print()

    # 4. Validate test structure
    print(f"{YELLOW}[4/5] Validating test structure...{RESET}")
    test_ok, test_issues = validate_test_structure(root / "tests")

    if test_ok:
        print(f"{GREEN}✓ Test structure is valid{RESET}")
    else:
        print(f"{RED}✗ Test issues found:{RESET}")
        for issue in test_issues:
            print(f"  {RED}- {issue}{RESET}")
        all_passed = False
    print()

    # 5. Check configuration files
    print(f"{YELLOW}[5/5] Checking configuration files...{RESET}")
    config_ok, config_issues = check_config_files()

    if config_ok:
        print(f"{GREEN}✓ Configuration files are valid{RESET}")
    else:
        print(f"{RED}✗ Configuration issues found:{RESET}")
        for issue in config_issues:
            print(f"  {RED}- {issue}{RESET}")
        all_passed = False
    print()

    # Summary
    print(f"{BLUE}{'='*70}{RESET}")
    if all_passed:
        print(f"{GREEN}✓ ALL VALIDATIONS PASSED{RESET}")
        print(f"{GREEN}Repository is ready for production!{RESET}")
        return 0
    else:
        print(f"{RED}✗ SOME VALIDATIONS FAILED{RESET}")
        print(f"{RED}Please fix the issues above before deploying.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
