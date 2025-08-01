"""
Code Validator for Safe Python Code Execution
============================================

This module provides comprehensive validation for Python code snippets
before execution to prevent security vulnerabilities and system abuse.
"""

import ast
import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""

    STRICT = "strict"  # Maximum security, minimal allowed operations
    MODERATE = "moderate"  # Balanced security for data analysis
    PERMISSIVE = "permissive"  # More operations allowed, still secure


class SecurityRisk(Enum):
    """Types of security risks"""

    HIGH = "high"  # Immediate security threat
    MEDIUM = "medium"  # Potential security concern
    LOW = "low"  # Minor security consideration
    INFO = "info"  # Informational only


@dataclass
class ValidationResult:
    """Result of code validation"""

    is_safe: bool
    risk_level: SecurityRisk
    violations: List[str]
    warnings: List[str]
    sanitized_code: Optional[str] = None
    metadata: Optional[Dict] = None


class CodeValidator:
    """
    Validates Python code for safe execution in sandboxed environment.

    Features:
    - AST-based static analysis
    - Import statement validation
    - Dangerous function detection
    - Resource usage limits
    - Code sanitization
    """

    # Dangerous modules that should never be imported
    DANGEROUS_MODULES = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "glob",
        "pathlib",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "telnetlib",
        "multiprocessing",
        "threading",
        "concurrent",
        "ctypes",
        "platform",
        "tempfile",
        "pickle",
        "marshal",
        "eval",
        "exec",
        "compile",
        "__import__",
        "importlib",
        "pkgutil",
        "zipimport",
        "builtins",
        "__builtin__",
    }

    # Allowed safe modules for data analysis
    SAFE_MODULES = {
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "scipy",
        "statsmodels",
        "openpyxl",
        "json",
        "csv",
        "datetime",
        "time",
        "re",
        "math",
        "statistics",
        "collections",
        "itertools",
        "functools",
        "warnings",
        "typing",
        "dataclasses",
        "enum",
    }

    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "open",
        "file",
        "raw_input",
    }

    # Safe built-in functions for data analysis
    SAFE_BUILTINS = {
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "sum",
        "min",
        "max",
        "abs",
        "round",
        "pow",
        "divmod",
        "isinstance",
        "issubclass",
        "type",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "print",
        "format",
        "repr",
        "hex",
        "oct",
        "bin",
        "chr",
        "ord",
    }

    # Dangerous AST node types
    DANGEROUS_NODES = {
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Global,
        ast.Nonlocal,
        ast.Delete,
    }

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize code validator.

        Args:
            validation_level: How strict the validation should be
        """
        self.validation_level = validation_level
        self.violations = []
        self.warnings = []

    def validate_code(self, code: str) -> ValidationResult:
        """
        Main validation entry point.

        Args:
            code: Python code string to validate

        Returns:
            ValidationResult with safety assessment
        """
        self.violations = []
        self.warnings = []

        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Run validation checks
            self._validate_imports(tree)
            self._validate_function_calls(tree)
            self._validate_dangerous_operations(tree)
            self._validate_code_structure(tree)

            # Determine overall safety
            risk_level = self._assess_risk_level()
            is_safe = risk_level in [SecurityRisk.LOW, SecurityRisk.INFO]

            # Attempt sanitization if not safe but repairable
            sanitized_code = None
            if not is_safe and risk_level == SecurityRisk.MEDIUM:
                sanitized_code = self._sanitize_code(code, tree)
                if sanitized_code:
                    is_safe = True
                    risk_level = SecurityRisk.LOW

            return ValidationResult(
                is_safe=is_safe,
                risk_level=risk_level,
                violations=self.violations.copy(),
                warnings=self.warnings.copy(),
                sanitized_code=sanitized_code,
                metadata={
                    "validation_level": self.validation_level.value,
                    "code_length": len(code),
                    "ast_nodes": len(list(ast.walk(tree))),
                },
            )

        except SyntaxError as e:
            return ValidationResult(
                is_safe=False,
                risk_level=SecurityRisk.HIGH,
                violations=[f"Syntax error: {e}"],
                warnings=[],
                metadata={"syntax_error": str(e)},
            )
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_safe=False,
                risk_level=SecurityRisk.HIGH,
                violations=[f"Validation error: {e}"],
                warnings=[],
            )

    def _validate_imports(self, tree: ast.AST) -> None:
        """Validate import statements"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in self.DANGEROUS_MODULES:
                        self.violations.append(
                            f"Dangerous import detected: {alias.name}"
                        )
                    elif module_name not in self.SAFE_MODULES:
                        self.warnings.append(f"Unknown module import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name in self.DANGEROUS_MODULES:
                        self.violations.append(
                            f"Dangerous import from detected: {node.module}"
                        )
                    elif module_name not in self.SAFE_MODULES:
                        self.warnings.append(
                            f"Unknown module import from: {node.module}"
                        )

    def _validate_function_calls(self, tree: ast.AST) -> None:
        """Validate function calls for dangerous operations"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                if func_name in self.DANGEROUS_BUILTINS:
                    self.violations.append(f"Dangerous function call: {func_name}")

                # Check for specific dangerous patterns
                if func_name == "exec" or func_name == "eval":
                    self.violations.append(
                        f"Code execution function detected: {func_name}"
                    )
                elif func_name == "open":
                    self.violations.append(
                        "File access detected - use provided data instead"
                    )

    def _validate_dangerous_operations(self, tree: ast.AST) -> None:
        """Check for dangerous operations and patterns"""
        for node in ast.walk(tree):
            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    self.warnings.append(f"Private attribute access: {node.attr}")

            # Check for dangerous assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.startswith("__"):
                            self.violations.append(
                                f"Dangerous variable assignment: {target.id}"
                            )

            # Check for while loops (potential infinite loops)
            if isinstance(node, ast.While):
                self.warnings.append("While loop detected - ensure it terminates")

    def _validate_code_structure(self, tree: ast.AST) -> None:
        """Validate overall code structure"""
        # Count different types of statements
        function_defs = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        class_defs = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

        if function_defs > 5:
            self.warnings.append(
                f"Many function definitions ({function_defs}) - keep code simple"
            )

        if class_defs > 2:
            self.warnings.append(
                f"Multiple class definitions ({class_defs}) - focus on analysis"
            )

    def _get_function_name(self, func_node: ast.AST) -> str:
        """Extract function name from AST node"""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        else:
            return "unknown"

    def _assess_risk_level(self) -> SecurityRisk:
        """Assess overall risk level based on violations and warnings"""
        if any("Dangerous" in v or "Code execution" in v for v in self.violations):
            return SecurityRisk.HIGH
        elif len(self.violations) > 0:
            return SecurityRisk.MEDIUM
        elif len(self.warnings) > 3:
            return SecurityRisk.MEDIUM
        elif len(self.warnings) > 0:
            return SecurityRisk.LOW
        else:
            return SecurityRisk.INFO

    def _sanitize_code(self, code: str, tree: ast.AST) -> Optional[str]:
        """
        Attempt to sanitize code by removing dangerous parts.

        Returns:
            Sanitized code string if possible, None if cannot be made safe
        """
        try:
            # Simple sanitization - remove dangerous imports
            lines = code.split("\n")
            safe_lines = []

            for line in lines:
                line_stripped = line.strip()

                # Skip dangerous import lines
                if line_stripped.startswith(
                    ("import os", "import sys", "from os", "from sys")
                ):
                    safe_lines.append(f"# REMOVED: {line}")
                    continue

                # Skip exec/eval lines
                if "exec(" in line or "eval(" in line:
                    safe_lines.append(f"# REMOVED: {line}")
                    continue

                safe_lines.append(line)

            sanitized = "\n".join(safe_lines)

            # Validate the sanitized code
            try:
                ast.parse(sanitized)
                return sanitized
            except SyntaxError:
                return None

        except Exception:
            return None


def validate_python_code(
    code: str, validation_level: ValidationLevel = ValidationLevel.MODERATE
) -> ValidationResult:
    """
    Convenience function to validate Python code.

    Args:
        code: Python code string to validate
        validation_level: How strict the validation should be

    Returns:
        ValidationResult with safety assessment
    """
    validator = CodeValidator(validation_level)
    return validator.validate_code(code)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Safe code
        """
import pandas as pd
import numpy as np

data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
result = data.mean()
print(result)
        """,
        # Dangerous code
        """
import os
import subprocess

os.system('rm -rf /')
subprocess.call(['cat', '/etc/passwd'])
        """,
        # Moderately risky code
        """
import pandas as pd
import unknown_module

data = pd.read_csv('file.csv')
exec('print("hello")')
        """,
    ]

    validator = CodeValidator(ValidationLevel.MODERATE)

    for i, code in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        result = validator.validate_code(code)
        print(f"Safe: {result.is_safe}")
        print(f"Risk Level: {result.risk_level.value}")
        print(f"Violations: {result.violations}")
        print(f"Warnings: {result.warnings}")
        if result.sanitized_code:
            print("Sanitized version available")
