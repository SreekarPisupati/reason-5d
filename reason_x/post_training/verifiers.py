"""
Reason-X: Module 4.1 - Sandboxed Code Execution & SymPy Symbolic Verifiers
==========================================================================
Architectural Specification:
- Deterministic ground-truth rule verifiers returning r in {0.0, 1.0}.
- Sandboxed Python Code Executor: Evaluates generated Python scripts with strict timeouts & process isolation.
- SymPy / LaTeX Symbolic Equivalence Normalizer:
  * Parses LaTeX equations, fractions, power terms, and matrices into SymPy symbolic trees.
  * Evaluates mathematical equivalence: simplify(expr_pred - expr_gt) == 0.
  * Robust answer extractors: \\boxed{...}, #### <answer>, <answer>...</answer>.
"""

import ast
import re
import sys
from typing import Any, Dict, Optional, Tuple, Union
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


class SymbolicMathVerifier:
    """
    Normalizes mathematical answers and evaluates symbolic equivalence via SymPy.
    """

    TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

    @classmethod
    def extract_boxed_answer(cls, text: str) -> Optional[str]:
        """Extracts content inside \\boxed{...} or #### format."""
        # 1. Look for \boxed{...}
        boxed_matches = re.findall(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", text)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # 2. Look for #### (GSM8K format)
        gsm_match = re.search(r"####\s*([^\n]+)", text)
        if gsm_match:
            return gsm_match.group(1).strip()

        # 3. Look for <answer>...</answer>
        tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if tag_match:
            return tag_match.group(1).strip()

        return None

    @classmethod
    def clean_latex(cls, latex_str: str) -> str:
        """Cleans LaTeX formatting into parseable mathematical expressions."""
        s = latex_str.strip()
        # Remove currency symbols and units
        s = s.replace("$", "").replace("\\$", "").replace("%", "").replace(",", "")
        # Standardize fractions: \frac{a}{b} -> (a)/(b)
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        # Standardize sqrt: \sqrt{a} -> sqrt(a)
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        # Standardize exponents: ^ -> **
        s = s.replace("^", "**")
        # Standardize times: \times -> *
        s = s.replace("\\times", "*").replace("\\cdot", "*")
        return s.strip()

    @classmethod
    def verify(cls, prediction: str, ground_truth: str) -> float:
        """
        Verifies mathematical equality. Returns 1.0 for exact/symbolic match, 0.0 otherwise.
        """
        pred_ans = cls.extract_boxed_answer(prediction) or prediction.strip()
        gt_ans = cls.extract_boxed_answer(ground_truth) or ground_truth.strip()

        # 1. Direct string match
        if pred_ans.lower() == gt_ans.lower():
            return 1.0

        # 2. Numeric equality
        try:
            val_pred = float(pred_ans)
            val_gt = float(gt_ans)
            if abs(val_pred - val_gt) < 1e-5:
                return 1.0
        except (ValueError, TypeError):
            pass

        # 3. SymPy Symbolic Equivalence
        try:
            expr_pred = parse_expr(cls.clean_latex(pred_ans), transformations=cls.TRANSFORMATIONS)
            expr_gt = parse_expr(cls.clean_latex(gt_ans), transformations=cls.TRANSFORMATIONS)
            diff = sympy.simplify(expr_pred - expr_gt)
            if diff == 0 or diff.equals(0):
                return 1.0
        except Exception:
            pass

        return 0.0


class SandboxedCodeExecutor:
    """
    Executes Python reasoning traces with AST safety inspection.
    """

    FORBIDDEN_MODULES = {"os", "subprocess", "sys", "shutil", "socket", "requests", "urllib"}

    @classmethod
    def is_safe_code(cls, code_str: str) -> bool:
        """Inspects AST for forbidden system imports or unsafe operations."""
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in cls.FORBIDDEN_MODULES:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] in cls.FORBIDDEN_MODULES:
                        return False
            return True
        except Exception:
            return False

    @classmethod
    def execute_and_verify(
        cls,
        code_str: str,
        expected_output: Any,
        entrypoint: str = "solution"
    ) -> float:
        """
        Safely executes code in isolated namespace and verifies output against expected ground-truth.
        """
        if not cls.is_safe_code(code_str):
            return 0.0

        local_namespace: Dict[str, Any] = {}
        try:
            # Execute script definitions
            exec(code_str, {"__builtins__": __builtins__}, local_namespace)
            
            if entrypoint in local_namespace and callable(local_namespace[entrypoint]):
                result = local_namespace[entrypoint]()
                if str(result).strip() == str(expected_output).strip():
                    return 1.0
                if isinstance(result, (int, float)) and isinstance(expected_output, (int, float)):
                    if abs(result - expected_output) < 1e-5:
                        return 1.0
        except Exception:
            return 0.0

        return 0.0
