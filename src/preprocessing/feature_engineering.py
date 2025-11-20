"""
Research-grade feature engineering module.

Key features:
- Config-driven canonical feature mapping and transforms
- Safe, AST-based vectorized formula evaluation (no eval/exec)
- Centralized issue/diagnostic collection and logger integration
- Extensible transform registry and parameterized transforms
- Clear strict/flexible/ignore missing-feature modes
- Returns canonical DataFrame and a validation report
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import warnings

import numpy as np
import pandas as pd

# -------------------------
# Types
# -------------------------
Numeric = Union[int, float]
TransformCallable = Callable[[pd.Series], pd.Series]

@dataclass
class FeatureIssue:
    feature: str
    message: str
    critical: bool = False

@dataclass
class FeatureResult:
    series: pd.Series
    derived: bool = False
    used_mapping: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# Safe expression evaluator (vectorized)
# -------------------------
# We parse expressions into AST and allow only safe nodes:
# BinOp, UnaryOp, Name, Constant, Expr, Paren, and allowed ops (+, -, *, /, %, **).
# Names map to df[column_name]. If a name is not a df column, that's an error.


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def _safe_eval_series(expr: str, df: pd.DataFrame) -> pd.Series:
    """
    Safely evaluate a vectorized arithmetic expression over DataFrame columns.
    Supports binary ops, unary ops, numeric constants, parentheses, and column names.
    Raises ValueError for unsafe nodes or missing columns.
    """
    node = ast.parse(expr, mode="eval")

    def _eval(node) -> pd.Series:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_BINOPS:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            left = _eval(node.left)
            right = _eval(node.right)
            func = _ALLOWED_BINOPS[type(node.op)]
            # Pandas series operations are broadcasted
            return func(left, right)
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_UNARYOPS:
                raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")
            operand = _eval(node.operand)
            func = _ALLOWED_UNARYOPS[type(node.op)]
            return func(operand)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return pd.Series(node.value, index=df.index)
            raise ValueError("Only numeric constants are allowed in formulas")
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # Python < 3.8
            return pd.Series(node.n, index=df.index)
        if isinstance(node, ast.Name):
            col = node.id
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame for formula evaluation")
            return df[col]
        if isinstance(node, ast.Call):
            # disallow calls to functions for safety (could be extended carefully)
            raise ValueError("Function calls are not allowed in formula expressions")
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    result = _eval(node)
    # Ensure result is a Series (if constant number, it's broadcasted)
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    # fallback: wrap scalar into Series
    s = pd.Series(result, index=df.index)
    return s.replace([np.inf, -np.inf], np.nan)


# -------------------------
# Transform registry
# -------------------------
def _param_divide_by(param: str) -> TransformCallable:
    """Factory to create divide_by_X transform where param is numeric string"""
    divisor = float(param)
    if divisor == 0:
        raise ValueError("divide_by_0 is invalid")
    def _f(s: pd.Series) -> pd.Series:
        return s / divisor
    return _f

def _param_multiply_by(param: str) -> TransformCallable:
    factor = float(param)
    def _f(s: pd.Series) -> pd.Series:
        return s * factor
    return _f

_TRANSFORMS: Dict[str, TransformCallable] = {
    "identity": lambda s: s,
    "log": lambda s: pd.Series(np.log(s + 1e-8), index=s.index, name=s.name),
    "log1p": lambda s: pd.Series(np.log1p(s), index=s.index, name=s.name),
    "sqrt": lambda s: pd.Series(np.sqrt(s), index=s.index, name=s.name),
    "square": lambda s: s ** 2,
    "abs": lambda s: s.abs(),
    "to_numeric": lambda s: pd.to_numeric(s, errors="coerce"),
    # Treat 'sum' as a no-op; actual multi-column addition handled upstream
    "sum": lambda s: s,
    # Placeholders for higher-level semantic transforms defined in YAML config
    "binary_encode": lambda s: s,
    "categorical_encode": lambda s: s,
    "port_bucketing": lambda s: s,
    # parameterized transforms will be handled by name prefix rules below
}
def get_transform_callable(name: Optional[str]) -> TransformCallable:
    """
    Resolve a transform name to a callable.
    Supports base names in _TRANSFORMS and parameterized names:
      - divide_by_<number>
      - multiply_by_<number>
    If name is None or 'identity', returns identity.
    """
    if not name or name == "identity":
        return _TRANSFORMS["identity"]
    if name in _TRANSFORMS:
        return _TRANSFORMS[name]
    # parameterized patterns
    if name.startswith("divide_by_"):
        param = name.split("divide_by_")[-1]
        return _param_divide_by(param)
    if name.startswith("multiply_by_"):
        param = name.split("multiply_by_")[-1]
        return _param_multiply_by(param)
    # Unknown transform: raise
    raise ValueError(f"Unknown transform: {name}")


# -------------------------
# Main class
# -------------------------
class FeatureEngineer:
    """
    Transforms raw DataFrames into canonical feature schema based on a config.

    Config schema (example):
    {
        "canonical_features": {
            "critical": ["Flow Duration", "Total Fwd Packets"],
            "important": [...],
            "optional": [...],
        },
        "feature_mappings": {
            "cic_ids_2017": {
                "Flow Duration": {"source_column": "Flow Duration", "transform": "divide_by_1000000", "dtype": "float"},
                "Fwd_Packets_per_sec": {"formula": "Total Fwd Packets / Flow Duration", "transform": "identity"},
                "Total Length of Fwd Packets": {"source_column": "Total Length of Fwd Packets"},
                "SomeCombined": {"source_columns": ["A","B"], "transform": "to_numeric"}
            },
            "ton_iot": { ... }
        },
        "missing_features": {
            "mode": "flexible",  # 'strict'|'flexible'|'ignore'
            "imputation_value": 0.0
        }
    }
    """

    def __init__(self, config: Dict[str, Any], dataset_name: str, logger: Optional[logging.Logger] = None):
        self.config = config
        self.dataset_name = dataset_name
        self.logger = logger or logging.getLogger(__name__)
        self.issues: List[FeatureIssue] = []

        if "feature_mappings" not in config or dataset_name not in config["feature_mappings"]:
            raise ValueError(f"Feature mappings for dataset '{dataset_name}' not found in config")

        self.mapping = config["feature_mappings"][dataset_name]
        self.canonical_features = config.get("canonical_features", {})
        if not isinstance(self.canonical_features, dict):
            raise ValueError("canonical_features must be a dict with 'critical','important','optional' lists")

        # missing mode handling
        mf = config.get("missing_features", {})
        self.missing_mode: str = mf.get("mode", "strict")
        self.imputation_value: Any = mf.get("imputation_value", 0.0)
        # thresholds/other options could be added here

    # -------------------------
    # Public API
    # -------------------------
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transform raw dataframe -> canonical dataframe.
        Returns: (canonical_df, report_dict)
        report_dict contains issues and diagnostics.
        """
        self.issues.clear()
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # canonical order: critical, important, optional
        canon_order = []
        for k in ("critical", "important", "optional"):
            canon_order.extend(self.canonical_features.get(k, []))

        canonical_df = pd.DataFrame(index=df.index)
        diagnostics: Dict[str, Any] = {
            "derived_features": [],
            "missing_assigned": [],
            "transforms_applied": {},
            "nan_counts": {},
        }

        for feature_name in canon_order:
            try:
                # Merge raw df with already-built canonical features so formulas can reference
                # previously derived canonical feature names.
                merged_df = pd.concat([df, canonical_df], axis=1)
                fr = self._extract_feature(merged_df, feature_name)
                if fr is None:
                    # Missing/failed extraction
                    self._handle_missing(feature_name, canonical_df)
                    diagnostics["missing_assigned"].append(feature_name)
                    continue
                # fr is FeatureResult
                s = fr.series
                # apply dtype coercion if requested
                dtype = fr.used_mapping.get("dtype")
                if dtype:
                    try:
                        if dtype.startswith("int"):
                            s = s.astype(int)
                        elif dtype.startswith("float"):
                            s = s.astype(float)
                        # else: leave as is or extend types
                    except Exception as e:
                        self._record_issue(feature_name, f"dtype coercion failed: {e}", critical=False)
                # fillna if mapping requests
                fillna_val = fr.used_mapping.get("fillna", None)
                if fillna_val is not None:
                    s = s.fillna(fillna_val)
                canonical_df[feature_name] = s
                diagnostics["nan_counts"][feature_name] = int(s.isna().sum())
                if fr.derived:
                    diagnostics["derived_features"].append(feature_name)
                # record transform name if present
                if fr.used_mapping.get("transform"):
                    diagnostics["transforms_applied"][feature_name] = fr.used_mapping.get("transform")
            except Exception as e:
                # Any exception here: behave according to missing_mode
                self._record_issue(feature_name, f"Exception during extraction: {e}", critical=(self.missing_mode=="strict"))
                if self.missing_mode == "strict":
                    raise
                # flexible/ignore => assign imputation or NaN
                self._handle_missing(feature_name, canonical_df)
                diagnostics["missing_assigned"].append(feature_name)

        report = self.get_report()
        report["diagnostics"] = diagnostics
        return canonical_df, report

    # -------------------------
    # Extraction helpers
    # -------------------------
    def _extract_feature(self, df: pd.DataFrame, feature_name: str) -> Optional[FeatureResult]:
        """
        Return FeatureResult or None if not extractable (and missing_mode allows).
        """
        mapping = self.mapping.get(feature_name, {})
        used_mapping = dict(mapping)  # copy for diagnostics

        # Priority: formula -> source_column -> source_columns -> composite addition (deprecated)
        if "formula" in mapping and mapping.get("formula"):
            formula = mapping["formula"]
            try:
                series = _safe_eval_series(formula, df)
                # optional transform
                transform_name = mapping.get("transform")
                if transform_name:
                    fn = get_transform_callable(transform_name)
                    series = fn(series)
                # Clean infs
                series = series.replace([np.inf, -np.inf], np.nan)
                return FeatureResult(series=series, derived=True, used_mapping=used_mapping)
            except Exception as e:
                self._record_issue(feature_name, f"Derived formula error: {e}", critical=(self.missing_mode=="strict"))
                return None

        if "source_column" in mapping:
            src = mapping["source_column"]
            if src in df.columns:
                series = df[src]
                addl = mapping.get("additional_column")
                # Support simple additive combination of two columns (common in config) BEFORE transforms
                if addl and addl in df.columns:
                    try:
                        series = pd.to_numeric(series, errors="coerce") + pd.to_numeric(df[addl], errors="coerce")
                    except Exception as e:
                        self._record_issue(feature_name, f"Failed combining '{src}' + '{addl}': {e}", critical=False)
                # transform if present
                transform_name = mapping.get("transform")
                if transform_name:
                    fn = get_transform_callable(transform_name)
                    try:
                        series = fn(series)
                    except Exception as e:
                        self._record_issue(feature_name, f"Transform '{transform_name}' failed: {e}", critical=False)
                # Special case: binary_encode for labels (attack vs benign)
                if mapping.get("transform") == "binary_encode":
                    attack_labels = [a.lower() for a in mapping.get("attack_labels", [])]
                    def _bin(x: Any) -> int:
                        if x is None or (isinstance(x, float) and np.isnan(x)):
                            return 0
                        xl = str(x).lower()
                        return 0 if "benign" in xl else (1 if any(a in xl for a in attack_labels) else 1)
                    try:
                        series = series.apply(_bin).astype(int)
                    except Exception as e:
                        self._record_issue(feature_name, f"binary_encode failed: {e}", critical=False)
                # Clean infs
                series = series.replace([np.inf, -np.inf], np.nan)
                return FeatureResult(series=series, derived=False, used_mapping=used_mapping)
            else:
                self._record_issue(feature_name, f"Source column '{src}' missing", critical=(self.missing_mode=="strict"))
                return None

        if "source_columns" in mapping and isinstance(mapping["source_columns"], (list, tuple)):
            cols = [c for c in mapping["source_columns"] if c in df.columns]
            if cols:
                series = df[cols].sum(axis=1, skipna=True)
                transform_name = mapping.get("transform")
                if transform_name:
                    fn = get_transform_callable(transform_name)
                    try:
                        series = fn(series)
                    except Exception as e:
                        self._record_issue(feature_name, f"Transform '{transform_name}' failed: {e}", critical=False)
                # Clean infs
                series = series.replace([np.inf, -np.inf], np.nan)
                return FeatureResult(series=series, derived=False, used_mapping=used_mapping)
            else:
                self._record_issue(feature_name, f"None of source_columns {mapping['source_columns']} present", critical=(self.missing_mode=="strict"))
                return None

        # No valid mapping found
        self._record_issue(feature_name, f"No valid mapping for feature", critical=(self.missing_mode=="strict"))
        return None

    # -------------------------
    # Missing feature handling
    # -------------------------
    def _handle_missing(self, feature_name: str, canonical_df: pd.DataFrame) -> None:
        if self.missing_mode == "strict":
            raise ValueError(f"Required canonical feature '{feature_name}' missing in strict mode")
        elif self.missing_mode == "flexible":
            canonical_df[feature_name] = pd.Series(self.imputation_value, index=canonical_df.index)
            self._record_issue(feature_name, f"Assigned imputation value {self.imputation_value}", critical=False)
        else:  # ignore
            canonical_df[feature_name] = pd.Series([np.nan] * len(canonical_df.index), index=canonical_df.index)
            self._record_issue(feature_name, "Assigned NaN (ignore mode)", critical=False)

    # -------------------------
    # Issue handling & report
    # -------------------------
    def _record_issue(self, feature: str, message: str, critical: bool = False) -> None:
        self.issues.append(FeatureIssue(feature=feature, message=message, critical=critical))
        if self.logger:
            if critical:
                self.logger.error(f"[FeatureEngineer] CRITICAL: {feature}: {message}")
            else:
                self.logger.warning(f"[FeatureEngineer] WARNING: {feature}: {message}")
        else:
            # non-logger fallback: warnings for non-critical, raise for critical if strict
            if critical:
                # avoid immediate raising here; record only — caller will raise if needed
                warnings.warn(f"CRITICAL {feature}: {message}")
            else:
                warnings.warn(f"WARNING {feature}: {message}")

    def get_report(self) -> Dict[str, Any]:
        """
        Return a dict report of validation/feature issues.
        """
        critical = [i for i in self.issues if i.critical]
        warnings_ = [i for i in self.issues if not i.critical]
        return {
            "valid": len(critical) == 0,
            "issues": [{"feature": i.feature, "message": i.message, "critical": i.critical} for i in self.issues],
            "critical_count": len(critical),
            "warning_count": len(warnings_),
        }

    # -------------------------
    # Utility: apply feature selection or drop fully-NaN columns
    # -------------------------
    def drop_all_nan_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are entirely NaN (useful after ignore mode runs)."""
        return df.dropna(axis=1, how="all")

