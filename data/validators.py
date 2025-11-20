"""
Data validation utilities for checking dataset integrity and quality.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

class DataValidator:
    """
    Validates datasets for completeness, quality, and consistency.
    """

    def __init__(
        self,
        strict_mode: bool = True,
        logger: Optional[logging.Logger] = None,
        verbose: bool = True,
        imbalance_threshold: float = 0.95,
        missing_threshold: float = 0.2
    ):
        """
        Initialize validator.

        Args:
            strict_mode: If True, raise errors on critical violations.
            logger: Optional logger instance.
            verbose: If True, print warnings to console.
            imbalance_threshold: Threshold to flag class imbalance (0-1).
            missing_threshold: Threshold to flag columns with too many missing values.
        """
        self.strict_mode = strict_mode
        self.logger = logger
        self.verbose = verbose
        self.imbalance_threshold = imbalance_threshold
        self.missing_threshold = missing_threshold
        self.issues: List[Dict] = []

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        label_column: str = 'Label'
    ) -> bool:
        """
        Validate a pandas DataFrame.

        Returns:
            True if valid according to strict_mode, False otherwise.
        """
        self.issues.clear()

        if df.empty:
            self._add_issue("DataFrame is empty", critical=True)
            return self._handle_validation_result()

        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                self._add_issue(f"Missing required columns: {missing_cols}", critical=True)

        # Check label column
        if label_column not in df.columns:
            self._add_issue(f"Label column '{label_column}' not found", critical=True)

        # Check missing values
        missing_summary = self._check_missing_values(df)
        for col, pct in missing_summary['missing_percent'].items():
            if pct > self.missing_threshold * 100:
                self._add_issue(
                    f"Column '{col}' has {pct:.2f}% missing values (> {self.missing_threshold*100:.0f}%)",
                    critical=False
                )
        if missing_summary['has_missing']:
            self._add_issue(
                f"Missing values found: {missing_summary['missing_counts']}",
                critical=False
            )

        # Check duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            self._add_issue(
                f"Found {n_duplicates} duplicate rows ({n_duplicates/len(df)*100:.2f}%)",
                critical=False
            )

        # Check label distribution
        if label_column in df.columns:
            label_dist = self._check_label_distribution(df, label_column)
            if label_dist['imbalance_ratio'] > self.imbalance_threshold:
                self._add_issue(
                    f"Class imbalance detected: {label_dist['distribution']}. "
                    f"Imbalance ratio: {label_dist['imbalance_ratio']:.2f}",
                    critical=False
                )

        return self._handle_validation_result()

    def _handle_validation_result(self) -> bool:
        critical_issues = [i for i in self.issues if i['critical']]
        has_critical = len(critical_issues) > 0

        if self.strict_mode and has_critical:
            raise ValueError(f"Critical validation issues: {critical_issues}")

        return not has_critical

    def _add_issue(self, message: str, critical: bool = False):
        entry = {'message': message, 'critical': critical}
        self.issues.append(entry)

        if self.logger:
            log_method = self.logger.error if critical else self.logger.warning
            log_method(f"Validation {'CRITICAL' if critical else 'WARNING'}: {message}")
        elif self.verbose:
            print(f"{'CRITICAL' if critical else 'WARNING'}: {message}")

    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100
        return {
            'has_missing': missing_counts.sum() > 0,
            'missing_counts': missing_counts[missing_counts > 0].to_dict(),
            'missing_percent': missing_percent[missing_percent > 0].to_dict(),
        }

    def _check_label_distribution(self, df: pd.DataFrame, label_column: str) -> Dict:
        if df.empty or label_column not in df.columns:
            return {'distribution': {}, 'imbalance_ratio': 0.0, 'class_counts': {}}

        counts = df[label_column].value_counts()
        total = len(df)
        distribution = {label: count / total for label, count in counts.items()}

        if len(counts) < 2:
            imbalance_ratio = 0.0
        else:
            max_count = counts.max()
            min_count = counts.min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        return {
            'distribution': distribution,
            'imbalance_ratio': imbalance_ratio,
            'class_counts': counts.to_dict()
        }

    def get_validation_report(self) -> Dict:
        """
        Returns:
            Dict containing valid flag, issues, counts, and timestamp.
        """
        critical_issues = [i for i in self.issues if i['critical']]
        warning_issues = [i for i in self.issues if not i['critical']]

        return {
            'valid': len(critical_issues) == 0,
            'issues': self.issues,
            'critical_issues': critical_issues,
            'warning_issues': warning_issues,
            'critical_count': len(critical_issues),
            'warning_count': len(warning_issues),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
