"""
Core CSV analyzer class with basic analysis functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.logger import get_analyzer_logger
from utils.data_utils import (
    make_json_safe,
    get_column_types,
)
from config.constants import ANALYSIS_LIMITS

logger = get_analyzer_logger()


class CSVAnalyzer:
    """
    Core CSV analyzer class for basic data analysis operations.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the CSV analyzer with a DataFrame.

        Args:
            df: Pandas DataFrame to analyze
        """
        self.df = df
        column_types = get_column_types(df)
        self.numeric_columns = column_types["numeric_columns"]
        self.categorical_columns = column_types["categorical_columns"]
        self.datetime_columns = column_types["datetime_columns"]

        logger.info(
            f"Initialized analyzer with {len(df)} rows and {len(df.columns)} columns"
        )
        logger.debug(f"Numeric columns: {self.numeric_columns}")
        logger.debug(f"Categorical columns: {self.categorical_columns}")
        logger.debug(f"Datetime columns: {self.datetime_columns}")

    def get_basic_summary(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.

        Returns:
            Dictionary with basic dataset information
        """
        logger.info("Generating basic summary")

        # Convert dtypes to strings explicitly
        data_types_str = {
            col: str(dtype) for col, dtype in self.df.dtypes.to_dict().items()
        }

        info = {
            "shape": list(self.df.shape),
            "columns": self.df.columns.tolist(),
            "data_types": data_types_str,
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
        }

        return make_json_safe(info)

    def get_numeric_info(self) -> Dict[str, Any]:
        """
        Get summary statistics for numeric columns.

        Returns:
            Dictionary with numeric column statistics
        """
        if len(self.numeric_columns) == 0:
            logger.warning("No numeric columns found in the dataset")
            return "No numeric columns found in the dataset."

        logger.info("Generating numeric column information")
        summary = self.df[self.numeric_columns].describe()
        return make_json_safe(summary.to_dict())

    def get_categorical_info(self) -> Dict[str, Any]:
        """
        Get summary for categorical columns.

        Returns:
            Dictionary with categorical column statistics
        """
        if len(self.categorical_columns) == 0:
            logger.warning("No categorical columns found in the dataset")
            return "No categorical columns found in the dataset."

        logger.info("Generating categorical column information")
        summary = {}
        for col in self.categorical_columns:
            summary[col] = {
                "unique_values": self.df[col].nunique(),
                "top_values": self.df[col].value_counts().head(10).to_dict(),
                "missing_count": int(self.df[col].isnull().sum()),
            }
        return make_json_safe(summary)

    def get_column_analysis(self, column: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis of a specific column.

        Args:
            column: Column name to analyze

        Returns:
            Dictionary with comprehensive column analysis
        """
        if column not in self.df.columns:
            logger.error(f"Column '{column}' not found in dataset")
            return f"Column '{column}' not found in dataset."

        logger.info(f"Analyzing column: {column}")

        # Basic column information
        column_info = {
            "column_name": column,
            "data_type": str(self.df[column].dtype),
            "total_rows": len(self.df),
            "non_null_count": int(self.df[column].count()),
            "missing_count": int(self.df[column].isnull().sum()),
            "missing_percentage": round(
                (self.df[column].isnull().sum() / len(self.df)) * 100, 2
            ),
            "unique_values": int(self.df[column].nunique()),
            "unique_percentage": round(
                (self.df[column].nunique() / len(self.df)) * 100, 2
            ),
        }

        # Type-specific analysis
        if column in self.numeric_columns:
            column_info.update(self._analyze_numeric_column(column))
        elif column in self.categorical_columns:
            column_info.update(self._analyze_categorical_column(column))
        elif column in self.datetime_columns:
            column_info.update(self._analyze_datetime_column(column))
        else:
            column_info["column_type"] = "other"
            column_info["note"] = "Column type not specifically categorized"

        # Common analysis for all columns
        column_info["sample_values"] = self.df[column].dropna().head(10).tolist()
        column_info["memory_usage_bytes"] = int(self.df[column].memory_usage(deep=True))

        return make_json_safe(column_info)

    def _analyze_numeric_column(self, column: str) -> Dict[str, Any]:
        """Analyze a numeric column."""
        numeric_data = self.df[column].dropna()
        if len(numeric_data) == 0:
            return {"column_type": "numeric", "note": "No non-null data available"}

        return {
            "column_type": "numeric",
            "mean": float(numeric_data.mean()),
            "median": float(numeric_data.median()),
            "std": float(numeric_data.std()),
            "min": float(numeric_data.min()),
            "max": float(numeric_data.max()),
            "range": float(numeric_data.max() - numeric_data.min()),
            "skewness": float(numeric_data.skew()),
            "kurtosis": float(numeric_data.kurtosis()),
            "quartiles": {
                "Q1": float(numeric_data.quantile(0.25)),
                "Q2": float(numeric_data.quantile(0.50)),
                "Q3": float(numeric_data.quantile(0.75)),
                "IQR": float(numeric_data.quantile(0.75) - numeric_data.quantile(0.25)),
            },
            "outliers_iqr": self._get_outlier_info(numeric_data, "iqr"),
            "outliers_zscore": self._get_outlier_info(numeric_data, "zscore"),
            "zero_count": int((numeric_data == 0).sum()),
            "negative_count": int((numeric_data < 0).sum()),
            "positive_count": int((numeric_data > 0).sum()),
            "distribution": {
                "bins_10": self._get_histogram_bins(numeric_data, 10),
                "bins_20": self._get_histogram_bins(numeric_data, 20),
                "percentiles": {
                    str(p): float(numeric_data.quantile(p / 100))
                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                },
            },
        }

    def _analyze_categorical_column(self, column: str) -> Dict[str, Any]:
        """Analyze a categorical column."""
        categorical_data = self.df[column].dropna()
        if len(categorical_data) == 0:
            return {"column_type": "categorical", "note": "No non-null data available"}

        value_counts = categorical_data.value_counts()

        return {
            "column_type": "categorical",
            "mode": (
                str(categorical_data.mode().iloc[0])
                if len(categorical_data.mode()) > 0
                else None
            ),
            "mode_count": (int(value_counts.iloc[0]) if len(value_counts) > 0 else 0),
            "mode_percentage": (
                round((value_counts.iloc[0] / len(categorical_data)) * 100, 2)
                if len(value_counts) > 0
                else 0
            ),
            "top_10_values": value_counts.head(10).to_dict(),
            "bottom_10_values": value_counts.tail(10).to_dict(),
            "value_length_stats": (
                self._get_string_length_stats(categorical_data)
                if categorical_data.dtype == "object"
                else None
            ),
            "all_values": (
                value_counts.to_dict()
                if len(value_counts)
                <= ANALYSIS_LIMITS["max_categorical_values_display"]
                else None
            ),
            "cardinality_note": (
                f"High cardinality column with {len(value_counts)} unique values"
                if len(value_counts) > ANALYSIS_LIMITS["max_categorical_values_display"]
                else None
            ),
        }

    def _analyze_datetime_column(self, column: str) -> Dict[str, Any]:
        """Analyze a datetime column."""
        datetime_data = self.df[column].dropna()
        if len(datetime_data) == 0:
            return {"column_type": "datetime", "note": "No non-null data available"}

        return {
            "column_type": "datetime",
            "earliest_date": str(datetime_data.min()),
            "latest_date": str(datetime_data.max()),
            "date_range_days": (datetime_data.max() - datetime_data.min()).days,
            "year_range": {
                "start_year": int(datetime_data.dt.year.min()),
                "end_year": int(datetime_data.dt.year.max()),
            },
            "month_distribution": datetime_data.dt.month.value_counts()
            .sort_index()
            .to_dict(),
            "day_of_week_distribution": datetime_data.dt.dayofweek.value_counts()
            .sort_index()
            .to_dict(),
            "hour_distribution": (
                datetime_data.dt.hour.value_counts().sort_index().to_dict()
                if hasattr(datetime_data.dt, "hour")
                else None
            ),
        }

    def _get_outlier_info(
        self, data: pd.Series, method: str
    ) -> Optional[Dict[str, Any]]:
        """Helper method to get outlier information."""
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > 3]
        else:
            return None

        return {
            "count": int(len(outliers)),
            "percentage": round((len(outliers) / len(data)) * 100, 2),
            "values": (
                outliers.tolist()
                if len(outliers) <= ANALYSIS_LIMITS["max_outlier_values_display"]
                else outliers.head(
                    ANALYSIS_LIMITS["max_outlier_values_display"]
                ).tolist()
            ),
        }

    def _get_histogram_bins(
        self, data: pd.Series, bins: int
    ) -> Optional[Dict[str, Any]]:
        """Helper method to get histogram bin information."""
        try:
            hist, bin_edges = np.histogram(data, bins=bins)
            return {
                "bin_edges": bin_edges.tolist(),
                "bin_counts": hist.tolist(),
                "bin_centers": [
                    (bin_edges[i] + bin_edges[i + 1]) / 2
                    for i in range(len(bin_edges) - 1)
                ],
            }
        except Exception as e:
            logger.error(f"Error calculating histogram bins: {str(e)}")
            return None

    def _get_string_length_stats(self, data: pd.Series) -> Optional[Dict[str, Any]]:
        """Helper method to get string length statistics for categorical columns."""
        try:
            string_lengths = data.astype(str).str.len()
            return {
                "min_length": int(string_lengths.min()),
                "max_length": int(string_lengths.max()),
                "mean_length": float(string_lengths.mean()),
                "median_length": float(string_lengths.median()),
                "std_length": float(string_lengths.std()),
            }
        except Exception as e:
            logger.error(f"Error calculating string length stats: {str(e)}")
            return None
