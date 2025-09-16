"""
Utility functions for data processing and conversion.
"""

import pandas as pd
import numpy as np
import base64
import io
from typing import Any, Dict, List, Union
from .logger import get_analyzer_logger

logger = get_analyzer_logger()


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert numpy/pandas objects to JSON-serializable Python types.

    Args:
        value: Value to convert

    Returns:
        JSON-serializable value
    """
    try:
        # Handle pandas NA/NaT and numpy NaN
        if pd.isna(value):
            return None
    except Exception:
        pass

    # Numpy scalar types
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if np.isnan(v):
            return None
        return v
    if isinstance(value, (np.bool_,)):
        return bool(value)

    # Pandas timestamp/timedelta and numpy datetime
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if isinstance(value, (np.datetime64,)):
        try:
            return pd.to_datetime(value).isoformat()
        except Exception:
            return str(value)

    # Containers
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    # Pandas objects
    if isinstance(value, (pd.Series,)):
        return make_json_safe(value.to_dict())
    if isinstance(value, (pd.DataFrame,)):
        # Return list of records for dataframes by default
        return make_json_safe(value.to_dict(orient="records"))

    # Fallback for other numpy types
    if isinstance(value, (np.ndarray,)):
        return make_json_safe(value.tolist())

    return value


def figure_to_base64(fig, format: str = "png", dpi: int = 100) -> str:
    """
    Convert matplotlib figure to base64-encoded string.

    Args:
        fig: Matplotlib figure object
        format: Image format (png, jpg, etc.)
        dpi: DPI for the image

    Returns:
        Base64 encoded image string
    """
    try:
        # Save figure to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        # Close the figure to free memory
        import matplotlib.pyplot as plt

        plt.close(fig)

        return f"data:image/{format};base64,{img_base64}"
    except Exception as e:
        import matplotlib.pyplot as plt

        plt.close(fig)
        logger.error(f"Error converting figure to base64: {str(e)}")
        return f"Error converting figure to base64: {str(e)}"


def validate_column(column: str, df: pd.DataFrame, expected_type: str = None) -> bool:
    """
    Validate if a column exists and is of the expected type.

    Args:
        column: Column name to validate
        df: DataFrame to check
        expected_type: Expected column type ('numeric', 'categorical', 'datetime')

    Returns:
        True if column is valid, False otherwise
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataset")
        return False

    if expected_type:
        if expected_type == "numeric":
            if column not in df.select_dtypes(include=[np.number]).columns:
                logger.warning(f"Column '{column}' is not numeric")
                return False
        elif expected_type == "categorical":
            if column not in df.select_dtypes(include=["object"]).columns:
                logger.warning(f"Column '{column}' is not categorical")
                return False
        elif expected_type == "datetime":
            if column not in df.select_dtypes(include=["datetime64"]).columns:
                logger.warning(f"Column '{column}' is not datetime")
                return False

    return True


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get column types for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with column types
    """
    return {
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=["datetime64"]).columns.tolist(),
    }


def safe_numeric_conversion(value: Any) -> Union[float, int, None]:
    """
    Safely convert a value to numeric type.

    Args:
        value: Value to convert

    Returns:
        Converted numeric value or None if conversion fails
    """
    try:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return value
        return float(value)
    except (ValueError, TypeError):
        return None


def truncate_string(value: str, max_length: int = 100) -> str:
    """
    Truncate a string if it exceeds maximum length.

    Args:
        value: String to truncate
        max_length: Maximum allowed length

    Returns:
        Truncated string
    """
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def format_large_number(value: Union[int, float]) -> str:
    """
    Format large numbers with K, M, B suffixes.

    Args:
        value: Number to format

    Returns:
        Formatted string
    """
    if value >= 1e9:
        return f"{value/1e9:.1f}B"
    elif value >= 1e6:
        return f"{value/1e6:.1f}M"
    elif value >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return str(value)
