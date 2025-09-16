"""
Visualization analyzer for creating charts and plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from utils.logger import get_analyzer_logger
from utils.data_utils import figure_to_base64, validate_column
from config.constants import DEFAULT_PARAMS, ANALYSIS_LIMITS

logger = get_analyzer_logger()


class VisualizationAnalyzer:
    """
    Visualization analyzer for creating charts and plots.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualization analyzer with a DataFrame.

        Args:
            df: Pandas DataFrame to analyze
        """
        self.df = df
        column_types = self._get_column_types()
        self.numeric_columns = column_types["numeric_columns"]
        self.categorical_columns = column_types["categorical_columns"]
        self.datetime_columns = column_types["datetime_columns"]

        logger.info("Initialized visualization analyzer")

    def _get_column_types(self) -> Dict[str, list]:
        """Get column types for the DataFrame."""
        return {
            "numeric_columns": self.df.select_dtypes(
                include=[np.number]
            ).columns.tolist(),
            "categorical_columns": self.df.select_dtypes(
                include=["object"]
            ).columns.tolist(),
            "datetime_columns": self.df.select_dtypes(
                include=["datetime64"]
            ).columns.tolist(),
        }

    def create_histogram(
        self, column: str, bins: int = None, figsize: tuple = (10, 6)
    ) -> str:
        """
        Create histogram for a numeric column.

        Args:
            column: Column name
            bins: Number of bins (default: 20)
            figsize: Figure size

        Returns:
            Base64 encoded image string
        """
        if not validate_column(column, self.df, "numeric"):
            return f"Column '{column}' is not numeric."

        bins = bins or DEFAULT_PARAMS["histogram_bins"]
        bins = min(bins, ANALYSIS_LIMITS["max_histogram_bins"])

        logger.info(f"Creating histogram for column: {column} with {bins} bins")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(self.df[column].dropna(), bins=bins, edgecolor="black", alpha=0.7)
            ax.set_title(f"Histogram of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

            return figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return f"Error creating histogram: {str(e)}"

    def create_boxplot(self, column: str, figsize: tuple = (10, 6)) -> str:
        """
        Create boxplot for a numeric column.

        Args:
            column: Column name
            figsize: Figure size

        Returns:
            Base64 encoded image string
        """
        if not validate_column(column, self.df, "numeric"):
            return f"Column '{column}' is not numeric."

        logger.info(f"Creating boxplot for column: {column}")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.boxplot(self.df[column].dropna())
            ax.set_title(f"Boxplot of {column}")
            ax.set_ylabel(column)
            ax.grid(True, alpha=0.3)

            return figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating boxplot: {str(e)}")
            return f"Error creating boxplot: {str(e)}"

    def create_scatter_plot(
        self, x_col: str, y_col: str, figsize: tuple = (10, 6)
    ) -> str:
        """
        Create scatter plot between two numeric columns.

        Args:
            x_col: X-axis column name
            y_col: Y-axis column name
            figsize: Figure size

        Returns:
            Base64 encoded image string
        """
        if not validate_column(x_col, self.df, "numeric") or not validate_column(
            y_col, self.df, "numeric"
        ):
            return "Both columns must be numeric."

        logger.info(f"Creating scatter plot: {x_col} vs {y_col}")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6)
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(True, alpha=0.3)

            return figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return f"Error creating scatter plot: {str(e)}"

    def create_correlation_heatmap(self, figsize: tuple = (12, 10)) -> str:
        """
        Create correlation heatmap for numeric columns.

        Args:
            figsize: Figure size

        Returns:
            Base64 encoded image string
        """
        if len(self.numeric_columns) < 2:
            logger.warning("Need at least 2 numeric columns for correlation analysis")
            return "Need at least 2 numeric columns for correlation analysis."

        logger.info("Creating correlation heatmap")

        try:
            corr_matrix = self.df[self.numeric_columns].corr()

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title("Correlation Heatmap")

            return figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return f"Error creating correlation heatmap: {str(e)}"

    def create_bar_chart(
        self, column: str, top_n: int = None, figsize: tuple = (12, 6)
    ) -> str:
        """
        Create bar chart for categorical column.

        Args:
            column: Column name
            top_n: Number of top values to show
            figsize: Figure size

        Returns:
            Base64 encoded image string
        """
        if not validate_column(column, self.df, "categorical"):
            return f"Column '{column}' is not categorical."

        top_n = top_n or DEFAULT_PARAMS["bar_chart_top_n"]
        top_n = min(top_n, ANALYSIS_LIMITS["max_bar_chart_values"])

        logger.info(f"Creating bar chart for column: {column} with top {top_n} values")

        try:
            value_counts = self.df[column].value_counts().head(top_n)

            fig, ax = plt.subplots(figsize=figsize)
            value_counts.plot(kind="bar", ax=ax)
            ax.set_title(f"Top {top_n} Values in {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()

            return figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return f"Error creating bar chart: {str(e)}"

    def create_line_plot(self, x_col: str, y_col: str, figsize: tuple = (12, 6)) -> str:
        """
        Create line plot for time series or sequential data.

        Args:
            x_col: X-axis column name
            y_col: Y-axis column name
            figsize: Figure size

        Returns:
            Base64 encoded image string
        """
        if not validate_column(y_col, self.df, "numeric"):
            return f"Y column '{y_col}' must be numeric."

        logger.info(f"Creating line plot: {y_col} over {x_col}")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            if x_col in self.datetime_columns:
                # Sort by datetime
                sorted_df = self.df.sort_values(x_col)
                ax.plot(sorted_df[x_col], sorted_df[y_col])
            else:
                ax.plot(self.df[x_col], self.df[y_col])

            ax.set_title(f"Line Plot: {y_col} over {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()

            return figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating line plot: {str(e)}")
            return f"Error creating line plot: {str(e)}"

    def create_missing_data_plot(self, figsize: tuple = (12, 6)) -> str:
        """
        Create missing data visualization.

        Args:
            figsize: Figure size

        Returns:
            Base64 encoded image string
        """
        logger.info("Creating missing data plot")

        try:
            missing_data = self.df.isnull().sum()
            missing_percentage = (missing_data / len(self.df)) * 100

            fig, ax = plt.subplots(figsize=figsize)
            missing_percentage.plot(kind="bar", ax=ax)
            ax.set_title("Missing Data Percentage by Column")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Missing Data (%)")
            ax.tick_params(axis="x", rotation=45, ha="right")
            fig.tight_layout()

            return figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating missing data plot: {str(e)}")
            return f"Error creating missing data plot: {str(e)}"

    def create_cross_tabulation_heatmap(
        self, col1: str, col2: str, figsize: tuple = (10, 8)
    ) -> str:
        """
        Create cross-tabulation heatmap between two categorical columns.

        Args:
            col1: First categorical column
            col2: Second categorical column
            figsize: Figure size

        Returns:
            Base64 encoded image string
        """
        if not validate_column(col1, self.df, "categorical") or not validate_column(
            col2, self.df, "categorical"
        ):
            return "Both columns must be categorical."

        logger.info(f"Creating cross-tabulation heatmap: {col1} vs {col2}")

        try:
            crosstab = pd.crosstab(self.df[col1], self.df[col2])

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Cross-tabulation: {col1} vs {col2}")

            return figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error creating cross-tabulation heatmap: {str(e)}")
            return f"Error creating cross-tabulation heatmap: {str(e)}"
