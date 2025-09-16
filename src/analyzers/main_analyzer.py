"""
Main CSV analyzer that combines all specialized analyzers.
"""

import pandas as pd
from typing import Dict, Any
from .core_analyzer import CSVAnalyzer
from .visualization_analyzer import VisualizationAnalyzer
from .statistical_analyzer import StatisticalAnalyzer
from utils.logger import get_analyzer_logger

logger = get_analyzer_logger()


class MainCSVAnalyzer:
    """
    Main CSV analyzer that combines core, visualization, and statistical analyzers.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the main CSV analyzer with a DataFrame.

        Args:
            df: Pandas DataFrame to analyze
        """
        self.df = df
        self.core_analyzer = CSVAnalyzer(df)
        self.visualization_analyzer = VisualizationAnalyzer(df)
        self.statistical_analyzer = StatisticalAnalyzer(df)

        # Inherit column types from core analyzer
        self.numeric_columns = self.core_analyzer.numeric_columns
        self.categorical_columns = self.core_analyzer.categorical_columns
        self.datetime_columns = self.core_analyzer.datetime_columns

        logger.info("Initialized main CSV analyzer")

    # Core analysis methods
    def get_basic_summary(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return self.core_analyzer.get_basic_summary()

    def get_numeric_info(self) -> Dict[str, Any]:
        """Get summary statistics for numeric columns."""
        return self.core_analyzer.get_numeric_info()

    def get_categorical_info(self) -> Dict[str, Any]:
        """Get summary for categorical columns."""
        return self.core_analyzer.get_categorical_info()

    def get_column_analysis(self, column: str) -> Dict[str, Any]:
        """Get comprehensive analysis of a specific column."""
        return self.core_analyzer.get_column_analysis(column)

    # Visualization methods
    def create_histogram(
        self, column: str, bins: int = None, figsize: tuple = (10, 6)
    ) -> str:
        """Create histogram for a numeric column."""
        return self.visualization_analyzer.create_histogram(column, bins, figsize)

    def create_boxplot(self, column: str, figsize: tuple = (10, 6)) -> str:
        """Create boxplot for a numeric column."""
        return self.visualization_analyzer.create_boxplot(column, figsize)

    def create_scatter_plot(
        self, x_col: str, y_col: str, figsize: tuple = (10, 6)
    ) -> str:
        """Create scatter plot between two numeric columns."""
        return self.visualization_analyzer.create_scatter_plot(x_col, y_col, figsize)

    def create_correlation_heatmap(self, figsize: tuple = (12, 10)) -> str:
        """Create correlation heatmap for numeric columns."""
        return self.visualization_analyzer.create_correlation_heatmap(figsize)

    def create_bar_chart(
        self, column: str, top_n: int = None, figsize: tuple = (12, 6)
    ) -> str:
        """Create bar chart for categorical column."""
        return self.visualization_analyzer.create_bar_chart(column, top_n, figsize)

    def create_line_plot(self, x_col: str, y_col: str, figsize: tuple = (12, 6)) -> str:
        """Create line plot for time series or sequential data."""
        return self.visualization_analyzer.create_line_plot(x_col, y_col, figsize)

    def create_missing_data_plot(self, figsize: tuple = (12, 6)) -> str:
        """Create missing data visualization."""
        return self.visualization_analyzer.create_missing_data_plot(figsize)

    def create_cross_tabulation_heatmap(
        self, col1: str, col2: str, figsize: tuple = (10, 8)
    ) -> str:
        """Create cross-tabulation heatmap between two categorical columns."""
        return self.visualization_analyzer.create_cross_tabulation_heatmap(
            col1, col2, figsize
        )

    # Statistical analysis methods
    def perform_pca_analysis(
        self, n_components: int = None, figsize: tuple = (10, 8)
    ) -> Dict[str, Any]:
        """Perform PCA analysis on numeric columns."""
        return self.statistical_analyzer.perform_pca_analysis(n_components, figsize)

    def perform_clustering(
        self, n_clusters: int = None, figsize: tuple = (10, 8)
    ) -> Dict[str, Any]:
        """Perform K-means clustering on numeric columns."""
        return self.statistical_analyzer.perform_clustering(n_clusters, figsize)

    def perform_linear_regression(self, x_col: str, y_col: str) -> Dict[str, Any]:
        """Perform linear regression between two numeric columns."""
        return self.statistical_analyzer.perform_linear_regression(x_col, y_col)

    def get_outliers(self, column: str, method: str = None) -> Dict[str, Any]:
        """Detect outliers in a numeric column."""
        return self.statistical_analyzer.get_outliers(column, method)

    def get_missing_data_analysis(self) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        return self.statistical_analyzer.get_missing_data_analysis()

    def get_data_distribution(self, column: str) -> Dict[str, Any]:
        """Get detailed distribution information for a column."""
        return self.statistical_analyzer.get_data_distribution(column)

    def filter_data(self, conditions: str) -> Dict[str, Any]:
        """Filter data based on conditions and return statistics of filtered data."""
        return self.statistical_analyzer.filter_data(conditions)

    def get_cross_tabulation(self, col1: str, col2: str) -> Dict[str, Any]:
        """Create cross-tabulation between two categorical columns."""
        return self.statistical_analyzer.get_cross_tabulation(col1, col2)

    # Additional utility methods
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        logger.info("Getting comprehensive dataset information")

        return {
            "basic_summary": self.get_basic_summary(),
            "numeric_info": self.get_numeric_info(),
            "categorical_info": self.get_categorical_info(),
            "missing_data_analysis": self.get_missing_data_analysis(),
            "column_types": {
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns,
                "datetime_columns": self.datetime_columns,
            },
        }

    def get_quick_analysis(self) -> Dict[str, Any]:
        """Get a quick analysis summary of the dataset."""
        logger.info("Performing quick analysis")

        basic_summary = self.get_basic_summary()

        # Add some quick visualizations if possible
        quick_analysis = {"summary": basic_summary, "suggestions": []}

        # Suggest analyses based on data types
        if len(self.numeric_columns) >= 2:
            quick_analysis["suggestions"].append(
                "Consider correlation analysis between numeric columns"
            )
            quick_analysis["suggestions"].append("Try PCA or clustering analysis")

        if len(self.numeric_columns) >= 1:
            quick_analysis["suggestions"].append(
                "Create histograms and boxplots for numeric columns"
            )

        if len(self.categorical_columns) >= 1:
            quick_analysis["suggestions"].append(
                "Create bar charts for categorical columns"
            )

        if len(self.categorical_columns) >= 2:
            quick_analysis["suggestions"].append("Try cross-tabulation analysis")

        # Check for missing data
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            quick_analysis["suggestions"].append("Analyze missing data patterns")

        return quick_analysis
