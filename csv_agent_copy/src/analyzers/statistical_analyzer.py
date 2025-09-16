"""
Statistical analysis analyzer for advanced statistical operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Optional
from utils.logger import get_analyzer_logger
from utils.data_utils import make_json_safe, figure_to_base64, validate_column
from config.constants import DEFAULT_PARAMS, ANALYSIS_LIMITS

logger = get_analyzer_logger()


class StatisticalAnalyzer:
    """
    Statistical analyzer for advanced statistical operations.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the statistical analyzer with a DataFrame.

        Args:
            df: Pandas DataFrame to analyze
        """
        self.df = df
        column_types = self._get_column_types()
        self.numeric_columns = column_types["numeric_columns"]
        self.categorical_columns = column_types["categorical_columns"]
        self.datetime_columns = column_types["datetime_columns"]

        logger.info("Initialized statistical analyzer")

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

    def perform_pca_analysis(
        self, n_components: int = None, figsize: tuple = (10, 8)
    ) -> Dict[str, Any]:
        """
        Perform PCA analysis on numeric columns.

        Args:
            n_components: Number of components
            figsize: Figure size

        Returns:
            Dictionary with PCA results and base64 figure
        """
        if len(self.numeric_columns) < 2:
            logger.warning("Need at least 2 numeric columns for PCA analysis")
            return "Need at least 2 numeric columns for PCA analysis."

        n_components = n_components or DEFAULT_PARAMS["pca_components"]
        n_components = min(n_components, ANALYSIS_LIMITS["max_pca_components"])

        logger.info(f"Performing PCA analysis with {n_components} components")

        try:
            # Prepare data
            numeric_data = self.df[self.numeric_columns].dropna()
            if len(numeric_data) == 0:
                logger.warning("No complete numeric data available for PCA")
                return "No complete numeric data available for PCA."

            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)

            # Perform PCA
            pca = PCA(n_components=min(n_components, len(self.numeric_columns)))
            pca_result = pca.fit_transform(scaled_data)

            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            if n_components >= 2:
                ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            else:
                ax.plot(pca_result, alpha=0.6)
                ax.set_xlabel("Sample")
                ax.set_ylabel("PC1")

            ax.set_title("PCA Analysis")
            ax.grid(True, alpha=0.3)

            return make_json_safe(
                {
                    "figure": figure_to_base64(fig),
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "components": pca.components_.tolist(),
                }
            )
        except Exception as e:
            logger.error(f"Error performing PCA analysis: {str(e)}")
            return f"Error performing PCA analysis: {str(e)}"

    def perform_clustering(
        self, n_clusters: int = None, figsize: tuple = (10, 8)
    ) -> Dict[str, Any]:
        """
        Perform K-means clustering on numeric columns.

        Args:
            n_clusters: Number of clusters
            figsize: Figure size

        Returns:
            Dictionary with clustering results and base64 figure
        """
        if len(self.numeric_columns) < 2:
            logger.warning("Need at least 2 numeric columns for clustering")
            return "Need at least 2 numeric columns for clustering."

        n_clusters = n_clusters or DEFAULT_PARAMS["clustering_clusters"]
        n_clusters = min(n_clusters, ANALYSIS_LIMITS["max_clusters"])

        logger.info(f"Performing K-means clustering with {n_clusters} clusters")

        try:
            # Prepare data
            numeric_data = self.df[self.numeric_columns].dropna()
            if len(numeric_data) == 0:
                logger.warning("No complete numeric data available for clustering")
                return "No complete numeric data available for clustering."

            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)

            # Perform clustering
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(numeric_data)), random_state=42
            )
            clusters = kmeans.fit_predict(scaled_data)

            # Create plot (using first two dimensions)
            fig, ax = plt.subplots(figsize=figsize)
            scatter = ax.scatter(
                scaled_data[:, 0],
                scaled_data[:, 1],
                c=clusters,
                cmap="viridis",
                alpha=0.6,
            )
            plt.colorbar(scatter, ax=ax)
            ax.set_xlabel(f"{self.numeric_columns[0]} (standardized)")
            ax.set_ylabel(f"{self.numeric_columns[1]} (standardized)")
            ax.set_title(f"K-means Clustering (k={n_clusters})")
            ax.grid(True, alpha=0.3)

            return make_json_safe(
                {
                    "figure": figure_to_base64(fig),
                    "cluster_labels": clusters.tolist(),
                    "cluster_centers": kmeans.cluster_centers_.tolist(),
                }
            )
        except Exception as e:
            logger.error(f"Error performing clustering: {str(e)}")
            return f"Error performing clustering: {str(e)}"

    def perform_linear_regression(self, x_col: str, y_col: str) -> Dict[str, Any]:
        """
        Perform linear regression between two numeric columns.

        Args:
            x_col: Independent variable column
            y_col: Dependent variable column

        Returns:
            Dictionary with regression results and base64 figure
        """
        if not validate_column(x_col, self.df, "numeric") or not validate_column(
            y_col, self.df, "numeric"
        ):
            return "Both columns must be numeric."

        logger.info(f"Performing linear regression: {y_col} vs {x_col}")

        try:
            # Prepare data
            data = self.df[[x_col, y_col]].dropna()
            if len(data) < 2:
                logger.warning("Insufficient data for regression analysis")
                return "Insufficient data for regression analysis."

            X = data[x_col].values.reshape(-1, 1)
            y = data[y_col].values

            # Perform regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X, y, alpha=0.6, label="Actual Data")
            ax.plot(
                X,
                y_pred,
                color="red",
                linewidth=2,
                label=f"Regression Line (R²={r2:.3f})",
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Linear Regression: {y_col} vs {x_col}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            return make_json_safe(
                {
                    "figure": figure_to_base64(fig),
                    "slope": model.coef_[0],
                    "intercept": model.intercept_,
                    "r_squared": r2,
                    "mean_squared_error": mse,
                }
            )
        except Exception as e:
            logger.error(f"Error performing linear regression: {str(e)}")
            return f"Error performing linear regression: {str(e)}"

    def get_outliers(self, column: str, method: str = None) -> Dict[str, Any]:
        """
        Detect outliers in a numeric column.

        Args:
            column: Column name
            method: Method for outlier detection ('iqr' or 'zscore')

        Returns:
            Dictionary with outlier information
        """
        if not validate_column(column, self.df, "numeric"):
            return f"Column '{column}' is not numeric."

        method = method or DEFAULT_PARAMS["outlier_method"]

        logger.info(f"Detecting outliers in column {column} using {method} method")

        try:
            data = self.df[column].dropna()
            if len(data) == 0:
                logger.warning("No data available for outlier detection")
                return "No data available for outlier detection."

            outlier_info = self._get_outlier_info(data, method)
            if outlier_info is None:
                logger.error(f"Invalid outlier detection method: {method}")
                return "Method must be 'iqr' or 'zscore'."

            return make_json_safe(
                {
                    "outlier_count": outlier_info["count"],
                    "outlier_values": outlier_info["values"],
                    "outlier_percentage": outlier_info["percentage"],
                }
            )
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return f"Error detecting outliers: {str(e)}"

    def get_missing_data_analysis(self) -> Dict[str, Any]:
        """
        Analyze missing data patterns.

        Returns:
            Dictionary with missing data analysis and base64 figure
        """
        logger.info("Analyzing missing data patterns")

        try:
            missing_data = self.df.isnull().sum()
            missing_percentage = (missing_data / len(self.df)) * 100

            # Create missing data plot
            fig, ax = plt.subplots(figsize=(12, 6))
            missing_percentage.plot(kind="bar", ax=ax)
            ax.set_title("Missing Data Percentage by Column")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Missing Data (%)")
            ax.tick_params(axis="x", rotation=45, ha="right")
            fig.tight_layout()

            return make_json_safe(
                {
                    "figure": figure_to_base64(fig),
                    "missing_counts": missing_data.to_dict(),
                    "missing_percentages": missing_percentage.to_dict(),
                }
            )
        except Exception as e:
            logger.error(f"Error analyzing missing data: {str(e)}")
            return f"Error analyzing missing data: {str(e)}"

    def get_data_distribution(self, column: str) -> Dict[str, Any]:
        """
        Get detailed distribution information for a column.

        Args:
            column: Column name

        Returns:
            Dictionary with distribution statistics
        """
        if column not in self.df.columns:
            logger.error(f"Column '{column}' not found in dataset")
            return f"Column '{column}' not found in dataset."

        logger.info(f"Getting distribution information for column: {column}")

        try:
            if column in self.numeric_columns:
                data = self.df[column].dropna()
                return make_json_safe(
                    {
                        "mean": data.mean(),
                        "median": data.median(),
                        "std": data.std(),
                        "min": data.min(),
                        "max": data.max(),
                        "skewness": data.skew(),
                        "kurtosis": data.kurtosis(),
                        "quartiles": data.quantile([0.25, 0.5, 0.75]).to_dict(),
                    }
                )
            elif column in self.categorical_columns:
                value_counts = self.df[column].value_counts()
                return make_json_safe(
                    {
                        "unique_count": len(value_counts),
                        "most_common": value_counts.head(5).to_dict(),
                        "least_common": value_counts.tail(5).to_dict(),
                    }
                )
            else:
                return f"Column '{column}' is neither numeric nor categorical."
        except Exception as e:
            logger.error(f"Error getting distribution information: {str(e)}")
            return f"Error getting distribution information: {str(e)}"

    def filter_data(self, conditions: str) -> Dict[str, Any]:
        """
        Filter data based on conditions and return statistics of filtered data.

        Args:
            conditions: Pandas query string

        Returns:
            Dictionary with filtered data statistics
        """
        logger.info(f"Filtering data with conditions: {conditions}")

        try:
            filtered_df = self.df.query(conditions)
            filtered_count = len(filtered_df)
            original_count = len(self.df)

            # Get column statistics for filtered data
            column_stats = {}
            for col in filtered_df.columns:
                col_info = {
                    "data_type": str(filtered_df[col].dtype),
                    "total_rows": filtered_count,
                    "non_null_count": int(filtered_df[col].count()),
                    "missing_count": int(filtered_df[col].isnull().sum()),
                    "missing_percentage": (
                        round(
                            (filtered_df[col].isnull().sum() / filtered_count) * 100, 2
                        )
                        if filtered_count > 0
                        else 0
                    ),
                    "unique_values": int(filtered_df[col].nunique()),
                }

                # Numeric column statistics
                if filtered_df[col].dtype in ["int64", "float64"]:
                    numeric_data = filtered_df[col].dropna()
                    if len(numeric_data) > 0:
                        col_info.update(
                            {
                                "mean": float(numeric_data.mean()),
                                "median": float(numeric_data.median()),
                                "std": float(numeric_data.std()),
                                "min": float(numeric_data.min()),
                                "max": float(numeric_data.max()),
                                "range": float(numeric_data.max() - numeric_data.min()),
                                "quartiles": {
                                    "Q1": float(numeric_data.quantile(0.25)),
                                    "Q2": float(numeric_data.quantile(0.50)),
                                    "Q3": float(numeric_data.quantile(0.75)),
                                },
                            }
                        )

                # Categorical column statistics
                elif filtered_df[col].dtype == "object":
                    categorical_data = filtered_df[col].dropna()
                    if len(categorical_data) > 0:
                        value_counts = categorical_data.value_counts()
                        col_info.update(
                            {
                                "all_values": (
                                    value_counts.to_dict()
                                    if len(value_counts)
                                    <= ANALYSIS_LIMITS["max_categorical_values_display"]
                                    else value_counts.head(
                                        ANALYSIS_LIMITS[
                                            "max_categorical_values_display"
                                        ]
                                    ).to_dict()
                                ),
                                "most_common": (
                                    value_counts.head(5).to_dict()
                                    if len(value_counts) > 0
                                    else {}
                                ),
                                "least_common": (
                                    value_counts.tail(5).to_dict()
                                    if len(value_counts) > 0
                                    else {}
                                ),
                            }
                        )

                # Datetime column statistics
                elif "datetime" in str(filtered_df[col].dtype).lower():
                    datetime_data = filtered_df[col].dropna()
                    if len(datetime_data) > 0:
                        col_info.update(
                            {
                                "earliest_date": str(datetime_data.min()),
                                "latest_date": str(datetime_data.max()),
                                "date_range_days": (
                                    datetime_data.max() - datetime_data.min()
                                ).days,
                            }
                        )

                column_stats[col] = col_info

            return make_json_safe(
                {
                    "filtered_count": filtered_count,
                    "original_count": original_count,
                    "filter_efficiency": (
                        round((filtered_count / original_count) * 100, 2)
                        if original_count > 0
                        else 0
                    ),
                    "column_statistics": column_stats,
                }
            )
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            return f"Error filtering data: {str(e)}"

    def get_cross_tabulation(self, col1: str, col2: str) -> Dict[str, Any]:
        """
        Create cross-tabulation between two categorical columns.

        Args:
            col1: First categorical column
            col2: Second categorical column

        Returns:
            Dictionary with cross-tabulation data and base64 figure
        """
        if not validate_column(col1, self.df, "categorical") or not validate_column(
            col2, self.df, "categorical"
        ):
            return "Both columns must be categorical."

        logger.info(f"Creating cross-tabulation: {col1} vs {col2}")

        try:
            crosstab = pd.crosstab(self.df[col1], self.df[col2])

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Cross-tabulation: {col1} vs {col2}")

            return make_json_safe(
                {
                    "figure": figure_to_base64(fig),
                    "crosstab_data": crosstab.to_dict(),
                }
            )
        except Exception as e:
            logger.error(f"Error creating cross-tabulation: {str(e)}")
            return f"Error creating cross-tabulation: {str(e)}"

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
