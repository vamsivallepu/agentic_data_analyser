"""
Constants and function definitions used throughout the application.
"""

# Available analysis functions and their descriptions
AVAILABLE_FUNCTIONS = {
    "select_dataset": {
        "description": "Select which dataset to analyze based on the user's question",
        "parameters": {"dataset_name": "Name of the dataset to select"},
        "required": ["dataset_name"],
        "returns": "Confirmation of dataset selection",
    },
    "get_basic_summary": {
        "description": "Get summary about the dataset including shape, columns, data types, and missing values",
        "parameters": {},
        "required": [],
        "returns": "Dataset summary dictionary",
    },
    "get_column_analysis": {
        "description": "Get comprehensive information about a specific column including statistics, distribution, outliers, and other relevant information based on column type",
        "parameters": {"column": "Column name to analyze"},
        "required": ["column"],
        "returns": "Dictionary with comprehensive column analysis information",
    },
    "create_histogram": {
        "description": "Create a histogram for a numeric column to show data distribution",
        "parameters": {
            "column": "Column name",
            "bins": "Number of bins (default: 20)",
        },
        "required": ["column"],
        "returns": "Base64 encoded image string",
    },
    "create_boxplot": {
        "description": "Create a boxplot for a numeric column to show quartiles and outliers",
        "parameters": {"column": "Column name"},
        "required": ["column"],
        "returns": "Base64 encoded image string",
    },
    "create_scatter_plot": {
        "description": "Create a scatter plot between two numeric columns",
        "parameters": {
            "x_col": "X-axis column name",
            "y_col": "Y-axis column name",
        },
        "required": ["x_col", "y_col"],
        "returns": "Base64 encoded image string",
    },
    "create_correlation_heatmap": {
        "description": "Create a correlation heatmap for all numeric columns",
        "parameters": {},
        "required": [],
        "returns": "Base64 encoded image string",
    },
    "create_line_plot": {
        "description": "Create a line plot for time series or sequential data",
        "parameters": {
            "x_col": "X-axis column name",
            "y_col": "Y-axis column name",
        },
        "required": ["x_col", "y_col"],
        "returns": "Base64 encoded image string",
    },
    "create_bar_chart": {
        "description": "Create a bar chart for categorical column showing value counts",
        "parameters": {
            "column": "Column name",
            "top_n": "Number of top values to show (default: 10)",
        },
        "required": ["column"],
        "returns": "Base64 encoded image string",
    },
    "perform_pca_analysis": {
        "description": "Perform Principal Component Analysis on numeric columns",
        "parameters": {"n_components": "Number of components (default: 2)"},
        "required": [],
        "returns": "Dictionary with base64 figure and PCA results",
    },
    "perform_clustering": {
        "description": "Perform K-means clustering on numeric columns",
        "parameters": {"n_clusters": "Number of clusters (default: 3)"},
        "required": [],
        "returns": "Dictionary with base64 figure and clustering results",
    },
    "perform_linear_regression": {
        "description": "Perform linear regression between two numeric columns",
        "parameters": {
            "x_col": "Independent variable column",
            "y_col": "Dependent variable column",
        },
        "required": ["x_col", "y_col"],
        "returns": "Dictionary with base64 figure and regression statistics",
    },
    "get_outliers": {
        "description": "Detect outliers in a numeric column using IQR or Z-score method",
        "parameters": {
            "column": "Column name",
            "method": "Method: 'iqr' or 'zscore' (default: 'iqr')",
        },
        "required": ["column"],
        "returns": "Dictionary with outlier information",
    },
    "get_missing_data_analysis": {
        "description": "Analyze missing data patterns across all columns",
        "parameters": {},
        "required": [],
        "returns": "Dictionary with base64 figure and missing data statistics",
    },
    "get_data_distribution": {
        "description": "Get detailed distribution information for a specific column",
        "parameters": {"column": "Column name"},
        "required": ["column"],
        "returns": "Dictionary with distribution statistics",
    },
    "filter_data": {
        "description": "Filter data based on pandas query conditions",
        "parameters": {
            "conditions": "Pandas query string with 'and' and 'or' operators if required (e.g., 'age > 25 and city == \"NYC\"')"
        },
        "required": ["conditions"],
        "returns": "Dictionary with filtered data statistics",
    },
    "get_cross_tabulation": {
        "description": "Create cross-tabulation between two categorical columns",
        "parameters": {
            "col1": "First categorical column",
            "col2": "Second categorical column",
        },
        "required": ["col1", "col2"],
        "returns": "Dictionary with base64 figure and crosstab data",
    },
}

# Default values for analysis parameters
DEFAULT_PARAMS = {
    "histogram_bins": 20,
    "bar_chart_top_n": 10,
    "pca_components": 2,
    "clustering_clusters": 3,
    "outlier_method": "iqr",
    "figure_dpi": 100,
    "figure_format": "png",
}

# Analysis limits and constraints
ANALYSIS_LIMITS = {
    "max_histogram_bins": 100,
    "max_bar_chart_values": 50,
    "max_pca_components": 10,
    "max_clusters": 20,
    "max_outlier_values_display": 20,
    "max_categorical_values_display": 25,
    "max_sample_values": 10,
}

# File and data constraints
FILE_CONSTRAINTS = {
    "max_file_size_mb": 100,
    "supported_formats": ["csv"],
    "max_rows_preview": 10,
    "max_columns_preview": 5,
}

# LLM configuration
LLM_CONFIG = {
    "max_tokens_dataset_selection": 50,
    "temperature_dataset_selection": 0,
    "max_tool_calls": 5,
    "model_fallback": "gpt-3.5-turbo",
}
