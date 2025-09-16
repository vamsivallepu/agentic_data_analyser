"""
Configuration file containing all prompts and system instructions used in the application.
"""

# System instructions for different components
SYSTEM_INSTRUCTIONS = {
    "csv_analysis": (
        "You are a CSV data analysis assistant. "
        "Analyze the provided dataset and decide which analysis function to call based on the user's question. "
        "After receiving function responses, compose a clear, accurate answer for the user using those results."
    ),
    "dataset_selector": (
        "You are a dataset selector. Based on the user's question, select the most relevant dataset. "
        "Respond with only the dataset name, nothing else."
    ),
}

# User prompts and messages
USER_PROMPTS = {
    "dataset_selection": "User question: {query}\n\nWhich dataset should be used?",
    "analysis_request": "User question: {query}",
}

# Error messages
ERROR_MESSAGES = {
    "no_csv_loaded": "No CSV files loaded. Please upload CSV files first.",
    "no_dataset_selected": "No dataset selected. Please select a dataset first.",
    "unknown_function": "Unknown function: {function_name}",
    "insufficient_data": "Insufficient data for analysis.",
    "invalid_column": "Column '{column}' is not found or not suitable for this analysis.",
    "api_key_missing": "Missing API key. Set API_KEY or pass api_key.",
    "file_upload_error": "Error loading CSV files: {error}",
    "analysis_error": "Error during analysis: {error}",
    "plot_display_error": "Error displaying plot: {error}",
    "invalid_base64": "Invalid or empty base64 data",
    "empty_image_data": "Empty image data",
}

# Success messages
SUCCESS_MESSAGES = {
    "csv_loaded": "Successfully loaded CSV file with {rows} rows and {columns} columns.",
    "dataset_selected": "Selected dataset: {dataset_name}",
    "analysis_complete": "Analysis completed successfully.",
}

# UI messages
UI_MESSAGES = {
    "upload_help": "Upload multiple CSV files for analysis",
    "question_placeholder": "e.g., 'Show me a histogram of the age column from the users dataset'",
    "question_help": "Ask questions about your data in natural language. You can specify which dataset to analyze.",
    "analyzing": "Analyzing your question...",
    "please_enter_question": "Please enter a question to analyze.",
}
