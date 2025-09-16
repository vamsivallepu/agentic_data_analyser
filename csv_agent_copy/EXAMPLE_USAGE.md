# Example Usage

This file demonstrates how to use the reorganized Multi-CSV Agent application.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file:
   ```env
   GEMINI_API_KEY=your_openai_api_key_here
   MODEL=model_name
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Example Queries

### Basic Analysis
- "Show me a summary of the dataset"
- "What are the data types of each column?"
- "How many missing values are there?"

### Visualizations
- "Create a histogram of the age column"
- "Show me a boxplot of the price column"
- "Generate a correlation heatmap"
- "Create a bar chart of the category column"

### Statistical Analysis
- "Perform PCA analysis on numeric columns"
- "Run clustering analysis with 3 clusters"
- "Find outliers in the price column"
- "Perform linear regression between x and y"

### Data Exploration
- "Filter data where age > 25"
- "Show me the distribution of the salary column"
- "Analyze missing data patterns"
- "Create a cross-tabulation between gender and category"

## Code Structure

The application is now organized into logical modules:

- **`src/agents/`**: AI agent implementations
- **`src/analyzers/`**: Data analysis functionality
- **`src/config/`**: Configuration and constants
- **`src/ui/`**: Streamlit UI components
- **`src/utils/`**: Utility functions and logging

## Benefits of the New Structure

1. **Maintainability**: Code is organized into logical modules
2. **Logging**: Comprehensive logging for debugging
3. **Configuration**: Centralized configuration management
4. **Modularity**: Easy to extend and modify individual components
5. **Error Handling**: Better error handling and user feedback
