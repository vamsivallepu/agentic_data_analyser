"""
UI components for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import base64
from typing import Dict, Any, List
from utils.logger import get_ui_logger
from config.prompts import UI_MESSAGES, ERROR_MESSAGES

logger = get_ui_logger()


def setup_page_config():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Multi-CSV Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_custom_css():
    """Load custom CSS for better styling."""
    st.markdown(
        """
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .result-box {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff7f0e;
        }
        /* Prevent automatic scrolling glitches */
        * {
            overflow-anchor: none !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">Multi-CSV Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Upload multiple CSV files and ask questions about any of them!")


def render_sidebar(uploaded_files: List) -> str:
    """
    Render the sidebar with file upload and configuration.

    Args:
        uploaded_files: List of uploaded files

    Returns:
        API key
    """
    with st.sidebar:
        # API Key input
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Enter your OpenAI API key",
            value=st.session_state.get("api_key", ""),
        )

        if api_key:
            st.session_state.api_key = api_key

        # File upload
        st.header("Data Upload")
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help=UI_MESSAGES["upload_help"],
        )

        # Show uploaded files info
        if uploaded_files:
            st.subheader("Uploaded Files:")
            for i, file in enumerate(uploaded_files):
                st.write(f"📄 {file.name} ({file.size} bytes)")

    return api_key


def render_datasets_overview(agent, datasets_info: Dict[str, Any]):
    """
    Render the datasets overview section.

    Args:
        agent: The analysis agent
        datasets_info: Information about all datasets
    """
    st.header("Datasets Overview")

    for dataset_name, info in datasets_info.items():
        with st.expander(f"📋 {dataset_name}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", info["shape"][0])
                st.metric("Columns", info["shape"][1])

            with col2:
                st.metric("Numeric Columns", len(info["numeric_columns"]))
                st.metric("Categorical Columns", len(info["categorical_columns"]))

            # Show column information
            st.subheader("Column Information:")
            col_info_df = pd.DataFrame(
                [
                    {
                        "Column": col,
                        "Data Type": str(dtype),
                        "Missing Values": info["missing_values"][col],
                    }
                    for col, dtype in info["data_types"].items()
                ]
            )
            st.dataframe(col_info_df, use_container_width=True)


def render_query_section(agent) -> str:
    """
    Render the query input section.

    Args:
        agent: The analysis agent

    Returns:
        User question
    """
    st.header("Ask Questions About Your Data")

    # Query input
    user_question = st.text_input(
        "Enter your question:",
        value=st.session_state.get("user_question", ""),
        placeholder=UI_MESSAGES["question_placeholder"],
        help=UI_MESSAGES["question_help"],
    )

    # Clear example question after use
    if "user_question" in st.session_state:
        del st.session_state.user_question

    return user_question


def render_analysis_results(result: Dict[str, Any]):
    """
    Render the analysis results.

    Args:
        result: Analysis results dictionary
    """
    if result["success"]:
        # Show which dataset was analyzed
        if result.get("dataset_analyzed"):
            st.info(f"📊 Analyzed dataset: **{result['dataset_analyzed']}**")

        # Show function call details
        st.markdown("---")
        if result["functions_called"]:
            st.subheader("Function Executed")
            for function, parameters in zip(
                result["functions_called"], result["parameters"]
            ):
                st.write(f"**Function:** `{function}`")
                st.write(f"**Parameters:** `{parameters}`")

        # Show LLM response
        st.markdown("---")
        st.subheader("AI Response")
        st.write(result["llm_response"])
        st.markdown("---")

        # Show results
        if result["result"]:
            st.subheader("Analysis Results")
            render_result_content(result["result"])

        # Show additional plot if available (avoid duplicates)
        if result.get("plot_file") and result["plot_file"] != result.get("result"):
            render_plot(result["plot_file"])

    else:
        st.error(f"Analysis failed: {result['message']}")


def render_result_content(result: Any):
    """
    Render the result content based on its type.

    Args:
        result: The result to render
    """
    # Display dictionary results
    if isinstance(result, dict):
        display_dict = {}
        for key, value in result.items():
            if isinstance(value, str) and value.startswith("data:image/"):
                # Skip inline plots in Analysis Results
                continue
            display_dict[key] = value
        st.json(display_dict)
    elif isinstance(result, str):
        # Check if it's a base64 image
        if result.startswith("data:image/"):
            render_plot(result)
        else:
            st.write(result)
    else:
        st.write(result)


def render_plot(plot_data: str):
    """
    Render a plot from base64 data.

    Args:
        plot_data: Base64 encoded plot data
    """
    st.subheader("Generated Plot")
    try:
        # Extract base64 data
        if "," in plot_data:
            base64_content = plot_data.split(",", 1)[1]
        else:
            base64_content = plot_data

        # Validate base64 content
        if not base64_content or len(base64_content) < 10:
            st.error(ERROR_MESSAGES["invalid_base64"])
            return

        # Decode and display using st.image
        image_data = base64.b64decode(base64_content)
        if len(image_data) == 0:
            st.error(ERROR_MESSAGES["empty_image_data"])
            return

        st.image(image_data, use_column_width=True)

        # Download button
        st.download_button(
            label="Download Plot",
            data=image_data,
            file_name="plot.png",
            mime="image/png",
            key=f"download_plot_{hash(plot_data[:50])}",
        )
    except Exception as e:
        logger.error(f"Error displaying plot: {str(e)}")
        st.error(ERROR_MESSAGES["plot_display_error"].format(error=str(e)))


def render_footer():
    """Render the footer."""
    st.markdown("---")


def handle_file_upload(uploaded_files: List, agent) -> List[str]:
    """
    Handle file upload and processing.

    Args:
        uploaded_files: List of uploaded files
        agent: The analysis agent

    Returns:
        List of temporary file paths
    """
    import tempfile

    temp_files = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            temp_files.append(tmp_file_path)

        # Load CSV into agent
        load_result = agent.load_csv(tmp_file_path, uploaded_file.name)
        st.success(f"{uploaded_file.name}: {load_result}", icon="✅")

    return temp_files


def cleanup_temp_files(temp_files: List[str]):
    """
    Clean up temporary files.

    Args:
        temp_files: List of temporary file paths
    """
    import os

    for tmp_file in temp_files:
        try:
            os.unlink(tmp_file)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {tmp_file}: {str(e)}")


def show_error_message(message: str):
    """
    Show an error message.

    Args:
        message: Error message to display
    """
    st.error(message)
    logger.error(f"UI Error: {message}")


def show_success_message(message: str):
    """
    Show a success message.

    Args:
        message: Success message to display
    """
    st.success(message)
    logger.info(f"UI Success: {message}")


def show_warning_message(message: str):
    """
    Show a warning message.

    Args:
        message: Warning message to display
    """
    st.warning(message)
    logger.warning(f"UI Warning: {message}")
