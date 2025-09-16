"""
Main Streamlit application for the Multi-CSV Agent.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from agents.openai_agent import OpenAICSVAnalysisAgent
from ui.components import (
    setup_page_config,
    load_custom_css,
    render_header,
    render_datasets_overview,
    render_query_section,
    render_analysis_results,
    render_footer,
    handle_file_upload,
    cleanup_temp_files,
    show_error_message,
    show_warning_message,
)
from utils.logger import get_app_logger
from config.prompts import UI_MESSAGES, ERROR_MESSAGES

# Load .env file
load_dotenv()

# Setup logger
logger = get_app_logger()


def main():
    """Main application function."""
    try:
        # Setup page configuration
        setup_page_config()
        load_custom_css()

        # Render header
        render_header()

        # Sidebar for configuration and file upload
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help=UI_MESSAGES["upload_help"],
        )

        api_key = os.getenv("GEMINI_API_KEY")

        # Main content area
        if uploaded_files:
            try:
                # Initialize agent
                agent = OpenAICSVAnalysisAgent(api_key)

                # Handle file upload
                temp_files = handle_file_upload(uploaded_files, agent)

                # Store agent in session state
                st.session_state.agent = agent

                # Show datasets overview
                datasets_info = agent.get_all_datasets_info()
                render_datasets_overview(agent, datasets_info)

                # Clean up temp files after displaying info
                cleanup_temp_files(temp_files)

            except Exception as e:
                logger.error(f"Error loading CSV files: {str(e)}")
                show_error_message(
                    ERROR_MESSAGES["file_upload_error"].format(error=str(e))
                )
                return

        # Query input section
        if "agent" in st.session_state:
            user_question = render_query_section(st.session_state.agent)

            if st.button("Analyze", type="primary"):
                if user_question:
                    with st.spinner(UI_MESSAGES["analyzing"]):
                        try:
                            # Analyze the query
                            result = st.session_state.agent.analyze_query(user_question)
                            render_analysis_results(result)

                        except Exception as e:
                            logger.error(f"Error during analysis: {str(e)}")
                            show_error_message(
                                ERROR_MESSAGES["analysis_error"].format(error=str(e))
                            )
                else:
                    show_warning_message(UI_MESSAGES["please_enter_question"])

        # Footer
        render_footer()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
