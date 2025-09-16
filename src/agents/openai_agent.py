"""
OpenAI CSV analysis agent with tool/function calling support.
"""

import json
import os
from typing import Dict, Any, List, Optional
from openai import OpenAI
from analyzers.main_analyzer import MainCSVAnalyzer
from config.constants import AVAILABLE_FUNCTIONS, LLM_CONFIG
from config.prompts import (
    SYSTEM_INSTRUCTIONS,
    USER_PROMPTS,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
)
from utils.logger import get_agent_logger

logger = get_agent_logger()


class OpenAICSVAnalysisAgent:
    """
    CSV analysis agent that uses the OpenAI Python SDK with tool/function calling.
    Supports multiple CSV files.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the OpenAI CSV analysis agent.

        Args:
            api_key: OpenAI API key
            model: Model to use for analysis
            base_url: Custom base URL for OpenAI API
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("Missing API key")
            raise ValueError(ERROR_MESSAGES["api_key_missing"])

        self.model = model or os.getenv("MODEL")
        self.base_url = base_url or os.getenv("BASE_URL")

        # Initialize OpenAI client
        if self.base_url:
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        # Store multiple analyzers for different datasets
        self.analyzers: Dict[str, MainCSVAnalyzer] = {}
        self.dataset_names: List[str] = []
        self.current_dataset: Optional[str] = None

        self.available_functions = AVAILABLE_FUNCTIONS

        logger.info("Initialized OpenAI CSV analysis agent")

    def _get_openai_tools(self) -> List[Dict[str, Any]]:
        """Build OpenAI tools schema from available functions using JSON Schema."""
        tools: List[Dict[str, Any]] = []
        for name, meta in self.available_functions.items():
            params_meta = meta.get("parameters", {})
            properties: Dict[str, Any] = {}
            required: List[str] = []
            # Be permissive: use string-typed fields; casting handled in analyzer
            for param_name in params_meta.keys():
                properties[param_name] = {"type": "string"}
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": meta.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return tools

    def load_csv(self, file_path: str, dataset_name: str) -> str:
        """
        Load a CSV file and initialize the analyzer for a specific dataset.

        Args:
            file_path: Path to the CSV file
            dataset_name: Name for the dataset

        Returns:
            Success message
        """
        try:
            import pandas as pd

            logger.info(f"Loading CSV file: {file_path} as dataset: {dataset_name}")

            df = pd.read_csv(file_path)
            analyzer = MainCSVAnalyzer(df)

            # Store the analyzer with the dataset name
            self.analyzers[dataset_name] = analyzer
            self.dataset_names.append(dataset_name)

            # Set as current dataset if it's the first one
            if self.current_dataset is None:
                self.current_dataset = dataset_name

            logger.info(f"Successfully loaded dataset: {dataset_name}")
            return SUCCESS_MESSAGES["csv_loaded"].format(
                rows=len(df), columns=len(df.columns)
            )
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            return f"Error loading CSV file: {str(e)}"

    def select_dataset(self, dataset_name: str) -> str:
        """
        Select which dataset to analyze.

        Args:
            dataset_name: Name of the dataset to select

        Returns:
            Confirmation message
        """
        if dataset_name not in self.analyzers:
            available = ", ".join(self.dataset_names)
            logger.warning(
                f"Dataset '{dataset_name}' not found. Available: {available}"
            )
            return (
                f"Dataset '{dataset_name}' not found. Available datasets: {available}"
            )

        self.current_dataset = dataset_name
        logger.info(f"Selected dataset: {dataset_name}")
        return SUCCESS_MESSAGES["dataset_selected"].format(dataset_name=dataset_name)

    def _get_current_analyzer(self) -> Optional[MainCSVAnalyzer]:
        """Get the current analyzer for the selected dataset."""
        if self.current_dataset is None:
            return None
        return self.analyzers.get(self.current_dataset)

    def _execute_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute the specified function with given parameters.

        Args:
            function_name: Name of the function to execute
            parameters: Parameters for the function

        Returns:
            Function result
        """
        if function_name == "select_dataset":
            return self.select_dataset(parameters.get("dataset_name", ""))

        current_analyzer = self._get_current_analyzer()
        if not current_analyzer:
            logger.warning("No dataset selected")
            return ERROR_MESSAGES["no_dataset_selected"]

        if function_name not in self.available_functions:
            logger.error(f"Unknown function: {function_name}")
            return ERROR_MESSAGES["unknown_function"].format(
                function_name=function_name
            )

        try:
            logger.info(
                f"Executing function: {function_name} with parameters: {parameters}"
            )
            func = getattr(current_analyzer, function_name)
            if parameters:
                result = func(**parameters)
            else:
                result = func()
            return result
        except Exception as e:
            logger.error(f"Error executing {function_name}: {str(e)}")
            return f"Error executing {function_name}: {str(e)}"

    def _select_dataset_from_query(self, query: str) -> str:
        """
        Intelligently select the most appropriate dataset based on the query.

        Args:
            query: User query

        Returns:
            Selected dataset name
        """
        if len(self.dataset_names) == 1:
            return self.dataset_names[0]

        # Simple scoring system based on column names and dataset names
        best_dataset = self.dataset_names[0]
        best_score = 0

        query_lower = query.lower()

        for dataset_name in self.dataset_names:
            score = 0
            analyzer = self.analyzers[dataset_name]

            # Score based on dataset name matching
            if dataset_name.lower() in query_lower:
                score += 10

            # Score based on column names matching
            for col in analyzer.df.columns:
                if col.lower() in query_lower:
                    score += 5

            # Score based on data types mentioned
            if "numeric" in query_lower and len(analyzer.numeric_columns) > 0:
                score += 3
            if "categorical" in query_lower and len(analyzer.categorical_columns) > 0:
                score += 3

            if score > best_score:
                best_score = score
                best_dataset = dataset_name

        logger.info(f"Selected dataset '{best_dataset}' based on query analysis")
        return best_dataset

    def _select_dataset_with_llm(self, query: str) -> str:
        """
        Use a cheap LLM call to select the most relevant dataset based on the query.

        Args:
            query: User query

        Returns:
            Selected dataset name
        """
        if len(self.dataset_names) == 1:
            return self.dataset_names[0]

        # Create minimal context with only dataset names and key info
        datasets_summary = "Available datasets:\n"
        for dataset_name in self.dataset_names:
            analyzer = self.analyzers[dataset_name]
            # Only include basic info and a few key columns
            key_columns = list(analyzer.df.columns)[:5]  # Limit to first 5 columns
            if len(analyzer.df.columns) > 5:
                key_columns_str = (
                    f"{', '.join(key_columns)}... ({len(analyzer.df.columns)} total)"
                )
            else:
                key_columns_str = ", ".join(key_columns)

            datasets_summary += (
                f"- {dataset_name}: {analyzer.df.shape[0]} rows, {analyzer.df.shape[1]} columns, "
                f"key columns: {key_columns_str}\n"
            )

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS["dataset_selector"]},
            {"role": "user", "content": datasets_summary},
            {
                "role": "user",
                "content": USER_PROMPTS["dataset_selection"].format(query=query),
            },
        ]

        try:
            logger.info("Using LLM to select dataset")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=LLM_CONFIG["max_tokens_dataset_selection"],
                temperature=LLM_CONFIG["temperature_dataset_selection"],
            )

            selected = response.choices[0].message.content.strip()

            # Validate the selection
            if selected in self.dataset_names:
                logger.info(f"LLM selected dataset: {selected}")
                return selected
            else:
                # Fallback to the heuristic method if LLM response is invalid
                logger.warning(
                    f"LLM returned invalid dataset: {selected}, using fallback"
                )
                return self._select_dataset_from_query(query)

        except Exception as e:
            # Fallback to the heuristic method if LLM call fails
            logger.error(f"LLM dataset selection failed: {str(e)}, using fallback")
            return self._select_dataset_from_query(query)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a natural language query using OpenAI tool-calling and return results.

        Args:
            query: Natural language query

        Returns:
            Analysis results dictionary
        """
        if not self.analyzers:
            logger.warning("No CSV files loaded")
            return {
                "success": False,
                "message": ERROR_MESSAGES["no_csv_loaded"],
                "result": None,
                "plot_file": None,
            }

        logger.info(f"Analyzing query: {query}")

        # Step 1: Use cheap LLM call to select the most relevant dataset
        selected_dataset = self._select_dataset_with_llm(query)
        self.current_dataset = selected_dataset

        # Step 2: Use focused context with only the selected dataset
        current_analyzer = self._get_current_analyzer()
        if current_analyzer:
            dataset_context = (
                f"Dataset: {self.current_dataset}\n"
                f"- Shape: {current_analyzer.df.shape}\n"
                f"- Columns: {list(current_analyzer.df.columns)}\n"
                f"- Numeric columns: {current_analyzer.numeric_columns}\n"
                f"- Categorical columns: {current_analyzer.categorical_columns}\n"
                f"- Datetime columns: {current_analyzer.datetime_columns}"
            )
        else:
            dataset_context = "No dataset currently selected."

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS["csv_analysis"]},
            {"role": "user", "content": dataset_context},
            {
                "role": "user",
                "content": USER_PROMPTS["analysis_request"].format(query=query),
            },
        ]

        tools = self._get_openai_tools()

        try:
            all_function_results: List[Dict[str, Any]] = []
            last_plot_file: Optional[str] = None

            functions_called: List[str] = []
            parameters_called: List[Dict[str, Any]] = []

            safety_counter = 0
            while True:
                safety_counter += 1
                if safety_counter > LLM_CONFIG["max_tool_calls"]:
                    logger.warning("Reached maximum tool calls limit")
                    break

                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, tool_choice="auto"
                )

                choice = response.choices[0]
                msg = choice.message

                # If the model requested tool calls, execute them and continue
                if getattr(msg, "tool_calls", None):
                    # Add the assistant message with tool calls to history
                    assistant_msg: Dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                    messages.append(assistant_msg)

                    for tc in msg.tool_calls:
                        fname = tc.function.name
                        try:
                            fargs = (
                                json.loads(tc.function.arguments)
                                if tc.function.arguments
                                else {}
                            )
                        except Exception:
                            fargs = {}
                        if fargs is None:
                            fargs = {}

                        functions_called.append(fname)
                        parameters_called.append(fargs)

                        exec_result = self._execute_function(fname, fargs)
                        if isinstance(exec_result, dict) and "figure" in exec_result:
                            # Dictionary with base64 figure
                            last_plot_file = exec_result.get("figure")
                        elif isinstance(exec_result, str) and exec_result.startswith(
                            "data:image/"
                        ):
                            # Direct base64 string
                            last_plot_file = exec_result

                        # Add tool result to history
                        all_function_results.append(
                            {"name": fname, "args": fargs, "response": exec_result}
                        )
                        tool_content_obj: Any
                        if isinstance(exec_result, (dict, list)):
                            tool_content_obj = exec_result
                        else:
                            tool_content_obj = {"result": exec_result}

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": fname,
                                "content": json.dumps(tool_content_obj),
                            }
                        )

                    # Continue the loop to let the model incorporate tool results
                    continue

                # No tool calls: finalize
                final_text = msg.content or ""
                logger.info("Analysis completed successfully")
                return {
                    "success": True,
                    "functions_called": functions_called,
                    "parameters": parameters_called,
                    "result": (
                        all_function_results[-1]["response"]
                        if all_function_results
                        else None
                    ),
                    "plot_file": last_plot_file,
                    "llm_response": final_text,
                    "dataset_analyzed": self.current_dataset,
                }

            # Fallback if loop exits by safety counter
            logger.warning("Reached tool-calling step limit without final answer")
            return {
                "success": True,
                "functions_called": functions_called,
                "parameters": parameters_called,
                "result": None,
                "plot_file": last_plot_file,
                "llm_response": "Reached tool-calling step limit without final answer.",
                "dataset_analyzed": self.current_dataset,
            }

        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return {
                "success": False,
                "message": f"Error in LLM analysis: {str(e)}",
                "result": None,
                "plot_file": None,
            }

    def get_all_datasets_info(self) -> Dict[str, Any]:
        """
        Get information about all loaded datasets.

        Returns:
            Dictionary with information about all datasets
        """
        datasets_info = {}

        for dataset_name, analyzer in self.analyzers.items():
            data_types_str = {
                col: str(dtype) for col, dtype in analyzer.df.dtypes.to_dict().items()
            }
            missing_values = {
                col: int(count)
                for col, count in analyzer.df.isnull().sum().to_dict().items()
            }

            datasets_info[dataset_name] = {
                "shape": list(analyzer.df.shape),
                "columns": list(analyzer.df.columns),
                "data_types": data_types_str,
                "first_few_rows": analyzer.df.head(10).to_dict("records"),
                "missing_values": missing_values,
                "numeric_columns": analyzer.numeric_columns,
                "categorical_columns": analyzer.categorical_columns,
                "datetime_columns": analyzer.datetime_columns,
            }

        return datasets_info

    def get_dataset_preview(self) -> Dict[str, Any]:
        """
        Get a preview of the currently selected dataset.

        Returns:
            Dictionary with dataset preview information
        """
        current_analyzer = self._get_current_analyzer()
        if not current_analyzer:
            return {"error": "No dataset selected"}

        data_types_str = {
            col: str(dtype)
            for col, dtype in current_analyzer.df.dtypes.to_dict().items()
        }
        missing_values = {
            col: int(count)
            for col, count in current_analyzer.df.isnull().sum().to_dict().items()
        }
        return {
            "shape": list(current_analyzer.df.shape),
            "columns": list(current_analyzer.df.columns),
            "data_types": data_types_str,
            "first_few_rows": current_analyzer.df.head(10).to_dict("records"),
            "missing_values": missing_values,
        }
