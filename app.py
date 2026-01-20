# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optimized for Anthropic Claude API

import os, io, re
import pandas as pd
import numpy as np
import streamlit as st
from anthropic import Anthropic
import matplotlib.pyplot as plt
from typing import List, Any, Optional

# === Configuration ===
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
print(API_KEY)
DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 100
MAX_RESULT_DISPLAY_LENGTH = 300

class ModelConfig:
    """Configuration class for different Claude models."""
    def __init__(self, model_name: str, model_url: str, model_print_name: str,
                 # QueryUnderstandingTool parameters
                 query_understanding_temperature: float = 0.1,
                 query_understanding_max_tokens: int = 10,
                 # CodeGenerationAgent parameters
                 code_generation_temperature: float = 0.2,
                 code_generation_max_tokens: int = 2048,
                 # ReasoningAgent parameters
                 reasoning_temperature: float = 0.5,
                 reasoning_max_tokens: int = 1024,
                 # DataInsightAgent parameters
                 insights_temperature: float = 0.3,
                 insights_max_tokens: int = 1024):
        self.MODEL_NAME = model_name
        self.MODEL_URL = model_url
        self.MODEL_PRINT_NAME = model_print_name
        self.QUERY_UNDERSTANDING_TEMPERATURE = query_understanding_temperature
        self.QUERY_UNDERSTANDING_MAX_TOKENS = query_understanding_max_tokens
        self.CODE_GENERATION_TEMPERATURE = code_generation_temperature
        self.CODE_GENERATION_MAX_TOKENS = code_generation_max_tokens
        self.REASONING_TEMPERATURE = reasoning_temperature
        self.REASONING_MAX_TOKENS = reasoning_max_tokens
        self.INSIGHTS_TEMPERATURE = insights_temperature
        self.INSIGHTS_MAX_TOKENS = insights_max_tokens

# Predefined model configurations
MODEL_CONFIGS = {
    "claude-sonnet-4-5": ModelConfig(
        model_name="claude-sonnet-4-5-20250929",
        model_url="https://www.anthropic.com/claude/sonnet",
        model_print_name="Claude Sonnet 4.5",
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=10,
        code_generation_temperature=0.0,
        code_generation_max_tokens=2048,
        reasoning_temperature=0.5,
        reasoning_max_tokens=1024,
        insights_temperature=0.3,
        insights_max_tokens=1024
    ),
    "claude-opus-4-5": ModelConfig(
        model_name="claude-opus-4-5-20250514",
        model_url="https://www.anthropic.com/claude/opus",
        model_print_name="Claude Opus 4.5",
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=10,
        code_generation_temperature=0.0,
        code_generation_max_tokens=4096,
        reasoning_temperature=0.6,
        reasoning_max_tokens=2048,
        insights_temperature=0.3,
        insights_max_tokens=1024
    ),
    "claude-haiku-4-5": ModelConfig(
        model_name="claude-haiku-4-5-20251001",
        model_url="https://www.anthropic.com/claude/haiku",
        model_print_name="Claude Haiku 4.5",
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=10,
        code_generation_temperature=0.0,
        code_generation_max_tokens=1024,
        reasoning_temperature=0.5,
        reasoning_max_tokens=1024,
        insights_temperature=0.3,
        insights_max_tokens=512
    )
}

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-5")
Config = MODEL_CONFIGS.get(DEFAULT_MODEL, MODEL_CONFIGS["claude-sonnet-4-5"])

# Initialize Anthropic client
client = Anthropic(api_key=API_KEY)

def get_current_config():
    """Get the current model configuration based on session state."""
    if "current_model" in st.session_state:
        return MODEL_CONFIGS[st.session_state.current_model]
    return MODEL_CONFIGS[DEFAULT_MODEL]

# ------------------ QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on LLM understanding."""
    current_config = get_current_config()
    
    full_prompt = f"""You are a query classifier. Determine if a user query requests data visualization.

IMPORTANT: Respond with ONLY 'true' or 'false' (lowercase, no quotes, no punctuation).

Classify as 'true' ONLY if the query explicitly asks for:
- A plot, chart, graph, visualization, or figure
- To "show" or "display" data visually
- To "create" or "generate" a visual representation
- Words like: plot, chart, graph, visualize, show, display, create, generate, draw

Classify as 'false' for:
- Data analysis without visualization requests
- Statistical calculations, aggregations, filtering, sorting
- Questions about data content, counts, summaries
- Requests for tables, dataframes, or text results

User query: {query}"""

    response = client.messages.create(
        model=current_config.MODEL_NAME,
        max_tokens=current_config.QUERY_UNDERSTANDING_MAX_TOKENS,
        temperature=current_config.QUERY_UNDERSTANDING_TEMPERATURE,
        messages=[{"role": "user", "content": full_prompt}]
    )
    
    intent_response = response.content[0].text.strip().lower()
    return intent_response == "true"

# === CodeGeneration TOOLS ============================================
def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query."""
    return f"""Given DataFrame df with columns: {', '.join(cols)}

Write Python code (pandas **only**, no plotting) to answer: "{query}"

Rules
-----
1. Use pandas operations on df only.
2. Rely only on the columns in the DataFrame.
3. Assign the final result to result.
4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
5. Do not include any explanations, comments, or prose outside the code block.
6. Use **df** as the sole data source. **Do not** read files, fetch data, or use Streamlit.
7. Do **not** import any libraries (pandas is already imported as pd).
8. Handle missing values (dropna) before aggregations.

Example
-----
```python
result = df.groupby("some_column")["a_numeric_col"].mean().sort_values(ascending=False)
```"""

def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas + matplotlib code for a plot."""
    return f"""Given DataFrame df with columns: {', '.join(cols)}

Write Python code using pandas **and matplotlib** (as plt) to answer: "{query}"

Rules
-----
1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
2. Rely only on the columns in the DataFrame.
3. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named result.
4. Create only ONE relevant plot. Set figsize={DEFAULT_FIGSIZE}, add title/labels.
5. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
6. Do not include any explanations, comments, or prose outside the code block.
7. Handle missing values (dropna) before plotting/aggregations."""

# === CodeGenerationAgent ==============================================
def CodeGenerationAgent(query: str, df: pd.DataFrame, chat_context: Optional[str] = None):
    """Selects the appropriate code generation tool and gets code from the LLM."""
    should_plot = QueryUnderstandingTool(query)
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(), query)
    
    context_section = f"\nConversation context (recent user turns):\n{chat_context}\n" if chat_context else ""
    full_prompt = f"""You are a senior Python data analyst who writes clean, efficient code.

Solve the given problem with optimal pandas operations. Be concise and focused.

Your response must contain ONLY a properly-closed ```python code block with no explanations before or after.
Ensure your solution is correct, handles edge cases, and follows best practices for data analysis.

If the latest user request references prior results ambiguously (e.g., "it", "that", "same groups"), infer intent from the conversation context and choose the most reasonable interpretation.
{context_section}{prompt}"""

    current_config = get_current_config()

    response = client.messages.create(
        model=current_config.MODEL_NAME,
        max_tokens=current_config.CODE_GENERATION_MAX_TOKENS,
        temperature=current_config.CODE_GENERATION_TEMPERATURE,
        messages=[{"role": "user", "content": full_prompt}]
    )

    full_response = response.content[0].text
    code = extract_first_code_block(full_response)
    return code, should_plot, ""

# === ExecutionAgent ====================================================
def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment."""
    env = {"pd": pd, "df": df}
    
    if should_plot:
        plt.rcParams["figure.dpi"] = DEFAULT_DPI
        env["plt"] = plt
        env["io"] = io
    
    try:
        exec(code, {}, env)
        result = env.get("result", None)
        
        if result is None:
            return "No result was assigned to 'result' variable"
        
        return result
    except Exception as exc:
        return f"Error executing code: {exc}"

# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:MAX_RESULT_DISPLAY_LENGTH]

    if is_plot:
        prompt = f'''The user asked: "{query}".
Below is a description of the plot result:
{desc}
Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''The user asked: "{query}".
The result value is: {desc}
Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    return prompt

# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result."""
    current_config = get_current_config()
    prompt = ReasoningCurator(query, result)

    # Use streaming for better UX
    with client.messages.stream(
        model=current_config.MODEL_NAME,
        max_tokens=current_config.REASONING_MAX_TOKENS,
        temperature=current_config.REASONING_TEMPERATURE,
        messages=[{"role": "user", "content": "You are an insightful data analyst. " + prompt}]
    ) as stream:
        # Stream and display
        thinking_placeholder = st.empty()
        full_response = ""
        
        for text in stream.text_stream:
            full_response += text

    # After streaming, return the complete response
    return "", full_response.strip()

# === DataFrameSummary TOOL =========================================
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    prompt = f"""Given a dataset with {len(df)} rows and {len(df.columns)} columns:
Columns: {', '.join(df.columns)}
Data types: {df.dtypes.to_dict()}
Missing values: {df.isnull().sum().to_dict()}

Provide:
1. A brief description of what this dataset contains
2. 3-4 possible data analysis questions that could be explored
Keep it concise and focused."""
    return prompt

# === DataInsightAgent (upload-time only) ===============================
def DataInsightAgent(df: pd.DataFrame) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
    current_config = get_current_config()
    prompt = DataFrameSummaryTool(df)
    
    try:
        response = client.messages.create(
            model=current_config.MODEL_NAME,
            max_tokens=current_config.INSIGHTS_MAX_TOKENS,
            temperature=current_config.INSIGHTS_TEMPERATURE,
            messages=[{"role": "user", "content": "You are a data analyst providing brief, focused insights. " + prompt}]
        )
        return response.content[0].text
    except Exception as exc:
        raise Exception(f"Error generating dataset insights: {exc}")

# === Helpers ===========================================================
def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === Main Streamlit App ===============================================
def main():
    st.set_page_config(layout="wide")
    
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = DEFAULT_MODEL

    left, right = st.columns([3, 7])

    with left:
        st.header("Data Analysis Agent")
        
        # Model selector
        available_models = list(MODEL_CONFIGS.keys())
        model_display_names = {key: MODEL_CONFIGS[key].MODEL_PRINT_NAME for key in available_models}
        
        selected_model = st.selectbox(
            "Select Claude Model",
            options=available_models,
            format_func=lambda x: model_display_names[x],
            index=available_models.index(st.session_state.current_model)
        )
        
        # Display current model info
        display_config = MODEL_CONFIGS[selected_model]
        st.markdown(f"<medium>Powered by <a href='{display_config.MODEL_URL}'>{display_config.MODEL_PRINT_NAME}</a></medium>", unsafe_allow_html=True)
        
        file = st.file_uploader("Choose CSV", type=["csv"], key="csv_uploader")
        
        # Update configuration if model changed
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            new_config = MODEL_CONFIGS[selected_model]
            
            # Clear chat history when model changes
            if "messages" in st.session_state:
                st.session_state.messages = []
            if "plots" in st.session_state:
                st.session_state.plots = []
            
            # Regenerate insights immediately if we have data and file is present
            if "df" in st.session_state and file is not None:
                with st.spinner("Generating dataset insights with new model…"):
                    try:
                        st.session_state.insights = DataInsightAgent(st.session_state.df)
                        st.success(f"Insights updated with {new_config.MODEL_PRINT_NAME}")
                    except Exception as e:
                        st.error(f"Error updating insights: {str(e)}")
                        if "insights" in st.session_state:
                            del st.session_state.insights
                st.rerun()
        
        # Clear data if file is removed
        if not file and "df" in st.session_state and "current_file" in st.session_state:
            del st.session_state.df
            del st.session_state.current_file
            if "insights" in st.session_state:
                del st.session_state.insights
            st.rerun()
        
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                st.session_state.df = pd.read_csv(file)
                st.session_state.current_file = file.name
                st.session_state.messages = []
                
                # Generate insights with the current model
                with st.spinner("Generating dataset insights…"):
                    try:
                        st.session_state.insights = DataInsightAgent(st.session_state.df)
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
            
            elif "insights" not in st.session_state:
                with st.spinner("Generating dataset insights…"):
                    try:
                        st.session_state.insights = DataInsightAgent(st.session_state.df)
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
            
            # Display data and insights
            if "df" in st.session_state:
                st.markdown("### Dataset Insights")
                
                if "insights" in st.session_state and st.session_state.insights:
                    st.dataframe(st.session_state.df.head())
                    st.markdown(st.session_state.insights)
                    
                    current_config_left = get_current_config()
                    st.markdown(f"*<span style='color: grey; font-style: italic;'>Generated with {current_config_left.MODEL_PRINT_NAME}</span>*", unsafe_allow_html=True)
                else:
                    st.warning("No insights available.")
        else:
            st.info("Upload a CSV to begin chatting with your data.")

    with right:
        st.header("Chat with your data")
        
        if "df" in st.session_state:
            current_config_right = get_current_config()
            st.markdown(f"*<span style='color: grey; font-style: italic;'>Using {current_config_right.MODEL_PRINT_NAME}</span>*", unsafe_allow_html=True)
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Clear chat control
            clear_col1, clear_col2 = st.columns([9, 1])
            with clear_col2:
                if st.button("Clear chat"):
                    st.session_state.messages = []
                    st.session_state.plots = []
                    st.rerun()
            
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"], unsafe_allow_html=True)
                        if msg.get("plot_index") is not None:
                            idx = msg["plot_index"]
                            if 0 <= idx < len(st.session_state.plots):
                                st.pyplot(st.session_state.plots[idx], use_container_width=False)
            
            if "df" in st.session_state:
                if user_q := st.chat_input("Ask about your data…"):
                    st.session_state.messages.append({"role": "user", "content": user_q})
                    
                    with st.spinner("Working…"):
                        # Build brief chat context from the last few user messages
                        recent_user_turns = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-3:]
                        context_text = "\n".join(recent_user_turns[:-1]) if len(recent_user_turns) > 1 else None
                        
                        code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df, context_text)
                        result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                        raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                        
                        reasoning_txt = reasoning_txt.replace("<think>", "").replace("</think>", "")
                        
                        # Build assistant response
                        is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                        plot_idx = None
                        
                        if is_plot:
                            fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                            st.session_state.plots.append(fig)
                            plot_idx = len(st.session_state.plots) - 1
                            header = "Here is the visualization you requested:"
                        elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                            header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
                        else:
                            header = f"Result: {result_obj}"
                        
                        # Show reasoning
                        thinking_html = ""
                        if raw_thinking:
                            thinking_html = (
                                '<details class="thinking">'
                                '<summary>🧠 Reasoning</summary>'
                                f'<pre>{raw_thinking}</pre>'
                                '</details>'
                            )
                        
                        explanation_html = reasoning_txt
                        
                        # Code accordion
                        code_html = (
                            '<details class="code">'
                            '<summary>View code</summary>'
                            '<pre><code class="language-python">'
                            f'{code}'
                            '</code></pre>'
                            '</details>'
                        )
                        
                        assistant_msg = f"{thinking_html}{explanation_html}\n\n{code_html}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": assistant_msg,
                            "plot_index": plot_idx
                        })
                        
                    st.rerun()

if __name__ == "__main__":
    main()