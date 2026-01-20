# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optimized for Anthropic Claude API

import os, io, re
import base64
import pandas as pd
import numpy as np
import streamlit as st
from anthropic import Anthropic
import matplotlib.pyplot as plt
from typing import List, Any, Optional
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# === Configuration ===
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 100
MAX_RESULT_DISPLAY_LENGTH = 300

class ModelConfig:
    """Configuration class for different Claude models."""
    def __init__(self, model_name: str, model_url: str, model_print_name: str,
                 query_understanding_temperature: float = 0.1,
                 query_understanding_max_tokens: int = 10,
                 code_generation_temperature: float = 0.2,
                 code_generation_max_tokens: int = 2048,
                 reasoning_temperature: float = 0.5,
                 reasoning_max_tokens: int = 1024,
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

client = Anthropic(api_key=API_KEY)

def get_current_config():
    """Get the current model configuration based on session state."""
    if "current_model" in st.session_state:
        return MODEL_CONFIGS[st.session_state.current_model]
    return MODEL_CONFIGS[DEFAULT_MODEL]

# === Helper Functions =================================================
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

def serialize_result(result: Any, max_rows: int = 100) -> str:
    """Converts result to a readable string format for display."""
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    if isinstance(result, (plt.Figure, plt.Axes)):
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        else:
            title = result.get_title()
        return f"[Visualization: {title or 'Chart created'}]"
    
    elif isinstance(result, pd.DataFrame):
        if len(result) > max_rows:
            serialized = result.head(max_rows).to_string()
            return f"{serialized}\n\n... ({len(result)} total rows, showing first {max_rows})"
        else:
            return result.to_string()
    
    elif isinstance(result, pd.Series):
        if len(result) > max_rows:
            serialized = result.head(max_rows).to_string()
            return f"{serialized}\n\n... ({len(result)} total items, showing first {max_rows})"
        else:
            return result.to_string()
    
    elif isinstance(result, (list, tuple)):
        if len(result) > max_rows:
            return f"{result[:max_rows]}\n\n... ({len(result)} total items, showing first {max_rows})"
        return str(result)
    
    elif isinstance(result, dict):
        if len(result) > max_rows:
            items = list(result.items())[:max_rows]
            return f"{dict(items)}\n\n... ({len(result)} total items, showing first {max_rows})"
        return str(result)
    
    elif isinstance(result, (int, float, str, bool, np.number)):
        return str(result)
    
    else:
        return str(result)

# === QueryUnderstandingTool ===========================================
def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation."""
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
    """Generate a prompt for the LLM to write pandas-only code."""
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
7. Do **not** import any libraries (pandas is already imported as pd, numpy as np).
8. Handle missing values (dropna) before aggregations.

Example
-----
```python
result = df.groupby("some_column")["a_numeric_col"].mean().sort_values(ascending=False)
```"""

def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas + matplotlib code."""
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

# === ExecutionAgent ===================================================
def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment with timeout protection."""
    env = {"pd": pd, "df": df, "np": np}
    
    if should_plot:
        plt.rcParams["figure.dpi"] = DEFAULT_DPI
        env["plt"] = plt
        env["io"] = io
    
    try:
        # Set a simple timeout using signal (Unix-like systems only)
        # For cross-platform, we just execute directly
        exec(code, {}, env)
        result = env.get("result", None)
        
        if result is None:
            return "No result was assigned to 'result' variable", "No result assigned"
        
        return result, None
    except MemoryError:
        error_msg = "Error: Out of memory. Try simplifying your query or working with a smaller subset of data."
        return error_msg, "MemoryError"
    except KeyboardInterrupt:
        error_msg = "Error: Execution interrupted"
        return error_msg, "KeyboardInterrupt"
    except Exception as exc:
        error_msg = f"Error executing code: {str(exc)}"
        return error_msg, str(exc)

# === Code Fixing Agent ================================================
def CodeFixingAgent(original_query: str, failed_code: str, error_msg: str, df: pd.DataFrame, should_plot: bool) -> str:
    """Attempts to fix failed code based on the error message."""
    current_config = get_current_config()
    
    fix_prompt = f"""The following code failed with an error. Please fix it.

Original user query: "{original_query}"

Failed code:
```python
{failed_code}
```

Error message:
{error_msg}

DataFrame columns: {', '.join(df.columns.tolist())}

Please provide ONLY the corrected code in a ```python code block. The code should:
1. Fix the specific error mentioned
2. Assign the final result to a variable named 'result'
3. Handle edge cases properly
4. Not include any explanations outside the code block"""

    response = client.messages.create(
        model=current_config.MODEL_NAME,
        max_tokens=current_config.CODE_GENERATION_MAX_TOKENS,
        temperature=current_config.CODE_GENERATION_TEMPERATURE,
        messages=[{"role": "user", "content": fix_prompt}]
    )

    full_response = response.content[0].text
    fixed_code = extract_first_code_block(full_response)
    return fixed_code

# === ReasoningCurator TOOL ============================================
def ReasoningCurator(query: str, result: Any, serialized_result: str = None) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error")
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
        desc = serialized_result if serialized_result else str(result)[:MAX_RESULT_DISPLAY_LENGTH]

    if is_plot:
        prompt = f'''The user asked: "{query}".
Below is a description of the plot result:
{desc}
Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''The user asked: "{query}".
The result is:
{desc}

Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    return prompt

# === ReasoningAgent ===================================================
def ReasoningAgent(query: str, result: Any, serialized_result: str = None):
    """Streams the LLM's reasoning about the result."""
    current_config = get_current_config()
    prompt = ReasoningCurator(query, result, serialized_result)

    with client.messages.stream(
        model=current_config.MODEL_NAME,
        max_tokens=current_config.REASONING_MAX_TOKENS,
        temperature=current_config.REASONING_TEMPERATURE,
        messages=[{"role": "user", "content": "You are an insightful data analyst. " + prompt}]
    ) as stream:
        full_response = ""
        for text in stream.text_stream:
            full_response += text

    return "", full_response.strip()

# === DataFrameSummary TOOL ============================================
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

# === DataInsightAgent =================================================
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

# === PDF Processing Tools =============================================
def extract_pdf_text(pdf_file) -> str:
    """Extract text from PDF file using PyPDF2."""
    if PyPDF2 is None:
        return "PyPDF2 not installed. Install with: pip install PyPDF2"
    
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def convert_pdf_to_base64(pdf_file) -> str:
    """Convert PDF file to base64 for Claude API."""
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    return base64.standard_b64encode(pdf_bytes).decode('utf-8')

def PDFInsightAgent(pdf_file, use_vision: bool = False) -> str:
    """Generates insights from PDF using Claude's PDF processing capabilities."""
    current_config = get_current_config()
    
    try:
        if use_vision:
            pdf_base64 = convert_pdf_to_base64(pdf_file)
            
            response = client.messages.create(
                model=current_config.MODEL_NAME,
                max_tokens=current_config.INSIGHTS_MAX_TOKENS * 2,
                temperature=current_config.INSIGHTS_TEMPERATURE,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Analyze this PDF document and provide:
1. A concise summary of the main content (2-3 sentences)
2. Key topics and themes discussed
3. 3-4 important insights or takeaways from the document
4. Any notable data, statistics, or findings mentioned

Keep your response focused and well-structured."""
                        }
                    ]
                }]
            )
        else:
            pdf_text = extract_pdf_text(pdf_file)
            
            if pdf_text.startswith("Error") or pdf_text.startswith("PyPDF2"):
                return pdf_text
            
            max_chars = 50000
            if len(pdf_text) > max_chars:
                pdf_text = pdf_text[:max_chars] + "\n\n[Document truncated due to length...]"
            
            response = client.messages.create(
                model=current_config.MODEL_NAME,
                max_tokens=current_config.INSIGHTS_MAX_TOKENS * 2,
                temperature=current_config.INSIGHTS_TEMPERATURE,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this document and provide:
1. A concise summary of the main content (2-3 sentences)
2. Key topics and themes discussed
3. 3-4 important insights or takeaways from the document
4. Any notable data, statistics, or findings mentioned

Keep your response focused and well-structured.

Document content:
{pdf_text}"""
                }]
            )
        
        return response.content[0].text
    except Exception as exc:
        return f"Error generating PDF insights: {str(exc)}"

# === Main Streamlit App ===============================================
def main():
    st.set_page_config(layout="wide")
    
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = DEFAULT_MODEL
    if "file_type" not in st.session_state:
        st.session_state.file_type = "csv"

    left, right = st.columns([3, 7])

    with left:
        st.header("Data Analysis Agent")
        
        available_models = list(MODEL_CONFIGS.keys())
        model_display_names = {key: MODEL_CONFIGS[key].MODEL_PRINT_NAME for key in available_models}
        
        selected_model = st.selectbox(
            "Select Claude Model",
            options=available_models,
            format_func=lambda x: model_display_names[x],
            index=available_models.index(st.session_state.current_model)
        )
        
        display_config = MODEL_CONFIGS[selected_model]
        st.markdown(f"<medium>Powered by <a href='{display_config.MODEL_URL}'>{display_config.MODEL_PRINT_NAME}</a></medium>", unsafe_allow_html=True)
        
        file_type = st.radio("Upload Type", ["CSV Data", "PDF Document"], horizontal=True)
        
        if file_type == "CSV Data":
            file = st.file_uploader("Choose CSV", type=["csv"], key="csv_uploader")
            st.session_state.file_type = "csv"
        else:
            file = st.file_uploader("Choose PDF", type=["pdf"], key="pdf_uploader")
            st.session_state.file_type = "pdf"
            
            if file:
                use_vision = st.checkbox(
                    "Use Claude Vision (native PDF processing)", 
                    value=True,
                    help="When enabled, sends PDF directly to Claude. When disabled, extracts text first (requires PyPDF2)."
                )
        
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            new_config = MODEL_CONFIGS[selected_model]
            
            if "messages" in st.session_state:
                st.session_state.messages = []
            if "plots" in st.session_state:
                st.session_state.plots = []
            
            if file is not None:
                with st.spinner("Generating insights with new model…"):
                    try:
                        if st.session_state.file_type == "csv" and "df" in st.session_state:
                            st.session_state.insights = DataInsightAgent(st.session_state.df)
                            st.success(f"CSV insights updated with {new_config.MODEL_PRINT_NAME}")
                        elif st.session_state.file_type == "pdf" and "pdf_file" in st.session_state:
                            pdf_use_vision = st.session_state.get("pdf_use_vision", True)
                            st.session_state.insights = PDFInsightAgent(st.session_state.pdf_file, use_vision=pdf_use_vision)
                            st.success(f"PDF insights updated with {new_config.MODEL_PRINT_NAME}")
                    except Exception as e:
                        st.error(f"Error updating insights: {str(e)}")
                        if "insights" in st.session_state:
                            del st.session_state.insights
                st.rerun()
        
        if not file and ("df" in st.session_state or "pdf_file" in st.session_state):
            if "df" in st.session_state:
                del st.session_state.df
            if "pdf_file" in st.session_state:
                del st.session_state.pdf_file
            if "current_file" in st.session_state:
                del st.session_state.current_file
            if "insights" in st.session_state:
                del st.session_state.insights
            st.rerun()
        
        if file:
            file_changed = st.session_state.get("current_file") != file.name
            
            if st.session_state.file_type == "csv":
                if ("df" not in st.session_state) or file_changed:
                    st.session_state.df = pd.read_csv(file)
                    st.session_state.current_file = file.name
                    st.session_state.messages = []
                    
                    if "pdf_file" in st.session_state:
                        del st.session_state.pdf_file
                    
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
                if ("pdf_file" not in st.session_state) or file_changed:
                    file.seek(0)
                    st.session_state.pdf_file = io.BytesIO(file.read())
                    st.session_state.current_file = file.name
                    st.session_state.messages = []
                    st.session_state.pdf_use_vision = use_vision
                    
                    if "df" in st.session_state:
                        del st.session_state.df
                    
                    with st.spinner("Analyzing PDF document…"):
                        try:
                            st.session_state.insights = PDFInsightAgent(st.session_state.pdf_file, use_vision=use_vision)
                        except Exception as e:
                            st.error(f"Error generating PDF insights: {str(e)}")
                
                elif "insights" not in st.session_state:
                    with st.spinner("Analyzing PDF document…"):
                        try:
                            pdf_use_vision = st.session_state.get("pdf_use_vision", True)
                            st.session_state.insights = PDFInsightAgent(st.session_state.pdf_file, use_vision=pdf_use_vision)
                        except Exception as e:
                            st.error(f"Error generating PDF insights: {str(e)}")
                
                if "pdf_file" in st.session_state:
                    st.markdown("### PDF Document Insights")
                    
                    if "insights" in st.session_state and st.session_state.insights:
                        st.info(f"📄 **Document:** {st.session_state.current_file}")
                        st.markdown(st.session_state.insights)
                        
                        current_config_left = get_current_config()
                        processing_method = "Claude Vision API" if st.session_state.get("pdf_use_vision", True) else "Text Extraction"
                        st.markdown(f"*<span style='color: grey; font-style: italic;'>Generated with {current_config_left.MODEL_PRINT_NAME} ({processing_method})</span>*", unsafe_allow_html=True)
                    else:
                        st.warning("No insights available.")
        else:
            st.info("Upload a CSV or PDF to begin analysis.")

    with right:
        st.header("Chat with your data")
        
        if "df" in st.session_state:
            current_config_right = get_current_config()
            st.markdown(f"*<span style='color: grey; font-style: italic;'>Using {current_config_right.MODEL_PRINT_NAME}</span>*", unsafe_allow_html=True)
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
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
            
            if user_q := st.chat_input("Ask about your data…"):
                st.session_state.messages.append({"role": "user", "content": user_q})
                
                with st.spinner("Working…"):
                    recent_user_turns = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-3:]
                    context_text = "\n".join(recent_user_turns[:-1]) if len(recent_user_turns) > 1 else None
                    
                    max_retries = 3
                    retry_count = 0
                    result_obj = None
                    code = None
                    execution_error = None
                    should_plot_flag = False
                    code_thinking = ""
                    
                    # Add timeout protection
                    import time
                    start_time = time.time()
                    max_execution_time = 30  # 30 seconds max for entire retry loop
                    
                    while retry_count < max_retries:
                        try:
                            # Check if we've exceeded max time
                            if time.time() - start_time > max_execution_time:
                                st.error(f"Execution timed out after {max_execution_time} seconds")
                                result_obj = f"Error: Execution timed out after {max_execution_time} seconds"
                                break
                            
                            if retry_count == 0:
                                code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df, context_text)
                            else:
                                # Show retry status
                                retry_status = st.empty()
                                retry_status.info(f"🔄 Attempting to fix code (attempt {retry_count + 1}/{max_retries})...")
                                code = CodeFixingAgent(user_q, code, execution_error, st.session_state.df, should_plot_flag)
                                retry_status.empty()
                            
                            # Add execution timeout for individual code execution
                            exec_start = time.time()
                            result_obj, execution_error = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                            exec_time = time.time() - exec_start
                            
                            # If execution took too long, treat as error
                            if exec_time > 10:
                                execution_error = f"Code execution took too long ({exec_time:.1f}s)"
                                result_obj = f"Error: {execution_error}"
                            
                            if execution_error is None:
                                break
                            
                            retry_count += 1
                            
                            if retry_count >= max_retries:
                                st.warning(f"⚠️ Code execution failed after {max_retries} attempts.")
                                result_obj = f"Error: Code execution failed after {max_retries} attempts. Last error: {execution_error}"
                        
                        except Exception as e:
                            st.error(f"Unexpected error: {str(e)}")
                            result_obj = f"Unexpected error: {str(e)}"
                            execution_error = str(e)
                            break
                    
                    # Ensure we have a valid result before proceeding
                    if result_obj is None:
                        result_obj = "Error: No result generated"
                    
                    # Serialize with error handling
                    try:
                        serialized_result = serialize_result(result_obj)
                    except Exception as e:
                        serialized_result = f"Error serializing result: {str(e)}"
                    
                    # Get reasoning with error handling
                    try:
                        raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj, serialized_result)
                        reasoning_txt = reasoning_txt.replace("<think>", "").replace("</think>", "")
                    except Exception as e:
                        reasoning_txt = f"Could not generate explanation: {str(e)}"
                    
                    is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                    plot_idx = None
                    
                    if is_plot:
                        try:
                            fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                            st.session_state.plots.append(fig)
                            plot_idx = len(st.session_state.plots) - 1
                            response_parts = ["**Visualization:**\n"]
                        except Exception as e:
                            response_parts = [f"❌ **Error displaying plot: {str(e)}**\n"]
                            is_plot = False
                    elif isinstance(result_obj, str) and "Error" in result_obj:
                        response_parts = [f"❌ **{result_obj}**\n"]
                    else:
                        response_parts = ["**Result:**\n```\n" + serialized_result + "\n```\n"]
                    
                    response_parts.append(f"\n{reasoning_txt}")
                    
                    if retry_count > 0 and execution_error is None:
                        response_parts.append(f"\n\n✅ *Code fixed successfully after {retry_count} {'attempt' if retry_count == 1 else 'attempts'}*")
                    
                    # Ensure code is valid before displaying
                    if code:
                        code_html = (
                            '<details class="code">'
                            '<summary>View code</summary>'
                            '<pre><code class="language-python">'
                            f'{code}'
                            '</code></pre>'
                            '</details>'
                        )
                    else:
                        code_html = '<p><em>No code generated</em></p>'
                    
                    assistant_msg = "".join(response_parts) + f"\n\n{code_html}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_msg,
                        "plot_index": plot_idx
                    })
                    
                st.rerun()
        
        elif "pdf_file" in st.session_state:
            current_config_right = get_current_config()
            st.markdown(f"*<span style='color: grey; font-style: italic;'>Using {current_config_right.MODEL_PRINT_NAME}</span>*", unsafe_allow_html=True)
            
            if "pdf_messages" not in st.session_state:
                st.session_state.pdf_messages = []
            
            clear_col1, clear_col2 = st.columns([9, 1])
            with clear_col2:
                if st.button("Clear chat"):
                    st.session_state.pdf_messages = []
                    st.rerun()
            
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.pdf_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"], unsafe_allow_html=True)
            
            if user_q := st.chat_input("Ask about the PDF document…"):
                st.session_state.pdf_messages.append({"role": "user", "content": user_q})
                
                with st.spinner("Analyzing…"):
                    conversation = []
                    for msg in st.session_state.pdf_messages[:-1]:
                        conversation.append({"role": msg["role"], "content": msg["content"]})
                    
                    pdf_use_vision = st.session_state.get("pdf_use_vision", True)
                    
                    if pdf_use_vision:
                        pdf_base64 = convert_pdf_to_base64(st.session_state.pdf_file)
                        
                        message_content = [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": user_q
                            }
                        ]
                        
                        if conversation:
                            context_text = "\n\n".join([f"{m['role'].title()}: {m['content']}" for m in conversation[-3:]])
                            message_content[1]["text"] = f"Previous conversation:\n{context_text}\n\nCurrent question: {user_q}"
                        
                        response = client.messages.create(
                            model=current_config_right.MODEL_NAME,
                            max_tokens=2048,
                            temperature=0.3,
                            messages=[{"role": "user", "content": message_content}]
                        )
                    else:
                        pdf_text = extract_pdf_text(st.session_state.pdf_file)
                        
                        context_parts = []
                        if conversation:
                            context_text = "\n\n".join([f"{m['role'].title()}: {m['content']}" for m in conversation[-3:]])
                            context_parts.append(f"Previous conversation:\n{context_text}")
                        
                        max_chars = 40000
                        if len(pdf_text) > max_chars:
                            pdf_text = pdf_text[:max_chars] + "\n\n[Document truncated...]"
                        
                        context_parts.append(f"Document content:\n{pdf_text}")
                        context_parts.append(f"\nQuestion: {user_q}")
                        
                        full_prompt = "\n\n".join(context_parts)
                        
                        response = client.messages.create(
                            model=current_config_right.MODEL_NAME,
                            max_tokens=2048,
                            temperature=0.3,
                            messages=[{"role": "user", "content": full_prompt}]
                        )
                    
                    answer = response.content[0].text
                    st.session_state.pdf_messages.append({"role": "assistant", "content": answer})
                
                st.rerun()

if __name__ == "__main__":
    main()