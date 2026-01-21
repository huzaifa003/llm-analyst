# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optimized for Anthropic Claude API with Enhanced Error Handling and Crash Prevention
# ENHANCED: DataFrame context included in every prompt

import os, io, re
import base64
import pandas as pd
import numpy as np
import streamlit as st
from anthropic import Anthropic
import matplotlib.pyplot as plt
from typing import List, Any, Optional, Tuple
import traceback
import multiprocessing
from multiprocessing import Process, Queue
import ast
import pickle

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# === Configuration ===
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 100
MAX_RESULT_DISPLAY_LENGTH = 300
CODE_EXECUTION_TIMEOUT = 30  # seconds
TOTAL_RETRY_TIMEOUT = 60  # seconds

class ExecutionError(Exception):
    """Custom exception for code execution errors"""
    pass

class TimeoutError(Exception):
    """Custom exception for timeout errors"""
    pass

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
        query_understanding_max_tokens=1024,
        code_generation_temperature=0.0,
        code_generation_max_tokens=4096,
        reasoning_temperature=0.5,
        reasoning_max_tokens=2048,
        insights_temperature=0.3,
        insights_max_tokens=2048
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

# === NEW: DataFrame Context Generator ================================
def generate_dataframe_context(df: pd.DataFrame, max_sample_rows: int = 5) -> str:
    """
    Generate a comprehensive context about the DataFrame to include in prompts.
    This provides the model with column names, types, sample values, and basic statistics.
    """
    context_parts = []
    
    # Basic info
    context_parts.append(f"**DataFrame Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    context_parts.append(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Column information with types and sample values
    context_parts.append("\n**Columns with Data Types and Sample Values:**")
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        
        # Get sample non-null values
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            sample_values = non_null_values.head(3).tolist()
            sample_str = ", ".join([str(v)[:50] for v in sample_values])  # Limit length
        else:
            sample_str = "All null"
        
        null_info = f" ({null_count:,} nulls, {null_pct:.1f}%)" if null_count > 0 else ""
        context_parts.append(f"  - '{col}': {dtype}{null_info}")
        context_parts.append(f"    Sample values: {sample_str}")
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        context_parts.append(f"\n**Numeric Columns ({len(numeric_cols)}):** {', '.join(numeric_cols)}")
        context_parts.append("**Quick Statistics for Numeric Columns:**")
        stats = df[numeric_cols].describe().loc[['min', 'max', 'mean']].round(2)
        for col in numeric_cols[:5]:  # Limit to first 5 to avoid too much text
            if col in stats.columns:
                context_parts.append(f"  - {col}: min={stats.loc['min', col]}, max={stats.loc['max', col]}, mean={stats.loc['mean', col]}")
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        context_parts.append(f"\n**Categorical/Text Columns ({len(categorical_cols)}):** {', '.join(categorical_cols)}")
        for col in categorical_cols[:5]:  # Limit to first 5
            unique_count = df[col].nunique()
            context_parts.append(f"  - {col}: {unique_count:,} unique values")
            if unique_count <= 10:  # Show values if few enough
                top_values = df[col].value_counts().head(5).index.tolist()
                context_parts.append(f"    Top values: {', '.join([str(v)[:30] for v in top_values])}")
    
    # DateTime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        context_parts.append(f"\n**DateTime Columns ({len(datetime_cols)}):** {', '.join(datetime_cols)}")
        for col in datetime_cols:
            min_date = df[col].min()
            max_date = df[col].max()
            context_parts.append(f"  - {col}: {min_date} to {max_date}")
    
    return "\n".join(context_parts)

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
    try:
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
    except Exception as e:
        return f"Error serializing result: {str(e)}"

def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"

def analyze_code_safety(code: str, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Analyze code for patterns that might cause crashes."""
    
    warnings = []
    
    # Check for infinite loop patterns
    if re.search(r'\bwhile\s+True\b', code):
        return False, "Code contains 'while True' which may cause infinite loop"
    
    # Check for very long while loops
    if re.search(r'\bwhile\s+', code) and 'break' not in code:
        warnings.append("Warning: while loop without visible break statement")
    
    # Check for nested loops on large data
    loop_count = len(re.findall(r'\bfor\b', code))
    if loop_count >= 2 and len(df) > 10000:
        warnings.append("Warning: Nested loops on large dataset may be slow")
    
    # Check for operations that create large intermediate results
    if re.search(r'\.apply\(.*lambda', code) and len(df) > 50000:
        warnings.append("Warning: apply() with lambda on large dataset may be slow")
    
    # Check for cartesian products
    if 'merge' in code and 'how=' not in code and len(df) > 1000:
        warnings.append("Warning: merge without 'how=' parameter might create cartesian product")
    
    # Check for duplicating data
    if code.count('.copy()') > 2:
        warnings.append("Warning: Multiple .copy() operations may use excessive memory")
    
    # Check for recursive operations (look for def with function name, then search for that name later)
    func_def_match = re.search(r'\bdef\s+(\w+)\s*\(', code)
    if func_def_match:
        func_name = func_def_match.group(1)
        # Check if function calls itself in its body
        func_start = func_def_match.end()
        func_body = code[func_start:]
        if re.search(rf'\b{func_name}\s*\(', func_body):
            warnings.append("Warning: Possible recursive function detected")
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    return True, None

# === QueryUnderstandingTool ===========================================
def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation."""
    try:
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
    except Exception as e:
        st.error(f"Error in query understanding: {str(e)}")
        return False

# === CodeGeneration TOOLS (UPDATED) ===================================
def CodeWritingTool(cols: List[str], query: str, df_context: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code with full DataFrame context."""
    return f"""Given DataFrame 'df' with the following detailed information:

{df_context}

Write Python code (pandas **only**, no plotting) to answer: "{query}"

Rules
-----
1. Use pandas operations on df only.
2. Rely only on the columns available in the DataFrame (listed above).
3. Assign the final result to a variable named 'result'.
4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
5. Do not include any explanations, comments, or prose outside the code block.
6. Use **df** as the sole data source. **Do not** read files, fetch data, or use Streamlit.
7. Do **not** import any libraries (pandas is already imported as pd, numpy as np).
8. Handle missing values (dropna) before aggregations when appropriate.
9. Be memory-efficient: avoid creating unnecessary copies, use inplace operations where appropriate.
10. For large operations, consider using chunking or sampling if needed.
11. Avoid nested loops and use vectorized operations instead.
12. Do NOT use while loops - use vectorized pandas operations instead.
13. Pay attention to data types when performing operations.

Example
-----
```python
result = df.groupby("some_column")["a_numeric_col"].mean().sort_values(ascending=False)
```"""

def PlotCodeGeneratorTool(cols: List[str], query: str, df_context: str) -> str:
    """Generate a prompt for the LLM to write pandas + matplotlib code with full DataFrame context."""
    return f"""Given DataFrame 'df' with the following detailed information:

{df_context}

Write Python code using pandas **and matplotlib** (as plt) to answer: "{query}"

Rules
-----
1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
2. Rely only on the columns available in the DataFrame (listed above).
3. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named 'result'.
4. Create only ONE relevant plot. Set figsize={DEFAULT_FIGSIZE}, add title/labels.
5. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
6. Do not include any explanations, comments, or prose outside the code block.
7. Handle missing values (dropna) before plotting/aggregations when appropriate.
8. Be memory-efficient: sample data if needed for visualization (e.g., df.sample(10000) for scatter plots).
9. Close any intermediate figures to prevent memory leaks.
10. Avoid nested loops and use vectorized operations instead.
11. Do NOT use while loops - use vectorized pandas operations instead.
12. Pay attention to data types when creating visualizations."""

# === CodeGenerationAgent (UPDATED) ====================================
def CodeGenerationAgent(query: str, df: pd.DataFrame, chat_context: Optional[str] = None) -> Tuple[str, bool, str]:
    """Selects the appropriate code generation tool and gets code from the LLM with full DataFrame context."""
    try:
        # Generate comprehensive DataFrame context
        df_context = generate_dataframe_context(df)
        
        should_plot = QueryUnderstandingTool(query)
        prompt = PlotCodeGeneratorTool(df.columns.tolist(), query, df_context) if should_plot else CodeWritingTool(df.columns.tolist(), query, df_context)
        
        context_section = f"\nConversation context (recent user turns):\n{chat_context}\n" if chat_context else ""
        full_prompt = f"""You are a senior Python data analyst who writes clean, efficient code.

Solve the given problem with optimal pandas operations. Be concise and focused.

Your response must contain ONLY a properly-closed ```python code block with no explanations before or after.
Ensure your solution is correct, handles edge cases, and follows best practices for data analysis.

CRITICAL SAFETY RULES:
- NEVER use while loops (use vectorized operations instead)
- NEVER use recursive functions
- Avoid nested loops on large datasets
- Use vectorized pandas operations for performance
- Sample large datasets before visualization

IMPORTANT: Write memory-efficient code. For large datasets:
- Sample data when appropriate (e.g., for scatter plots)
- Use inplace operations where possible
- Avoid creating unnecessary copies
- Handle potential memory errors gracefully

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
        
        if not code:
            raise ExecutionError("No valid Python code block found in response")
        
        return code, should_plot, ""
    
    except Exception as e:
        st.error(f"Error generating code: {str(e)}")
        raise

# === Safe Execution with Process Isolation ===========================
def execute_code_in_process(code: str, df_pickle: bytes, should_plot: bool, result_queue: Queue):
    """Execute code in a separate process that can be forcefully terminated."""
    try:
        # Unpickle the dataframe
        df = pickle.loads(df_pickle)
        
        env = {"pd": pd, "df": df, "np": np}
        
        if should_plot:
            plt.rcParams["figure.dpi"] = DEFAULT_DPI
            env["plt"] = plt
            env["io"] = io
        
        # Validate for dangerous operations
        dangerous_patterns = [
            (r'\beval\b', 'eval'),
            (r'\bexec\b', 'exec'),
            (r'\b__import__\b', '__import__'),
            (r'\bopen\b', 'open'),
            (r'\bfile\b', 'file'),
            (r'\binput\b', 'input'),
            (r'\bos\.', 'os module'),
            (r'\bsys\.', 'sys module'),
            (r'\bsubprocess\b', 'subprocess')
        ]
        for pattern, name in dangerous_patterns:
            if re.search(pattern, code):
                result_queue.put(("error", f"Code contains potentially dangerous operation: {name}", "SecurityError"))
                return
        
        # Execute the entire code block at once (safer than line-by-line)
        exec(code, env)
        
        result = env.get("result", None)
        
        if result is None:
            result_queue.put(("error", "No result assigned to 'result' variable", "NoResult"))
            return
        
        # Handle matplotlib figures specially
        if should_plot and isinstance(result, (plt.Figure, plt.Axes)):
            # Convert figure to bytes
            fig = result.figure if isinstance(result, plt.Axes) else result
            fig_bytes = pickle.dumps(fig)
            result_queue.put(("success", fig_bytes, "plot"))
        else:
            # For non-plot results, pickle them
            result_queue.put(("success", pickle.dumps(result), "data"))
            
    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put(("error", f"{str(e)}\n\nTraceback:\n{tb}", str(type(e).__name__)))
    finally:
        # Clean up matplotlib
        if should_plot:
            plt.close('all')


def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool) -> Tuple[Any, Optional[str]]:
    """Executes code in a separate process with hard timeout."""
    
    # Check dataframe size and warn
    df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if df_size_mb > 100:
        st.warning(f"⚠️ Large dataset ({df_size_mb:.1f} MB). Execution may be slow.")
    
    try:
        # Pickle the dataframe for inter-process communication
        df_pickle = pickle.dumps(df)
        
        # Create a queue for results
        result_queue = Queue()
        
        # Create and start the process
        process = Process(
            target=execute_code_in_process,
            args=(code, df_pickle, should_plot, result_queue)
        )
        process.start()
        
        # Wait for result with timeout
        process.join(timeout=CODE_EXECUTION_TIMEOUT)
        
        if process.is_alive():
            # Force kill the process
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            
            return (f"Error: Code execution timed out after {CODE_EXECUTION_TIMEOUT} seconds. "
                   "The operation is taking too long. Try:\n"
                   "- Simplifying your query\n"
                   "- Using df.sample() to work with less data\n"
                   "- Breaking into smaller steps"), "TimeoutError"
        
        # Get result from queue
        if not result_queue.empty():
            status, data, data_type = result_queue.get()
            
            if status == "error":
                return data, data_type
            
            # Unpickle result
            result = pickle.loads(data)
            
            # Check result size
            if isinstance(result, pd.DataFrame):
                size_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
                if size_mb > 100:
                    st.warning(f"⚠️ Result DataFrame is large ({size_mb:.1f} MB).")
            
            return result, None
        else:
            return "Error: No result returned from execution", "NoResult"
            
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error in execution wrapper: {str(e)}\n\nTraceback:\n{tb}", str(type(e).__name__)
    
    finally:
        # Clean up
        if should_plot:
            plt.close('all')

# === Code Fixing Agent (UPDATED) ======================================
def CodeFixingAgent(original_query: str, failed_code: str, error_msg: str, 
                    df: pd.DataFrame, should_plot: bool, retry_count: int, 
                    previous_errors: Optional[List[str]] = None) -> str:
    """Attempts to fix failed code based on the error message with full DataFrame context."""
    
    if previous_errors is None:
        previous_errors = []
    
    # Check if we're seeing the same error repeatedly
    repeated_error = previous_errors.count(error_msg) >= 2
    
    try:
        current_config = get_current_config()
        
        # Generate DataFrame context for fixing
        df_context = generate_dataframe_context(df)
        
        # Provide more context based on error type
        hint = ""
        if repeated_error:
            hint = "\n\n🔴 CRITICAL: This exact error has occurred multiple times. You MUST try a COMPLETELY DIFFERENT approach. Do not just modify the previous code - write it from scratch using a different strategy."
        elif "timeout" in error_msg.lower():
            hint = "\n\n💡 HINT: The code timed out. Consider:\n- Using .sample() for large datasets\n- Simplifying complex operations\n- Breaking into smaller steps\n- Using vectorized operations instead of loops"
        elif "memory" in error_msg.lower():
            hint = "\n\n💡 HINT: Out of memory. Consider:\n- Using .head() or .sample() to work with less data\n- Using more efficient data types\n- Avoiding unnecessary copies\n- Using chunking for large operations"
        elif "KeyError" in error_msg:
            hint = f"\n\n💡 HINT: Column not found. Review the DataFrame context below for available columns.\nDouble-check spelling and case sensitivity."
        elif "was never closed" in error_msg or "SyntaxError" in error_msg:
            hint = "\n\n💡 HINT: Syntax error detected. Carefully check:\n- All brackets [], (), {} are properly closed\n- All quotes are properly closed\n- No incomplete expressions\n- Proper indentation"
        elif "while" in failed_code.lower() or "TimeoutError" in error_msg:
            hint = "\n\n💡 HINT: Avoid while loops entirely. Use pandas vectorized operations:\n- Use .groupby(), .apply(), .transform()\n- Use boolean indexing: df[df['col'] > value]\n- Use .iterrows() only as last resort"
        
        # Show previous error history if applicable
        error_history = ""
        if len(previous_errors) > 0:
            unique_errors = list(set(previous_errors[-3:]))
            error_history = f"\n\n📋 Previous errors encountered:\n" + "\n".join([f"  - {e}" for e in unique_errors])
        
        fix_prompt = f"""The following code failed with an error. This is retry attempt {retry_count} of 3.

Original user query: "{original_query}"

Failed code:
```python
{failed_code}
```

❌ Error message:
{error_msg}{hint}{error_history}

📊 DataFrame Context (REVIEW THIS CAREFULLY):
{df_context}

🎯 CRITICAL INSTRUCTIONS:
1. The code MUST have valid Python syntax - check all brackets and quotes carefully
2. Test your logic step by step mentally before writing
3. {"Write COMPLETELY NEW code using a DIFFERENT approach - do not just modify the failed code" if repeated_error else "Fix the specific error while keeping the logic sound"}
4. Assign the final result to a variable named 'result'
5. Handle edge cases properly (missing values, empty results, etc.)
6. Be memory-efficient for large datasets
7. NEVER use while loops - always use vectorized pandas operations
8. Avoid nested loops - use .groupby(), .merge(), or vectorized operations instead
9. For large datasets, use .sample() to work with a subset
10. **Verify column names match exactly** (case-sensitive) with the DataFrame context above

⚠️ FORBIDDEN OPERATIONS:
- while loops (use vectorized operations)
- Nested for loops on large data (use .groupby())
- Recursive functions
- Operations without handling NaN values when appropriate

Provide ONLY the corrected code in a ```python code block with NO explanations or commentary before or after."""

        # Adjust temperature based on retry count and error type
        if "Syntax" in error_msg or "was never closed" in error_msg:
            temperature = 0.0  # Deterministic for syntax errors
        elif repeated_error:
            temperature = min(0.4, current_config.CODE_GENERATION_TEMPERATURE + 0.15 * retry_count)
        else:
            temperature = min(0.3, current_config.CODE_GENERATION_TEMPERATURE + 0.1 * retry_count)

        response = client.messages.create(
            model=current_config.MODEL_NAME,
            max_tokens=current_config.CODE_GENERATION_MAX_TOKENS,
            temperature=temperature,
            messages=[{"role": "user", "content": fix_prompt}]
        )

        full_response = response.content[0].text
        fixed_code = extract_first_code_block(full_response)
        
        if not fixed_code:
            raise ExecutionError("No valid Python code block found in fix response")
        
        return fixed_code
    
    except Exception as e:
        st.error(f"Error in code fixing agent: {str(e)}")
        raise

# === ReasoningCurator TOOL ============================================
def ReasoningCurator(query: str, result: Any, serialized_result: str = None) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and "Error" in result
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
def ReasoningAgent(query: str, result: Any, serialized_result: str = None) -> Tuple[str, str]:
    """Streams the LLM's reasoning about the result."""
    try:
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
    
    except Exception as e:
        return "", f"Could not generate explanation: {str(e)}"

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
    try:
        current_config = get_current_config()
        prompt = DataFrameSummaryTool(df)
        
        response = client.messages.create(
            model=current_config.MODEL_NAME,
            max_tokens=current_config.INSIGHTS_MAX_TOKENS,
            temperature=current_config.INSIGHTS_TEMPERATURE,
            messages=[{"role": "user", "content": "You are a data analyst providing brief, focused insights. " + prompt}]
        )
        return response.content[0].text
    except Exception as exc:
        error_msg = f"Error generating dataset insights: {str(exc)}"
        st.error(error_msg)
        return error_msg

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
    try:
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        return base64.standard_b64encode(pdf_bytes).decode('utf-8')
    except Exception as e:
        raise Exception(f"Error converting PDF to base64: {str(e)}")

def PDFInsightAgent(pdf_file, use_vision: bool = False) -> str:
    """Generates insights from PDF using Claude's PDF processing capabilities."""
    try:
        current_config = get_current_config()
        
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
        error_msg = f"Error generating PDF insights: {str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
        st.error(error_msg)
        return error_msg

# === Main Streamlit App ===============================================
def main():
    # Configure multiprocessing (must be at the very start)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
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
                    try:
                        st.session_state.df = pd.read_csv(file)
                        st.session_state.current_file = file.name
                        st.session_state.messages = []
                        
                        if "pdf_file" in st.session_state:
                            del st.session_state.pdf_file
                        
                        with st.spinner("Generating dataset insights…"):
                            st.session_state.insights = DataInsightAgent(st.session_state.df)
                    except Exception as e:
                        st.error(f"❌ Error loading CSV file: {str(e)}\n\nPlease check your file format.")
                        return
                
                elif "insights" not in st.session_state:
                    with st.spinner("Generating dataset insights…"):
                        st.session_state.insights = DataInsightAgent(st.session_state.df)
                
                if "df" in st.session_state:
                    st.markdown("### Dataset Insights")
                    
                    # Show dataset info
                    with st.expander("📊 Dataset Info", expanded=False):
                        st.write(f"**Shape:** {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns")
                        st.write(f"**Memory:** {st.session_state.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                        st.write(f"**Missing values:** {st.session_state.df.isnull().sum().sum():,}")
                    
                    if "insights" in st.session_state and st.session_state.insights and not st.session_state.insights.startswith("Error"):
                        st.dataframe(st.session_state.df.head())
                        st.markdown(st.session_state.insights)
                        
                        current_config_left = get_current_config()
                        st.markdown(f"*<span style='color: grey; font-style: italic;'>Generated with {current_config_left.MODEL_PRINT_NAME}</span>*", unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Could not generate insights. You can still chat with the data.")
            
            else:  # PDF
                if ("pdf_file" not in st.session_state) or file_changed:
                    try:
                        file.seek(0)
                        st.session_state.pdf_file = io.BytesIO(file.read())
                        st.session_state.current_file = file.name
                        st.session_state.messages = []
                        st.session_state.pdf_use_vision = use_vision
                        
                        if "df" in st.session_state:
                            del st.session_state.df
                        
                        with st.spinner("Analyzing PDF document…"):
                            st.session_state.insights = PDFInsightAgent(st.session_state.pdf_file, use_vision=use_vision)
                    except Exception as e:
                        st.error(f"❌ Error loading PDF file: {str(e)}\n\nPlease check your file.")
                        return
                
                elif "insights" not in st.session_state:
                    with st.spinner("Analyzing PDF document…"):
                        pdf_use_vision = st.session_state.get("pdf_use_vision", True)
                        st.session_state.insights = PDFInsightAgent(st.session_state.pdf_file, use_vision=pdf_use_vision)
                
                if "pdf_file" in st.session_state:
                    st.markdown("### PDF Document Insights")
                    
                    if "insights" in st.session_state and st.session_state.insights and not st.session_state.insights.startswith("Error"):
                        st.info(f"📄 **Document:** {st.session_state.current_file}")
                        st.markdown(st.session_state.insights)
                        
                        current_config_left = get_current_config()
                        processing_method = "Claude Vision API" if st.session_state.get("pdf_use_vision", True) else "Text Extraction"
                        st.markdown(f"*<span style='color: grey; font-style: italic;'>Generated with {current_config_left.MODEL_PRINT_NAME} ({processing_method})</span>*", unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Could not generate insights. You can still chat about the document.")
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
                    plt.close('all')  # Clean up any lingering figures
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
                
                # Create a status container for verbose feedback
                status_container = st.empty()
                
                with st.spinner("Working…"):
                    recent_user_turns = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-3:]
                    context_text = "\n".join(recent_user_turns[:-1]) if len(recent_user_turns) > 1 else None
                    
                    max_retries = 3
                    retry_count = 0
                    result_obj = None
                    code = None
                    execution_error = None
                    previous_errors = []
                    should_plot_flag = False
                    code_thinking = ""
                    
                    import time
                    start_time = time.time()
                    
                    while retry_count < max_retries:
                        try:
                            elapsed = time.time() - start_time
                            if elapsed > TOTAL_RETRY_TIMEOUT:
                                status_container.error(f"⏱️ Total execution timed out after {TOTAL_RETRY_TIMEOUT} seconds")
                                result_obj = f"Error: Total execution timed out after {TOTAL_RETRY_TIMEOUT} seconds. Your query may be too complex."
                                break
                            
                            if retry_count == 0:
                                status_container.info("🤖 Generating code...")
                                code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df, context_text)
                                
                                # Validate syntax before first execution
                                is_valid, syntax_error = validate_python_syntax(code)
                                if not is_valid:
                                    status_container.warning(f"⚠️ Initial code has syntax error, regenerating...")
                                    execution_error = f"Syntax error: {syntax_error}"
                                    previous_errors.append(execution_error)
                                    retry_count += 1
                                    continue
                                
                                # Check for dangerous patterns
                                is_safe, safety_error = analyze_code_safety(code, st.session_state.df)
                                if not is_safe:
                                    status_container.error(f"⚠️ Unsafe code detected: {safety_error}")
                                    execution_error = safety_error
                                    previous_errors.append(execution_error)
                                    retry_count += 1
                                    continue
                                
                                status_container.info(f"⚙️ Executing code (timeout: {CODE_EXECUTION_TIMEOUT}s)...")
                            else:
                                status_container.warning(f"🔄 Attempt {retry_count + 1}/{max_retries}: Fixing code...")
                                code = CodeFixingAgent(
                                    user_q, 
                                    code, 
                                    execution_error, 
                                    st.session_state.df, 
                                    should_plot_flag, 
                                    retry_count,
                                    previous_errors
                                )
                                
                                # Validate fixed code syntax
                                is_valid, syntax_error = validate_python_syntax(code)
                                if not is_valid:
                                    status_container.warning(f"⚠️ Fixed code still has syntax error, retrying...")
                                    execution_error = f"Syntax error in fixed code: {syntax_error}"
                                    previous_errors.append(execution_error)
                                    retry_count += 1
                                    continue
                                
                                status_container.info("⚙️ Re-executing fixed code...")
                            
                            result_obj, execution_error = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                            
                            if execution_error is None:
                                status_container.success("✅ Code executed successfully!")
                                time.sleep(0.5)  # Brief pause to show success
                                status_container.empty()
                                break
                            
                            # Track this error
                            previous_errors.append(execution_error)
                            retry_count += 1
                            
                            if retry_count >= max_retries:
                                status_container.error(f"❌ Failed after {max_retries} attempts")
                                result_obj = f"Error: Code execution failed after {max_retries} attempts.\n\nLast error:\n{execution_error}\n\nSuggestions:\n- Try simplifying your query\n- Be more specific about what you want\n- Break complex requests into smaller steps\n- Try asking 'Show me the first 10 rows' to verify data structure"
                        
                        except ExecutionError as e:
                            status_container.error(f"❌ Execution error: {str(e)}")
                            result_obj = f"Error: {str(e)}"
                            execution_error = str(e)
                            break
                        
                        except Exception as e:
                            tb = traceback.format_exc()
                            status_container.error(f"❌ Unexpected error: {str(e)}")
                            st.error(f"**Detailed traceback:**\n```\n{tb}\n```")
                            result_obj = f"Unexpected error: {str(e)}"
                            execution_error = str(e)
                            break
                    
                    status_container.empty()
                    
                    if result_obj is None:
                        result_obj = "Error: No result generated"
                    
                    try:
                        serialized_result = serialize_result(result_obj)
                    except Exception as e:
                        serialized_result = f"Error serializing result: {str(e)}"
                        st.error(f"Serialization error: {str(e)}")
                    
                    try:
                        raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj, serialized_result)
                        reasoning_txt = reasoning_txt.replace("<think>", "").replace("</think>", "")
                    except Exception as e:
                        reasoning_txt = f"Could not generate explanation: {str(e)}"
                        st.error(f"Reasoning error: {str(e)}")
                    
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
                    try:
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
                            
                            if pdf_text.startswith("Error"):
                                answer = f"❌ Could not process PDF: {pdf_text}"
                                st.session_state.pdf_messages.append({"role": "assistant", "content": answer})
                                st.rerun()
                                return
                            
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
                    API_KEY = os.environ.get("ANTHROPIC_API_KEY")
                    except Exception as e:
                        tb = traceback.format_exc()
                        error_msg = f"❌ Error processing your question:\n{str(e)}\n\nPlease try rephrasing your question."
                        st.error(f"**Detailed error:**\n```\n{tb}\n```")
                        st.session_state.pdf_messages.append({"role": "assistant", "content": error_msg})
                
                st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Fatal error in application:\n{str(e)}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")
        st.info("Please refresh the page and try again.")