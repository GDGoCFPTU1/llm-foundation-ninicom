"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from urllib import response
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    def call_openai(
        prompt: str,
        model: str = OPENAI_MODEL,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> tuple[str, float, dict]:

        start = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        latency = time.time() - start

        response_text = response.choices[0].message.content or ""

        usage_stats = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        return response_text, latency, usage_stats


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    
    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    latency_seconds = time.time() - start_time
    
    usage = {
        'input_tokens': response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
        'output_tokens': response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    }
    
    return response.text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    latency_seconds = time.time() - start_time
    
    usage = {
        'input_tokens': response.usage.input_tokens,
        'output_tokens': response.usage.output_tokens
    }
    
    response_text = response.content[0].text if response.content else ""
    
    return response_text, latency_seconds, usage

# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # TODO: Call call_openai with default gpt-4o model
    # TODO: Call call_openai with gpt-4o-mini model
    # TODO: Call call_gemini with default gemini-2.5-flash model
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.
    gpt4o_resp, gpt4o_lat, gpt4o_usage = retry_with_backoff(
        lambda: call_openai(prompt, model=OPENAI_MODEL)
    )
    gpt4o_in = gpt4o_usage.get("input_tokens", 0)
    gpt4o_out = gpt4o_usage.get("output_tokens", 0)
    gpt4o_cost = (
        (gpt4o_in * PRICING_1M_TOKENS[OPENAI_MODEL]["input"]) + 
        (gpt4o_out * PRICING_1M_TOKENS[OPENAI_MODEL]["output"])
    ) / 1_000_000

    # TODO: Call call_openai with gpt-4o-mini model (có bọc retry)
    mini_resp, mini_lat, mini_usage = retry_with_backoff(
        lambda: call_openai(prompt, model=OPENAI_MINI_MODEL)
    )
    mini_in = mini_usage.get("input_tokens", 0)
    mini_out = mini_usage.get("output_tokens", 0)
    mini_cost = (
        (mini_in * PRICING_1M_TOKENS[OPENAI_MINI_MODEL]["input"]) + 
        (mini_out * PRICING_1M_TOKENS[OPENAI_MINI_MODEL]["output"])
    ) / 1_000_000

    # TODO: Call call_gemini with default gemini-2.5-flash model (có bọc retry)
    gemini_resp, gemini_lat, gemini_usage = retry_with_backoff(
        lambda: call_gemini(prompt, model=GEMINI_MODEL)
    )
    gemini_in = gemini_usage.get("input_tokens", 0)
    gemini_out = gemini_usage.get("output_tokens", 0)
    gemini_cost = (
        (gemini_in * PRICING_1M_TOKENS[GEMINI_MODEL]["input"]) + 
        (gemini_out * PRICING_1M_TOKENS[GEMINI_MODEL]["output"])
    ) / 1_000_000

    # TODO: Assemble and return the comparison dictionary.
    return {
        "gpt4o": {
            "response": gpt4o_resp,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_in,
            "output_tokens": gpt4o_out
        },
        "gpt4o_mini": {
            "response": mini_resp,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_in,
            "output_tokens": mini_out
        },
        "gemini_flash": {
            "response": gemini_resp,
            "latency": gemini_lat,
            "cost": gemini_cost,
            "input_tokens": gemini_in,
            "output_tokens": gemini_out
        }
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    # TODO: Setup interactive session, prompt user for input, stream response, and update history.
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                print("Exiting chatbot.")
                break
            
            if not user_input.strip():
                continue

            history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

            if len(history) > 6:
                history = history[-6:]

            print("Gemini: ", end="", flush=True)
            
            response_stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=history
            )

            full_response = ""
            for chunk in response_stream:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text

            print()
            
            history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_response)]))

        except Exception as e:
            print(f"\n[Lỗi kết nối hoặc API]: {e}")
            if history and history[-1].role == "user":
                history.pop()


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    delay = base_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                
    raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
    results = []
    for prompt in prompts:
        # Gọi hàm compare_models (đã viết ở Task 4)
        comparison_result = compare_models(prompt)
        # Chèn thêm prompt gốc vào dictionary kết quả
        comparison_result["prompt"] = prompt
        results.append(comparison_result)
        
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|"
    ]

    for res in results:
        raw_prompt = res.get("prompt", "")
        clean_prompt = raw_prompt.replace("\n", " ").replace("\r", "")
        trunc_prompt = clean_prompt[:47] + "..." if len(clean_prompt) > 50 else clean_prompt

        for model_key in ["gpt4o", "gpt4o_mini", "gemini_flash"]:
            if model_key not in res:
                continue
                
            stats = res[model_key]
            
            raw_resp = stats.get("response", "")
            clean_resp = raw_resp.replace("\n", " ").replace("\r", "")
            trunc_resp = clean_resp[:47] + "..." if len(clean_resp) > 50 else clean_resp
            
            latency = f"{stats.get('latency', 0):.2f}s"
            tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
            cost = f"${stats.get('cost', 0):.6f}"

            lines.append(f"| {trunc_prompt} | {model_key} | {trunc_resp} | {latency} | {tokens} | {cost} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            if model_name == "prompt": continue
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
            
        print("\n=== Testing Batch Compare & Markdown Table ===")
        batch_results = batch_compare(["1+1 bằng mấy?", "Kể một chuyện cười ngắn."])
        print(format_comparison_table(batch_results))
        
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
