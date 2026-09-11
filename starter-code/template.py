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

from dotenv import load_dotenv

load_dotenv() # Lệnh này sẽ tự động tìm file .env và nạp API key vào hệ thống

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-3.6-flash": {"input": 0.075, "output": 0.300},
    "gemini-3.1-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.6-flash"
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
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "mock-key"

    client = OpenAI(api_key=api_key)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    
    return text, latency, usage


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
    
    Supports dual-import fallback (new google-genai and legacy google-generativeai)
    to ensure zero-friction execution.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    start_time = time.time()
    
    try:
        # Option A: New Google GenAI SDK (preferred standard)
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        latency = time.time() - start_time
        
        text = response.text or ""
        usage = {
            "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
        }
        return text, latency, usage
        
    except (ImportError, Exception):
        # Option B: Fallback to legacy google-generativeai SDK
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model_inst = genai.GenerativeModel(model)
        config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )
        response = model_inst.generate_content(prompt, generation_config=config)
        latency = time.time() - start_time
        
        text = response.text or ""
        try:
            # Retrieve token counts from legacy API
            input_tokens = model_inst.count_tokens(prompt).total_tokens
            output_tokens = model_inst.count_tokens(text).total_tokens
        except Exception:
            # Fallback heuristic calculation if token service fails
            input_tokens = int(len(prompt.split()) * 1.5)
            output_tokens = int(len(text.split()) * 1.5)
            
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        return text, latency, usage


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
    import anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"
    client = anthropic.Anthropic(api_key=api_key)
    
    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time
    
    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }
    
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-3.6-flash)
    with the same prompt and return a structured comparison dictionary.
    """
    # Call GPT-4o
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"] +
        gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000
    
    # Call GPT-4o-mini
    mini_text, mini_lat, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    mini_cost = (
        mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"] +
        mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000
    
    # Call Gemini 2.5 Flash
    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gemini_cost = (
        gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-3.6-flash"]["input"] +
        gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-3.6-flash"]["output"]
    ) / 1_000_000
    
    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"]
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"]
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"]
        }
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
    Maintains the last 3 turns of conversation history for context.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set. Running in dummy mode.\033[0m")
        api_key = "mock-key"
        
    print("\n\033[94m==============================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\033[0m\n")
    
    # Store history as a list of dictionaries with 'role' and 'text'
    # Gemini 2.5 structure generally expects 'user' and 'model' roles.
    history = []
    
    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\033[93mGoodbye!\033[0m")
                break
                
            # Compile messages from history (trimmed to last 3 turns / 6 messages)
            messages_to_send = []
            for h in history[-6:]:
                messages_to_send.append(h)
            messages_to_send.append({"role": "user", "text": user_input})
            
            print("\033[92mGemini:\033[0m ", end="", flush=True)
            
            full_reply = ""
            try:
                # Option A: New GenAI client streaming
                from google import genai
                from google.genai import types
                
                client = genai.Client(api_key=api_key)
                
                # Format messages for the new SDK
                # For chat history, new SDK prefers client.chats.create() or genai.types.Content structure
                formatted_contents = []
                for msg in messages_to_send:
                    formatted_contents.append(
                        types.Content(
                            role=msg["role"],
                            parts=[types.Part.from_text(text=msg["text"])]
                        )
                    )
                    
                response_stream = client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=formatted_contents
                )
                for chunk in response_stream:
                    chunk_text = chunk.text or ""
                    print(chunk_text, end="", flush=True)
                    full_reply += chunk_text
                    
            except (ImportError, Exception):
                # Option B: Fallback to legacy google-generativeai stream
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_inst = genai.GenerativeModel(GEMINI_MODEL)
                
                # Format legacy messages: role is 'user' or 'model'
                legacy_contents = []
                for msg in messages_to_send:
                    legacy_contents.append({
                        "role": msg["role"] if msg["role"] == "user" else "model",
                        "parts": [msg["text"]]
                    })
                    
                response_stream = model_inst.generate_content(legacy_contents, stream=True)
                for chunk in response_stream:
                    chunk_text = chunk.text or ""
                    print(chunk_text, end="", flush=True)
                    full_reply += chunk_text
                    
            print("\n")
            
            # Record the turn in history
            history.append({"role": "user", "text": user_input})
            history.append({"role": "model", "text": full_reply})
            
        except KeyboardInterrupt:
            print("\n\033[93mSession interrupted. Goodbye!\033[0m")
            break
        except Exception as e:
            print(f"\n\033[91m[Error Calling API]: {e}\033[0m\n")


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
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    if last_exception:
        raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    """
    results = []
    for prompt in prompts:
        try:
            comp = compare_models(prompt)
            comp["prompt"] = prompt
            results.append(comp)
        except Exception as e:
            # Return partial mock results for error resistance during testing
            results.append({
                "prompt": prompt,
                "gpt4o": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
                "gpt4o_mini": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
                "gemini_flash": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0}
            })
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
    """
    table_lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for r in results:
        prompt = r["prompt"]
        if len(prompt) > 40:
            prompt = prompt[:37] + "..."
            
        models = [
            ("GPT-4o", r.get("gpt4o", r.get("gpt4o_response"))),
            ("GPT-4o-Mini", r.get("gpt4o_mini", r.get("mini_response"))),
            ("Gemini-Flash", r.get("gemini_flash"))
        ]
        
        for name, data in models:
            if not data:
                continue
            
            # Support both structure types (nested dict vs legacy flat dict)
            if isinstance(data, dict):
                resp = data.get("response", "")
                lat = data.get("latency", 0.0)
                in_t = data.get("input_tokens", 0)
                out_t = data.get("output_tokens", 0)
                cost = data.get("cost", 0.0)
            else:
                # Legacy flat dict structure mapping (for backwards compatibility if needed)
                resp = str(data)
                lat = r.get("gpt4o_latency" if name == "GPT-4o" else "mini_latency", 0.0)
                in_t = int(len(prompt.split()) * 1.5)
                out_t = int(len(resp.split()) * 1.5)
                cost = r.get("gpt4o_cost_estimate" if name == "GPT-4o" else "gpt4o_cost_estimate", 0.0)
                
            resp_trunc = resp.replace("\n", " ")
            if len(resp_trunc) > 50:
                resp_trunc = resp_trunc[:47] + "..."
                
            table_lines.append(
                f"| {prompt} | {name} | {resp_trunc} | {lat:.2f}s | {in_t}/{out_t} | ${cost:.6f} |"
            )
            
    return "\n".join(table_lines)

if __name__ == "__main__":
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")