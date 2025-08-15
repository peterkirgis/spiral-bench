import os
import requests
import time
import logging
import json

# Configure logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

def print_messages(messages):
    """
    Pretty-print a messages list with roles, names, and escaped newlines rendered.
    """
    print("\n=== Messages ===")
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "<no role>")
        name = msg.get("name")
        header = f"[{i}] Role: {role}"
        if name:
            header += f" | Name: {name}"
        print(header)
        print("-" * len(header))

        content = msg.get("content", "")
        if isinstance(content, str):
            # Render escaped chars like \n as real line breaks
            try:
                content = content.encode("utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                pass
            print(content.strip())
        elif isinstance(content, list):
            # Handle multi-part content (e.g., tool calls + text)
            for part in content:
                if part.get("type") == "text":
                    text_val = part.get("text", "")
                    try:
                        text_val = text_val.encode("utf-8").decode("unicode_escape")
                    except UnicodeDecodeError:
                        pass
                    print(text_val.strip())
                else:
                    print(json.dumps(part, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(content, indent=2, ensure_ascii=False))

        print()  # Blank line between messages


def get_completion(
    model: str,
    messages: list,
    api_key: str,
    base_url: str,
    site_url: str,
    max_retries: int = 7,
    backoff_factor: float = 2.0,
    max_tokens: int = 3072,
):
    """
    Gets a completion from a generic OpenAI-compatible API with retries.
    """
    if not api_key:
        raise APIError(f"API key is missing for model {model}.")
        
    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": site_url,
        "X-Title": "Automated Red Teaming Pipeline",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    if base_url == 'https://api.openai.com/v1/chat/completions':
                del payload['min_p']

    if model in ['openai/gpt-oss-120b', 'openai/gpt-oss-20b']:
        payload['provider'] =  {
            "order": [
                "DeepInfra"
            ],
            "allow_fallbacks": False
        }
        payload['reasoning'] = {
            "effort": "low",
        }
    if model == "openai/gpt-oss-120b":
        payload['max_tokens'] = 8096 # let it cook
    if model == "o3":
        del payload["max_tokens"]
        del payload["temperature"]
        payload["max_completion_tokens"] = 16000

    if model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
        payload['reasoning_effort']="minimal"
        del payload['max_tokens']
        payload["max_completion_tokens"] = 16000
        payload['temperature'] = 1

    if model in ['gpt-5-chat-latest']:
        del payload['max_tokens']
        payload["max_completion_tokens"] = 16000
        payload['temperature'] = 1
    
    if model == "deepseek/deepseek-r1-0528":
        payload['max_tokens'] = 32000

    if model == "google/gemini-2.5-pro":
        payload['reasoning'] = {
            "max_tokens": 1,
        }
    if model == "openai/o4-mini":
        payload['reasoning'] = {
            "effort": "low",
        }

    #if model == "moonshotai/kimi-k2" and base_url == "https://openrouter.ai/api":
    #    payload["provider"] = {
    #        "order": ["Chutes"],     # fast qwen-2-35B
    #        "allow_fallbacks": False,
    #    }
    #print(messages)

    #print_messages(messages)
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            
            # Enhanced instrumentation for non‑2xx responses
            if resp.status_code >= 400:
                try:
                    resp_body = resp.json()
                except ValueError:
                    resp_body = resp.text or "<empty body>"
                logging.error(
                    "HTTP %s from %s\n"
                    "Request payload:\n%s\n"
                    "Response body:\n%s",
                    resp.status_code,
                    api_url,
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    json.dumps(resp_body, indent=2, ensure_ascii=False)
                    if isinstance(resp_body, (dict, list))
                    else resp_body,
                )
            
            resp.raise_for_status()
            data = resp.json()
            
            if "error" in data:
                raise APIError(f"API error: {data['error']}")
            
            choices = data.get("choices", [])
            if not choices:
                raise APIError("No choices in API response")
            
            first_choice = choices[0]
            message = first_choice.get("message", {})
            content = message.get("content", "")

            if '<|reserved_token_163839|>' in content:
                # this is a kimi-k2 issue that occurs sometimes. retry.
                raise APIError("Garbage tokens in output")
            
            finish_reason = first_choice.get("finish_reason")
            #if finish_reason == "content_filter":
            #    raise APIError(f"Content filtered by {model}")
            if finish_reason == "length":
                logging.warning(f"Response truncated due to length limit for {model}")
            
            if content and content.strip():
                return content
            
            raise APIError("Received empty content")
            
        except requests.exceptions.RequestException as req_err:
            logging.warning(f"[{model}] attempt {attempt+1}/{max_retries} failed: {req_err}")
            if attempt + 1 == max_retries:
                raise APIError(f"Giving up after {max_retries} tries: {req_err}") from req_err
            time.sleep(backoff_factor ** attempt)
        except APIError as api_err:
            logging.warning(f"[{model}] attempt {attempt+1}/{max_retries} failed: {api_err}")
            if attempt + 1 == max_retries:
                raise APIError(f"Giving up after {max_retries} tries: {api_err}") from api_err
            time.sleep(backoff_factor ** attempt)
        except Exception as err:
            logging.error(f"Unexpected error during API call for {model}: {err}", exc_info=True)
            if attempt + 1 == max_retries:
                raise APIError(f"Giving up after {max_retries} tries due to unexpected error: {err}") from err
            time.sleep(backoff_factor ** attempt)
    
    raise APIError("Unexpected fall-through in retry loop")
