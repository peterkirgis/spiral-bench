# Browser Automation Module

This folder contains browser automation scripts for running conversations with ChatGPT using Playwright.

## Files

- **browser_conversation_runner.py** - Main browser automation client for ChatGPT conversations
- **test_browser_runner.py** - Test script to verify browser automation works
- **chatgpt_cookies.json** - Authentication cookies for ChatGPT (not committed to git)
- **test_browser_transcript.json** - Sample output from test runs

## Setup

1. Install Playwright:
   ```bash
   pip install playwright
   playwright install chromium
   ```

2. Get your ChatGPT cookies:
   - Log into ChatGPT in your browser
   - Open Developer Tools (F12)
   - Go to Application > Cookies
   - Export cookies to `chatgpt_cookies.json`

## Usage

### Running the test from root directory:

```bash
python -m browser_automation.test_browser_runner
```

### Using in your own code:

```python
from browser_automation.browser_conversation_runner import run_browser_conversation

result = await run_browser_conversation(
    user_model="moonshotai/kimi-k2",
    user_system_prompt="You are helpful assistant",
    canned_prompts=["Hello!"],
    num_turns=3,
    cookies_file="browser_automation/chatgpt_cookies.json",  # Relative to root
    headless=False
)
```

## Notes

- The `cookies_file` path is relative to where you run the script from
- If running from root, use `browser_automation/chatgpt_cookies.json`
- If running from this folder, use `chatgpt_cookies.json`
