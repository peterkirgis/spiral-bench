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
   - Go to Application > Cookies > https://chatgpt.com
   - Export all cookies to `browser_automation/chatgpt_cookies.json`
   - You can use a browser extension like "EditThisCookie" or "Cookie-Editor"

3. **Calibrate ChatGPT UI (REQUIRED):**
   ```bash
   python -m browser_automation.calibrate_chatgpt
   ```

   **How it works:**
   - Browser opens → ChatGPT loads
   - Script tells you what to hover over
   - You hover and hold still for 3 seconds
   - Position auto-captures
   - Repeat for 4 elements

   **What you'll record:**
   1. Dropdown button
   2. "Legacy models" row
   3. "GPT-5 Instant" (in submenu)
   4. "GPT-4o" (in submenu)

   **Important:** Just hover - don't click or press keys!

   **Run once per computer.**

## Usage

### Running the basic test:

```bash
python -m browser_automation.test_browser_runner
```

### Cycling through multiple models:

Test multiple ChatGPT models automatically:

```bash
# Test both GPT-5 and GPT-4o with default settings
python -m browser_automation.cycle_models_test --models gpt-5 gpt-4o

# Custom configuration
python -m browser_automation.cycle_models_test \
  --models gpt-5 gpt-4o \
  --turns 5 \
  --prompt "Explain black holes in simple terms" \
  --delay 10 \
  --output-dir results/model_comparison

# Run in headless mode (no browser window)
python -m browser_automation.cycle_models_test --models gpt-5 gpt-4o --headless
```

**Available options:**
- `--models` - Models to test (e.g., gpt-5, gpt-4o, instant)
- `--turns` - Number of conversation turns per model (default: 3)
- `--prompt` - Initial prompt to send (default: quantum entanglement question)
- `--delay` - Delay between turns in seconds (default: 5)
- `--headless` - Run browser in headless mode
- `--output-dir` - Directory to save results (default: browser_automation/model_test_results)
- `--cookies` - Path to cookies file (default: browser_automation/chatgpt_cookies.json)

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
