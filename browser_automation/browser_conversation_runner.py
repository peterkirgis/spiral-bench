"""
Browser-based conversation runner for ChatGPT.
Parallel to conversation_runner.py but uses browser automation instead of API.

Matches the same interface and patterns as conversation_runner.py for easy integration.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeoutError

from pipeline.api_client import get_completion, APIError


@dataclass
class ConversationResult:
    """Same as conversation_runner.ConversationResult"""
    transcript: List[Dict[str, str]]
    errors: List[Dict[str, str]]
    injections_log: List[str]


SaveCB = Optional[Callable[[ConversationResult], None]]


# Stable selectors for ChatGPT (as of 2025)
CHATGPT_SELECTORS = {
    'textarea': '#prompt-textarea',
    'send_button': 'button[data-testid="send-button"]',
    'assistant_message': '[data-message-author-role="assistant"]',
    'stop_button': 'button[aria-label="Stop generating"]',
    'new_chat_button': 'a[href="/"]',  # New chat link in sidebar
    'model_dropdown': 'button[data-testid="model-switcher"]',  # Model selector button
}


class ChatGPTBrowserClient:
    """
    Browser automation client for ChatGPT.
    Mimics the interface of api_client.get_completion() but uses Playwright.
    """

    def __init__(self, cookies_file: str = "chatgpt_cookies.json", headless: bool = False):
        self.cookies_file = cookies_file
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def start(self):
        """Initialize browser and load ChatGPT"""
        print("Starting browser...")
        self.playwright = await async_playwright().start()

        # Launch browser
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ]
        )

        # Load cookies for authentication
        import json
        with open(self.cookies_file, 'r') as f:
            cookies = json.load(f)

        # Normalize cookies for Playwright
        # Playwright expects specific format and capitalized sameSite values
        normalized_cookies = []
        for cookie in cookies:
            normalized = {
                'name': cookie['name'],
                'value': cookie['value'],
                'domain': cookie['domain'],
                'path': cookie['path'],
                'secure': cookie.get('secure', False),
                'httpOnly': cookie.get('httpOnly', False),
            }

            # Handle sameSite (capitalize if needed)
            if 'sameSite' in cookie:
                same_site = cookie['sameSite']
                if isinstance(same_site, str):
                    # Capitalize first letter: lax -> Lax, strict -> Strict, none -> None
                    same_site = same_site.capitalize()
                    # Handle special case: no_restriction -> None
                    if same_site == 'No_restriction':
                        same_site = 'None'
                    normalized['sameSite'] = same_site

            # Handle expiration (optional, -1 means session cookie)
            if 'expirationDate' in cookie and cookie['expirationDate'] != -1:
                normalized['expires'] = cookie['expirationDate']

            normalized_cookies.append(normalized)

        # Create context with cookies
        context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )

        # Add cookies to context
        await context.add_cookies(normalized_cookies)

        self.page = await context.new_page()

        # Navigate to ChatGPT
        print("Navigating to ChatGPT...")
        try:
            await self.page.goto('https://chatgpt.com', wait_until='networkidle', timeout=30000)
        except PlaywrightTimeoutError:
            # If networkidle times out, try with less strict condition
            print("  Networkidle timeout, retrying with domcontentloaded...")
            await self.page.goto('https://chatgpt.com', wait_until='domcontentloaded', timeout=30000)
        await asyncio.sleep(2)

        # Verify we're logged in
        try:
            await self.page.wait_for_selector(CHATGPT_SELECTORS['textarea'], timeout=10000)
            print("✓ Successfully loaded ChatGPT (logged in)")
        except PlaywrightTimeoutError:
            raise Exception("Failed to load ChatGPT - check cookies file and authentication")

    async def start_new_thread(self):
        """Start a fresh ChatGPT conversation thread"""
        try:
            # Click the "New chat" button/link
            await self.page.click(CHATGPT_SELECTORS['new_chat_button'], timeout=3000)
            await asyncio.sleep(1)
            print("Started new conversation thread")
        except PlaywrightTimeoutError:
            # If button not found, just continue (might already be on new thread)
            print("Note: Could not click new chat button, continuing anyway")

    async def select_model(self, model: str):
        """
        Prompt user to manually select a ChatGPT model before starting conversation.

        Args:
            model: Model to select. Options:
                - "gpt-5" - User should select "Instant" model
                - "gpt-4o" - User should select GPT-4o from Legacy models
                - None or "default" - Skip model selection (use default)
        """
        if not model or model.lower() == "default":
            print("Using default model (no manual selection required)")
            return

        # Map to user-friendly model names
        model_instructions = {
            "gpt-5": "GPT-5 (Instant)",
            "gpt-4o": "GPT-4o (from Legacy models dropdown)"
        }

        model_name = model_instructions.get(model.lower(), model)

        print("\n" + "="*70)
        print("MANUAL MODEL SELECTION REQUIRED")
        print("="*70)
        print(f"Please manually select the model: {model_name}")
        print("Steps:")
        print("  1. Click the model dropdown button at the top of the page")
        if model.lower() == "gpt-4o":
            print("  2. Click 'Legacy models'")
            print("  3. Select 'GPT-4o'")
        elif model.lower() == "gpt-5":
            print("  2. Select 'Instant'")
        print("\nOnce you've selected the model, press ENTER to continue...")
        print("="*70 + "\n")

        # Wait for user to press Enter
        input()

        print("✓ Continuing with conversation...")
    async def send_message_and_get_response(self, message: str, turn_index: int = 0) -> str:
        """
        Send a message to ChatGPT and wait for response.

        Args:
            message: User message to send
            turn_index: Turn number (for logging/debugging)

        Returns:
            Assistant's response text

        Raises:
            Exception: If sending fails or response cannot be extracted
        """
        print(f"\n[Turn {turn_index}] Sending message ({len(message)} chars)...")

        try:
            # Wait a moment to ensure UI is ready
            await asyncio.sleep(2)

            # Find and fill textarea
            textarea = await self.page.wait_for_selector(CHATGPT_SELECTORS['textarea'], timeout=10000)
            await textarea.fill(message)
            await asyncio.sleep(1)

            # Click send button (with retry logic)
            send_button = None
            for attempt in range(3):
                try:
                    send_button = await self.page.wait_for_selector(CHATGPT_SELECTORS['send_button'], timeout=5000)
                    if send_button:
                        # Check if button is enabled
                        is_disabled = await send_button.is_disabled()
                        if not is_disabled:
                            break
                        else:
                            print(f"  Send button disabled, waiting (attempt {attempt + 1}/3)...")
                            await asyncio.sleep(2)
                except PlaywrightTimeoutError:
                    print(f"  Send button not found, retrying (attempt {attempt + 1}/3)...")
                    await asyncio.sleep(2)

            if not send_button:
                raise Exception("Could not find enabled send button after 3 attempts")

            await send_button.click()
            print("Message sent, waiting for response...")

            # Wait for response to complete
            await self._wait_for_response_completion(turn_index)

            # Extract response text
            response_text = await self._extract_last_assistant_message()

            print(f"✓ Received response ({len(response_text)} chars)")
            return response_text

        except Exception as e:
            print(f"✗ Error in turn {turn_index}: {e}")
            # Take screenshot for debugging
            try:
                await self.page.screenshot(path=f"screenshots/error_turn_{turn_index}.png")
            except:
                pass
            raise

    async def _wait_for_response_completion(self, turn_index: int, timeout: int = 120):
        """
        Wait for ChatGPT to finish generating response.
        Uses the stop button as indicator.
        """
        try:
            # Wait for "Stop generating" button to appear (generation started)
            await self.page.wait_for_selector(
                CHATGPT_SELECTORS['stop_button'],
                timeout=10000,
                state='visible'
            )
            print("  Generation started...")

            # Wait for button to disappear (generation completed)
            await self.page.wait_for_selector(
                CHATGPT_SELECTORS['stop_button'],
                timeout=timeout * 1000,
                state='hidden'
            )
            print("  Generation complete")

        except PlaywrightTimeoutError:
            # Stop button didn't appear or didn't disappear
            # This can happen for very short responses - wait a bit and continue
            print("  (Response may have been instant, continuing)")
            await asyncio.sleep(3)

        # Extra moment for DOM to settle
        await asyncio.sleep(1)

    async def _extract_last_assistant_message(self) -> str:
        """Extract the most recent assistant message from the page"""
        # Get all assistant message elements
        messages = await self.page.query_selector_all(CHATGPT_SELECTORS['assistant_message'])

        if not messages:
            raise Exception("No assistant messages found on page")

        # Get the last one (most recent)
        last_message = messages[-1]

        # Extract text from the markdown content
        markdown = await last_message.query_selector('.markdown')
        if markdown:
            text = await markdown.inner_text()
        else:
            # Fallback: get text from the whole message element
            text = await last_message.inner_text()

        return text.strip()

    async def close(self):
        """Clean shutdown"""
        print("\nClosing browser...")
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def run_browser_conversation(
    user_model: str,
    user_system_prompt: str,
    canned_prompts: List[Optional[str]],
    num_turns: int,
    user_agent_api_key: str,
    user_agent_base_url: str,
    site_url: str,
    max_retries: int,
    backoff_factor: float,
    save_turn_callback: SaveCB = None,
    injections: Optional[List[str]] = None,
    injection_frequency: int = 5,
    seed: Optional[str] = None,
    resume_state: Optional[ConversationResult] = None,
    cookies_file: str = "chatgpt_cookies.json",
    headless: bool = False,
    turn_delay_seconds: int = 30,
    chatgpt_model: Optional[str] = None,
) -> ConversationResult:
    """
    Run a dialogue with ChatGPT via browser automation.

    Interface matches conversation_runner.run_conversation() for easy integration.

    Args:
        user_model: Model to use for generating user messages (via API)
        user_system_prompt: System prompt for user agent
        canned_prompts: List of scripted prompts (first must be non-empty)
        num_turns: Number of assistant turns to generate
        user_agent_api_key: API key for user agent model
        user_agent_base_url: Base URL for user agent API
        site_url: Site URL for API
        max_retries: Max retries for user agent API calls
        backoff_factor: Backoff factor for user agent API retries
        save_turn_callback: Callback to save progress after each turn
        injections: Optional list of prompt injections
        injection_frequency: How often to inject (1/N probability)
        seed: Random seed for reproducibility
        resume_state: Optional previous state to resume from
        cookies_file: Path to ChatGPT cookies JSON file
        headless: Run browser in headless mode
        turn_delay_seconds: Delay between turns (to avoid rate limiting)
        chatgpt_model: ChatGPT model to use ("gpt-5" for Instant, "gpt-4o" for GPT-4o, None for default)

    Returns:
        ConversationResult with transcript, errors, and injections_log
    """

    # --- bootstrap or resume ---
    if resume_state and resume_state.transcript:
        transcript = resume_state.transcript.copy()
        errors = resume_state.errors.copy()
        injections_log = resume_state.injections_log.copy()
    else:
        initial_user_message = canned_prompts[0]
        if not (isinstance(initial_user_message, str) and initial_user_message.strip()):
            raise AssertionError("Initial canned prompt (index 0) must be a non-empty string.")
        transcript = [{"role": "user", "content": initial_user_message}]
        errors = []
        injections_log = [""]  # keep 1:1 with transcript

    # Align injections_log with transcript
    if len(injections_log) < len(transcript):
        injections_log.extend([""] * (len(transcript) - len(injections_log)))
    elif len(injections_log) > len(transcript):
        injections_log = injections_log[:len(transcript)]

    rnd = random.Random(seed)

    # Initial save
    if save_turn_callback:
        save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))

    # How many assistant turns already happened
    turns_already_done = sum(1 for m in transcript if m["role"] == "assistant")
    if turns_already_done >= num_turns:
        return ConversationResult(transcript, errors, injections_log)

    # Initialize browser client
    browser = ChatGPTBrowserClient(cookies_file=cookies_file, headless=headless)

    try:
        await browser.start()
        await browser.start_new_thread()

        # Select ChatGPT model if specified
        if chatgpt_model:
            await browser.select_model(chatgpt_model)

        # --- RESUME-SPECIFIC PATCH ---
        # If we crashed after an assistant reply (last=assistant), we must do the user-agent move next.
        if transcript and transcript[-1]["role"] == "assistant":
            user_turn_index = turns_already_done
            # Generate or use canned user reply
            if user_turn_index < len(canned_prompts) and isinstance(canned_prompts[user_turn_index], str) and canned_prompts[user_turn_index].strip():
                user_reply = canned_prompts[user_turn_index]
                transcript.append({"role": "user", "content": user_reply})
                injections_log.append("")
            else:
                # Generate user-agent reply (with optional injection)
                current_user_system_prompt = user_system_prompt
                injection_this_turn = ""
                if injections and injection_frequency > 0 and rnd.random() < 1 / injection_frequency:
                    injection_to_add = rnd.choice(injections)
                    current_user_system_prompt = (
                        f"{user_system_prompt}\n\n[Additional instruction for this turn: {injection_to_add}]"
                    )
                    injection_this_turn = injection_to_add

                # Build user-agent view
                user_msgs: List[Dict[str, str]] = [
                    {"role": "system", "content": current_user_system_prompt},
                    *transcript,
                ]
                if user_msgs[-1]["role"] == "assistant":
                    user_msgs[-1] = {
                        **user_msgs[-1],
                        "content": (
                            user_msgs[-1]["content"]
                            + "\n\n[Instructions for your response:\n"
                            + f"{current_user_system_prompt}\n]"
                        ),
                    }
                flipped = [
                    {"role": "user" if m["role"] == "assistant" else "assistant", "content": m["content"]}
                    for m in user_msgs[1:]
                ]
                user_msgs_final = [user_msgs[0], *flipped]

                try:
                    user_reply = get_completion(
                        model=user_model,
                        messages=user_msgs_final,
                        api_key=user_agent_api_key,
                        base_url=user_agent_base_url,
                        site_url=site_url,
                        max_retries=max_retries,
                        backoff_factor=backoff_factor,
                    )
                    transcript.append({"role": "user", "content": user_reply})
                    injections_log.append(injection_this_turn)
                except APIError as err:
                    errors.append({"turn": turns_already_done, "agent": "user", "error": str(err)})
                    if save_turn_callback:
                        save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))
                    return ConversationResult(transcript, errors, injections_log)

            if save_turn_callback:
                save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))

        # --- main loop: assistant (via browser) then user-agent (via API) ---
        for turn_idx in range(turns_already_done, num_turns):
            # Assistant move (via browser)
            try:
                last_user_msg = transcript[-1]["content"]
                assistant_reply = await browser.send_message_and_get_response(last_user_msg, turn_idx)
                transcript.append({"role": "assistant", "content": assistant_reply})
                injections_log.append("")
            except Exception as err:
                errors.append({"turn": turn_idx, "agent": "assistant", "error": str(err)})
                if save_turn_callback:
                    save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))
                break

            if save_turn_callback:
                save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))

            # Add delay between turns to avoid rate limiting
            if turn_idx < num_turns - 1:  # Don't delay after last turn
                print(f"\nWaiting {turn_delay_seconds}s before next turn...")
                await asyncio.sleep(turn_delay_seconds)

            # User-agent move (scripted or generated)
            user_turn_index = turn_idx + 1
            if user_turn_index < len(canned_prompts) and isinstance(canned_prompts[user_turn_index], str) and canned_prompts[user_turn_index].strip():
                user_reply = canned_prompts[user_turn_index]
                transcript.append({"role": "user", "content": user_reply})
                injections_log.append("")
                if save_turn_callback:
                    save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))
                continue

            current_user_system_prompt = user_system_prompt
            injection_this_turn = ""
            if injections and injection_frequency > 0 and rnd.random() < 1 / injection_frequency:
                injection_to_add = rnd.choice(injections)
                current_user_system_prompt = (
                    f"{user_system_prompt}\n\n[Additional instruction for this turn: {injection_to_add}]"
                )
                injection_this_turn = injection_to_add

            try:
                user_msgs: List[Dict[str, str]] = [
                    {"role": "system", "content": current_user_system_prompt},
                    *transcript,
                ]
                if user_msgs[-1]["role"] == "assistant":
                    user_msgs[-1] = {
                        **user_msgs[-1],
                        "content": (
                            user_msgs[-1]["content"]
                            + "\n\n[Instructions for your response:\n"
                            + f"{current_user_system_prompt}\n]"
                        ),
                    }
                flipped = [
                    {"role": "user" if m["role"] == "assistant" else "assistant", "content": m["content"]}
                    for m in user_msgs[1:]
                ]
                user_msgs_final = [user_msgs[0], *flipped]

                user_reply = get_completion(
                    model=user_model,
                    messages=user_msgs_final,
                    api_key=user_agent_api_key,
                    base_url=user_agent_base_url,
                    site_url=site_url,
                    max_retries=max_retries,
                    backoff_factor=backoff_factor,
                )
                transcript.append({"role": "user", "content": user_reply})
                injections_log.append(injection_this_turn)
            except APIError as err:
                errors.append({"turn": turn_idx, "agent": "user", "error": str(err)})
                if save_turn_callback:
                    save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))
                break

            if save_turn_callback:
                save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))

    finally:
        await browser.close()

    return ConversationResult(transcript, errors, injections_log)
