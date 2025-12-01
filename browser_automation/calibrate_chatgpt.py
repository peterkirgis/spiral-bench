#!/usr/bin/env python3
"""
Calibration script for ChatGPT browser automation.

This script opens ChatGPT and asks you to click on specific UI elements.
It records the coordinates and saves them for automated use.

Usage:
    python -m browser_automation.calibrate_chatgpt
"""

import asyncio
import json
import os
from playwright.async_api import async_playwright


async def main():
    print("="*70)
    print("ChatGPT UI CALIBRATION")
    print("="*70)
    print("\nThis script will guide you through clicking on ChatGPT UI elements.")
    print("We'll record the positions for automated clicking.\n")

    # Load cookies
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cookies_file = os.path.join(script_dir, "chatgpt_cookies.json")
    calibration_file = os.path.join(script_dir, "chatgpt_calibration.json")

    if not os.path.exists(cookies_file):
        print(f"✗ Error: {cookies_file} not found")
        print("Please export your ChatGPT cookies first.")
        return 1

    print(f"Loading cookies from: {cookies_file}")

    with open(cookies_file, 'r') as f:
        cookies = json.load(f)

    # Normalize cookies
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
        if 'sameSite' in cookie:
            same_site = cookie['sameSite']
            if isinstance(same_site, str):
                same_site = same_site.capitalize()
                if same_site == 'No_restriction':
                    same_site = 'None'
                normalized['sameSite'] = same_site
        if 'expirationDate' in cookie and cookie['expirationDate'] != -1:
            normalized['expires'] = cookie['expirationDate']
        normalized_cookies.append(normalized)

    # Start browser
    print("\nStarting browser...")
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=False,  # Must be visible for calibration
        args=['--disable-blink-features=AutomationControlled']
    )

    context = await browser.new_context(
        viewport={'width': 1280, 'height': 720},
        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    )
    await context.add_cookies(normalized_cookies)

    page = await context.new_page()

    print("Loading ChatGPT...")
    await page.goto('https://chatgpt.com', wait_until='domcontentloaded', timeout=60000)
    await asyncio.sleep(3)

    print("✓ ChatGPT loaded\n")

    # Enable mouse tracking on the page
    await page.evaluate("""
        () => {
            window.mouseX = 0;
            window.mouseY = 0;
            document.addEventListener('mousemove', (e) => {
                window.mouseX = e.clientX;
                window.mouseY = e.clientY;
            });
        }
    """)

    # Calibration data to collect
    calibration_data = {
        "viewport_width": 1280,
        "viewport_height": 720,
        "clicks": {},
        "hovers": {}
    }

    async def record_position(element_name: str, instruction: str, position_type: str = "click"):
        """Record mouse position automatically after hovering for 3 seconds"""
        print(f"\n→ {instruction}")

        # Give user time to read and position mouse
        await asyncio.sleep(2)

        pos_data = await page.evaluate("""
            () => {
                return new Promise((resolve) => {
                    let lastX = window.mouseX;
                    let lastY = window.mouseY;
                    let stillTime = 0;
                    const checkInterval = 200; // Check every 200ms
                    const requiredStillTime = 3000; // Must be still for 3 seconds

                    const checker = setInterval(() => {
                        const currentX = window.mouseX;
                        const currentY = window.mouseY;

                        // Check if mouse moved (with small tolerance for jitter)
                        if (Math.abs(currentX - lastX) < 5 && Math.abs(currentY - lastY) < 5) {
                            stillTime += checkInterval;

                            if (stillTime >= requiredStillTime) {
                                clearInterval(checker);
                                resolve({
                                    x: currentX,
                                    y: currentY,
                                    pageX: currentX,
                                    pageY: currentY
                                });
                            }
                        } else {
                            // Mouse moved, reset timer
                            stillTime = 0;
                            lastX = currentX;
                            lastY = currentY;
                        }
                    }, checkInterval);
                });
            }
        """)

        print(f"  ✓ Captured at ({pos_data['x']}, {pos_data['y']})")

        if position_type == "click":
            calibration_data["clicks"][element_name] = pos_data
        else:
            calibration_data["hovers"][element_name] = pos_data

        await asyncio.sleep(1)

    print("\n========== CALIBRATION START ==========")
    print("Hover over each element and hold still for 3 seconds.")
    print("DON'T click or press keys!\n")
    input("Press ENTER to begin...")

    # Step 1: Dropdown button
    await record_position(
        "model_dropdown",
        "Hover over the model dropdown button (shows 'ChatGPT 5.1')",
        "click"
    )

    print("  Opening dropdown...")
    await page.mouse.click(
        calibration_data["clicks"]["model_dropdown"]["x"],
        calibration_data["clicks"]["model_dropdown"]["y"]
    )
    await asyncio.sleep(3)

    # Step 2: Legacy models row
    await record_position(
        "legacy_models_hover",
        "Hover over 'Legacy models' row (submenu will appear)",
        "hover"
    )

    await asyncio.sleep(2)

    # Step 3: GPT-5 Instant
    await record_position(
        "gpt5_instant",
        "Hover over 'GPT-5 Instant' in the side submenu",
        "click"
    )

    # Step 4: GPT-4o
    print("  Reopening for GPT-4o...")
    await page.mouse.click(400, 400)
    await asyncio.sleep(2)
    await page.mouse.click(
        calibration_data["clicks"]["model_dropdown"]["x"],
        calibration_data["clicks"]["model_dropdown"]["y"]
    )
    await asyncio.sleep(2)
    await page.mouse.move(
        calibration_data["hovers"]["legacy_models_hover"]["x"],
        calibration_data["hovers"]["legacy_models_hover"]["y"]
    )
    await asyncio.sleep(2)

    await record_position(
        "gpt4o",
        "Hover over 'GPT-4o' in the side submenu",
        "click"
    )

    # Optional: GPT-5 Thinking
    print("\n→ Record GPT-5 Thinking? (y/n)")
    response = input("  ").lower().strip()

    if response in ['y', 'yes']:
        print("  Reopening for GPT-5 Thinking...")
        await page.mouse.click(400, 400)
        await asyncio.sleep(2)
        await page.mouse.click(
            calibration_data["clicks"]["model_dropdown"]["x"],
            calibration_data["clicks"]["model_dropdown"]["y"]
        )
        await asyncio.sleep(2)
        await page.mouse.move(
            calibration_data["hovers"]["legacy_models_hover"]["x"],
            calibration_data["hovers"]["legacy_models_hover"]["y"]
        )
        await asyncio.sleep(2)

        await record_position(
            "gpt5_thinking",
            "Hover over 'GPT-5 Thinking' in the side submenu",
            "click"
        )

    # Save calibration data
    print("\n========== SAVING ==========")

    with open(calibration_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print(f"✓ Saved to: {calibration_file}")
    print(f"\nRecorded {len(calibration_data['clicks'])} clicks, {len(calibration_data['hovers'])} hovers")
    print("\n========== DONE! ==========")
    print("\nRun this to test:")
    print("  python -m browser_automation.cycle_models_test --models gpt-5 gpt-4o\n")

    # Cleanup
    await browser.close()
    await playwright.stop()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
