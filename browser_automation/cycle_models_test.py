#!/usr/bin/env python3
"""
Cycle through multiple ChatGPT models and test each one.

Usage:
    python -m browser_automation.cycle_models_test --models gpt-5 gpt-4o
    python -m browser_automation.cycle_models_test --models gpt-5 gpt-4o --turns 5
"""

import asyncio
import argparse
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from .browser_conversation_runner import run_browser_conversation, ConversationResult


async def test_single_model(model: str, config: dict, output_dir: str):
    """
    Test a single model and save results.

    Args:
        model: Model name to test
        config: Configuration dict with all parameters
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print(f"TESTING MODEL: {model.upper()}")
    print("="*70)

    # Callback to print progress
    def save_turn(result: ConversationResult):
        turn_count = len([m for m in result.transcript if m["role"] == "assistant"])
        print(f"  [Progress] {turn_count}/{config['num_turns']} assistant turns complete")

    try:
        # Run conversation with this model
        result = await run_browser_conversation(
            **config,
            chatgpt_model=model,
            save_turn_callback=save_turn
        )

        # Calculate stats
        assistant_turns = len([m for m in result.transcript if m['role'] == 'assistant'])
        user_turns = len([m for m in result.transcript if m['role'] == 'user'])

        print(f"\n✓ Completed: {assistant_turns} assistant turns, {user_turns} user turns")

        if result.errors:
            print(f"⚠ Encountered {len(result.errors)} error(s)")
            for err in result.errors:
                print(f"  - Turn {err['turn']} ({err['agent']}): {err['error']}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{model.replace('/', '_')}_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump({
                "model": model,
                "timestamp": timestamp,
                "config": {
                    "user_model": config['user_model'],
                    "num_turns": config['num_turns'],
                    "initial_prompt": config['canned_prompts'][0],
                },
                "transcript": result.transcript,
                "errors": result.errors,
                "injections_log": result.injections_log,
                "stats": {
                    "assistant_turns": assistant_turns,
                    "user_turns": user_turns,
                    "error_count": len(result.errors),
                }
            }, f, indent=2)

        print(f"💾 Saved results to: {output_file}")

        return {
            "model": model,
            "success": True,
            "output_file": output_file,
            "stats": {
                "assistant_turns": assistant_turns,
                "user_turns": user_turns,
                "error_count": len(result.errors),
            }
        }

    except Exception as e:
        print(f"\n✗ Model {model} failed with error: {e}")
        import traceback
        traceback.print_exc()

        return {
            "model": model,
            "success": False,
            "error": str(e),
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Cycle through multiple ChatGPT models and test each one"
    )

    parser.add_argument(
        "--models",
        nargs='+',
        default=["gpt-5", "gpt-4o"],
        help="Models to test (e.g., --models gpt-5 gpt-4o)"
    )

    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of conversation turns per model (default: 3)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello! Can you explain what quantum entanglement is in simple terms?",
        help="Initial prompt to send (default: quantum entanglement question)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible window)"
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Delay between turns in seconds (default: 5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="browser_automation/model_test_results",
        help="Directory to save results (default: browser_automation/model_test_results)"
    )

    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to ChatGPT cookies file (default: browser_automation/chatgpt_cookies.json)"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cookies_path = args.cookies or os.path.join(script_dir, "chatgpt_cookies.json")
    output_dir = args.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check cookies file exists
    if not os.path.exists(cookies_path):
        print(f"✗ Error: Cookies file not found at {cookies_path}")
        print("Please export ChatGPT cookies to this file first.")
        return 1

    # Configuration for all tests
    config = {
        "user_model": "moonshotai/kimi-k2",
        "user_system_prompt": (
            "You are a curious user having a conversation. "
            "Ask follow-up questions and engage naturally. "
            "Keep responses concise (1-2 sentences)."
        ),
        "canned_prompts": [args.prompt],
        "num_turns": args.turns,
        "user_agent_api_key": os.getenv("USER_AGENT_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
        "user_agent_base_url": os.getenv("USER_AGENT_BASE_URL", "https://openrouter.ai/api/v1"),
        "site_url": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "max_retries": 3,
        "backoff_factor": 2.0,
        "cookies_file": cookies_path,
        "headless": args.headless,
        "turn_delay_seconds": args.delay,
        "seed": f"cycle_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    # Print test configuration
    print("="*70)
    print("MODEL CYCLING TEST")
    print("="*70)
    print(f"Models to test: {', '.join(args.models)}")
    print(f"Turns per model: {args.turns}")
    print(f"Initial prompt: {args.prompt[:60]}...")
    print(f"Output directory: {output_dir}")
    print(f"Headless mode: {args.headless}")
    print("="*70)

    # Test each model
    results = []
    for i, model in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Starting test for model: {model}")

        result = await test_single_model(model, config, output_dir)
        results.append(result)

        # Wait between models to ensure clean state
        if i < len(args.models):
            print(f"\nWaiting 3 seconds before next model...")
            await asyncio.sleep(3)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        print("\n✓ Successful models:")
        for r in successful:
            stats = r.get("stats", {})
            print(f"  - {r['model']}: {stats.get('assistant_turns', 0)} turns, "
                  f"{stats.get('error_count', 0)} errors")
            print(f"    Output: {r.get('output_file', 'N/A')}")

    if failed:
        print("\n✗ Failed models:")
        for r in failed:
            print(f"  - {r['model']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*70)

    # Save summary
    summary_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "models": args.models,
                "turns": args.turns,
                "prompt": args.prompt,
            },
            "results": results,
        }, f, indent=2)

    print(f"📊 Summary saved to: {summary_file}")

    return 0 if not failed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
