#!/usr/bin/env python3
"""
Load API LLM scores from CSV into the database.
This script reads analysis/api_llm_scores.csv and inserts the data into the api_llm_scores table.
"""

import csv
import os
import sys
import time
import json
import hashlib
import psycopg2
from pathlib import Path

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable not set")
    sys.exit(1)

# Standardize behavior labels to match database conventions
STANDARDIZE_LABELS = {
    'de-escalation': 'de_escalation',
    'suggestions to seek help': 'help_referral',
    'sycophancy or praise': 'sycophancy',
    'emotional or narrative escalation': 'escalation',
    'harmful advice': 'harmful_advice',
    'safe redirection': 'redirection',
    'delusion reinforcement': 'delusion_reinforcement',
    'consciousness claims': 'consciousness_claims'
}

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def standardize_label(label: str) -> str:
    """Standardize label names to match database conventions"""
    return STANDARDIZE_LABELS.get(label, label)

def clean_model_name(model: str) -> str:
    """Strip 'openai/' prefix from model name if present"""
    if model and isinstance(model, str):
        return model.replace("openai/", "")
    return model

def normalize_text(text: str) -> str:
    """Canonical text normalization (same as server.py)"""
    if not text:
        return ""
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = ''.join(c if c.isprintable() or c in '\n\t' else ' ' for c in text)
    return text.strip()

def load_transcript_cache():
    """Load all API transcripts into memory for fast lookup"""
    cache = {}
    # Get the repository root (parent of utils folder)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    results_dir = repo_root / "results"

    if not results_dir.exists():
        print("WARNING: results/ directory not found. Character positions will be set to 0.")
        return cache

    print("Loading API transcripts into cache...")
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = json_file.stem

            for run_idx, run_data in data.items():
                # Skip __meta__ entries
                if run_idx == "__meta__" or not isinstance(run_data, dict):
                    continue

                for prompt_file, prompt_data in run_data.items():
                    # Skip __meta__ entries
                    if prompt_file == "__meta__" or not isinstance(prompt_data, dict):
                        continue

                    for pid, convos in prompt_data.items():
                        # convos should be a list of conversation objects
                        if not isinstance(convos, list):
                            continue

                        for convo_idx, convo in enumerate(convos):
                            if not isinstance(convo, dict):
                                continue

                            transcript = convo.get("transcript", [])
                            category = convo.get("category", "")

                            # Create cache key
                            key = (model_name, category, pid, int(run_idx), convo_idx)
                            cache[key] = transcript
        except Exception as e:
            print(f"WARNING: Failed to load {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Cached {len(cache)} transcripts")
    return cache

def find_snippet_position(transcript, turn_index: int, snippet: str):
    """
    Find the character position of a snippet within a specific turn of the transcript.
    Returns (start_char, end_char) or (0, len(snippet)) if not found.
    """
    # Get assistant turns only
    assistant_turns = [t for t in transcript if t.get("role") == "assistant"]

    if turn_index >= len(assistant_turns):
        return 0, len(snippet)

    # Get the assistant turn content
    turn_content = assistant_turns[turn_index].get("content", "")
    normalized_content = normalize_text(turn_content)
    normalized_snippet = normalize_text(snippet)

    # Try to find the snippet in the normalized text
    pos = normalized_content.find(normalized_snippet)

    if pos != -1:
        return pos, pos + len(normalized_snippet)

    # If exact match not found, try partial matching (snippet might be truncated)
    # Search for first 50 chars of snippet
    if len(normalized_snippet) > 50:
        partial_snippet = normalized_snippet[:50]
        pos = normalized_content.find(partial_snippet)
        if pos != -1:
            return pos, pos + len(normalized_snippet)

    # If still not found, return placeholder
    return 0, len(snippet)

def find_matching_transcript(transcript_cache, csv_model, category, prompt_id, run_index, convo_index):
    """
    Find the matching transcript and return both the transcript and the canonical model name.
    Returns (transcript, canonical_model_name) or (None, None) if not found.
    """
    # Clean the CSV model name
    cleaned_model = clean_model_name(csv_model)

    # Try exact matches first
    for cache_key, transcript in transcript_cache.items():
        cache_model, cache_cat, cache_pid, cache_run, cache_convo = cache_key
        if (cache_cat == category and cache_pid == prompt_id and
            cache_run == run_index and cache_convo == convo_index):
            # Found a match on all other fields, check model
            if cleaned_model in cache_model or cache_model in cleaned_model:
                return transcript, cache_model

    return None, None

def load_csv_to_db(csv_path: str, limit: int = None):
    """Load CSV data into api_llm_scores table

    Args:
        csv_path: Path to the CSV file
        limit: Maximum number of rows to process (None for all rows)
    """

    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Load transcript cache first
    transcript_cache = load_transcript_cache()

    conn = get_conn()
    cur = conn.cursor()

    try:
        # Clear existing data
        print("\nClearing existing api_llm_scores data...")
        cur.execute("DELETE FROM api_llm_scores")
        deleted_count = cur.rowcount
        print(f"Deleted {deleted_count} existing rows")

        # Read CSV and insert data
        limit_msg = f" (limited to {limit} rows)" if limit else ""
        print(f"\nReading CSV: {csv_path}{limit_msg}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            inserted = 0
            skipped = 0
            label_mappings = {}
            position_found = 0
            position_not_found = 0

            for row in reader:
                # Stop if we've reached the limit
                if limit and inserted >= limit:
                    print(f"\nReached limit of {limit} rows")
                    break

                try:
                    # Extract fields from CSV
                    csv_model = row['model']
                    category = row['category']
                    prompt_id = row['scenario_id']  # CSV uses 'scenario_id' instead of 'prompt_id'
                    run_index = int(row['run_index'])
                    convo_index = int(row['convo_index'])
                    turn_index = int(row['turn_index'])

                    # Standardize label
                    raw_label = row['label']
                    label = standardize_label(raw_label)

                    # Track label mappings for reporting
                    if raw_label != label:
                        if raw_label not in label_mappings:
                            label_mappings[raw_label] = label

                    strength = int(row['strength'])
                    snippet = row['snippet']

                    # Find the matching transcript and get the canonical model name
                    transcript, canonical_model = find_matching_transcript(
                        transcript_cache, csv_model, category, prompt_id, run_index, convo_index
                    )

                    if transcript and canonical_model:
                        # Use the canonical model name from the JSON file
                        model = canonical_model
                        start_char, end_char = find_snippet_position(transcript, turn_index, snippet)
                        if start_char > 0 or end_char != len(snippet):
                            position_found += 1
                        else:
                            position_not_found += 1
                    else:
                        # Transcript not found, use cleaned CSV model name and placeholder positions
                        model = clean_model_name(csv_model)
                        start_char = 0
                        end_char = len(snippet)
                        position_not_found += 1

                    created_at = time.time()

                    # Insert into database
                    cur.execute("""
                        INSERT INTO api_llm_scores
                        (model, category, prompt_id, run_index, convo_index, turn_index,
                         label, strength, start_char, end_char, snippet, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (model, category, prompt_id, run_index, convo_index, turn_index,
                          label, strength, start_char, end_char, snippet, created_at))

                    inserted += 1

                    if inserted % 100 == 0:
                        print(f"Inserted {inserted} rows...", end='\r')

                except Exception as e:
                    print(f"\nWARNING: Skipped row due to error: {e}")
                    print(f"Row data: {row}")
                    skipped += 1
                    continue

        # Commit the transaction
        conn.commit()

        print(f"\n\n✓ Successfully loaded {inserted} rows into api_llm_scores table")
        if skipped > 0:
            print(f"⚠ Skipped {skipped} rows due to errors")

        print(f"\nCharacter position stats:")
        print(f"  Positions found: {position_found}")
        print(f"  Positions not found (using placeholders): {position_not_found}")

        # Report label mappings
        if label_mappings:
            print("\nLabel standardizations applied:")
            for old_label, new_label in sorted(label_mappings.items()):
                print(f"  '{old_label}' → '{new_label}'")

    except Exception as e:
        conn.rollback()
        print(f"\n✗ ERROR: Failed to load data: {e}")
        sys.exit(1)

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    import argparse

    # Get repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    default_csv = repo_root / "analysis" / "api_llm_scores.csv"

    parser = argparse.ArgumentParser(description='Load API LLM scores from CSV into database')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of rows to process (for testing)')
    parser.add_argument('--csv', type=str, default=str(default_csv),
                        help='Path to CSV file')
    args = parser.parse_args()

    csv_path = args.csv
    limit = args.limit

    print("=" * 60)
    print("API LLM Scores CSV to Database Loader")
    print("=" * 60)

    load_csv_to_db(csv_path, limit=limit)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
