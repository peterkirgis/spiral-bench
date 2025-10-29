#!/usr/bin/env python3
"""
Extract per-chunk metrics from res_v0.2 JSON files and create a dataframe
matching the llm_scores structure.
"""

import json
import pandas as pd
from pathlib import Path


def extract_chunk_metrics(json_path):
    """Extract chunk-level metrics from a single JSON file."""

    with open(json_path, 'r') as f:
        data = json.load(f)

    records = []

    # Iterate through the nested structure
    for run_index, run_data in data.items():
        if not isinstance(run_data, dict):
            continue

        for prompt_file, prompt_data in run_data.items():
            if not isinstance(prompt_data, dict):
                continue

            for prompt_id, conversations in prompt_data.items():
                # Ensure conversations is a list
                if not isinstance(conversations, list):
                    continue

                for convo in conversations:
                    # Skip if convo is not a dictionary
                    if not isinstance(convo, dict):
                        print(f"  Warning: skipping non-dict convo: {type(convo)}")
                        continue

                    # Extract metadata
                    category = convo.get('category', '')
                    evaluated_model = convo.get('evaluated_model', '')
                    convo_index = convo.get('convo_index', 0)

                    # Get judgements (per-chunk evaluations)
                    judgements = convo.get('judgements', {})

                    # Extract assistant turn indexes from transcript
                    transcript = convo.get('transcript', [])
                    assistant_turns = {}
                    for i, turn in enumerate(transcript):
                        if turn.get('role') == 'assistant':
                            assistant_turns[len(assistant_turns)] = i

                    # Process each chunk
                    for chunk_name, chunk_data in judgements.items():
                        metrics = chunk_data.get('metrics', {})
                        full_metrics = chunk_data.get('full_metrics', {})
                        assistant_turn_indexes = chunk_data.get('assistant_turn_indexes', [])

                        # Determine the turn index (use the first assistant turn in the chunk)
                        turn_index = assistant_turn_indexes[0] if assistant_turn_indexes else None

                        # For each label that has instances in this chunk
                        for label, instances in full_metrics.items():
                            if instances:  # If there are any instances of this label
                                for text_snippet, strength in instances:
                                    # Handle run_index as either int or string
                                    try:
                                        run_idx_value = int(run_index)
                                    except (ValueError, TypeError):
                                        # If it's a string like 'run_92bf600c', keep it as-is
                                        run_idx_value = run_index

                                    record = {
                                        'model': evaluated_model,
                                        'category': category,
                                        'scenario_id': prompt_id,
                                        'run_index': run_idx_value,
                                        'convo_index': convo_index,
                                        'chunk_name': chunk_name,
                                        'turn_index': turn_index,
                                        'label': label,
                                        'strength': strength,
                                        'snippet': text_snippet,
                                        'assistant_length_chars': chunk_data.get('assistant_length_chars', 0)
                                    }
                                    records.append(record)

    return records


def main():
    # Get the repository root (parent of utils folder)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    res_dir = repo_root / 'results'

    # Process only specific JSON files
    target_files = [
        'gpt-5-chat-latest-full-added.json',
        'chatgpt-4o-latest-full-added.json'
    ]
    json_files = [res_dir / f for f in target_files if (res_dir / f).exists()]

    print(f"Processing {len(json_files)} specified JSON files from {res_dir}")

    all_records = []

    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        try:
            records = extract_chunk_metrics(json_file)
            all_records.extend(records)
            print(f"  Extracted {len(records)} records")
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")

    # Create dataframe
    df = pd.DataFrame(all_records)

    print(f"\nTotal records extracted: {len(df)}")
    print(f"\nDataframe shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nUnique models: {df['model'].nunique()}")
    print(f"Unique categories: {df['category'].nunique()}")
    print(f"Unique labels: {df['label'].nunique()}")

    # Save to CSV in the analysis folder
    output_path = repo_root / 'analysis' / 'api_llm_scores_added.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Display sample
    print(f"\nSample rows:")
    print(df.head(10))

    # Show label distribution
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())

    return df


if __name__ == '__main__':
    df = main()
