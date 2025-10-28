# Utility Scripts

This folder contains utility scripts for data processing and database operations.

## Scripts

### `load_api_llm_scores.py`

Loads LLM-generated scores from CSV into the PostgreSQL database.

**Features:**
- Reads from `analysis/api_llm_scores.csv`
- Loads API transcripts from `results/` folder
- Finds exact character positions for each score snippet in the transcript
- Standardizes label names to match database conventions
- Strips "openai/" prefix from model names
- Uses canonical model names from JSON files

**Usage:**
```bash
# From repository root or utils folder:
python utils/load_api_llm_scores.py --limit 100  # Test with 100 rows
python utils/load_api_llm_scores.py              # Load all rows

# With custom CSV path:
python utils/load_api_llm_scores.py --csv path/to/scores.csv
```

**Requirements:**
- PostgreSQL database with `api_llm_scores` table
- `DATABASE_URL` environment variable set
- `psycopg2` installed

**Note:** The script automatically detects the repository root, so it works correctly whether run from the root directory or from within the `utils/` folder.

---

### `extract_chunk_metrics.py`

Extracts per-chunk metrics from result JSON files and creates a CSV matching the LLM scores structure.

**Features:**
- Processes JSON files from `results/` folder
- Extracts chunk-level judgments with snippets
- Creates a dataframe with model, category, turn_index, label, strength, snippet
- Saves output to `analysis/api_llm_scores_old.csv`

**Usage:**
```bash
# From repository root or utils folder:
python utils/extract_chunk_metrics.py
```

**Output:**
- CSV file: `analysis/api_llm_scores_old.csv`
- Console: Summary statistics and label distribution

**Note:** Currently processes only:
- `gpt-5-chat-latest.json`
- `chatgpt-4o-latest.json`

Edit the `target_files` list in the script to process different files.

---

## Path Resolution

Both scripts use dynamic path resolution based on `__file__` to locate the repository root. This means:

✓ Can be run from any directory
✓ Will always find `results/` and `analysis/` folders relative to repo root
✓ Output files are saved to correct locations

## Database Connection

`load_api_llm_scores.py` requires the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql://user:password@host:port/database"
```

Or create a `.env` file in the repository root.
