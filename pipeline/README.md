# Pipeline Module

This folder contains the original pipeline scripts for running automated red teaming evaluations.

## Files

- **api_client.py** - Handles API calls to LLM providers (OpenAI, Anthropic, etc.)
- **conversation_runner.py** - Handles running conversations between user and evaluated models
- **results_manager.py** - Manages storage and retrieval of conversation results
- **metrics.py** - Defines evaluation metrics and scoring calculations
- **scoring.py** - Implements scoring logic for evaluating model responses

## Usage

These modules are imported by `main.py` in the root directory. To run the pipeline:

```bash
python main.py --help
```

See the main repository README for full usage instructions.
