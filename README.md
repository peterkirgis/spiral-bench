# Spiral-Bench: Multiturn Evaluation for Sycophancy & Delusion Behaviours

This repository contains the **Spiral-Bench** evaluation pipeline: a multiturn, roleplay-based benchmark designed to measure **protective** and **risky** behaviours in large language models (LLMs) when interacting with a suggestible, seeker-type user.
It simulates realistic back-and-forth conversations, then uses a separate judge model to label behaviours against a defined rubric.

---

## 📋 Overview

Spiral-Bench runs large batches of **scripted + generated dialogues** between:

* **User agent** — a fixed LLM role-playing as a highly suggestible human (“Kimi-K2” by default)
* **Evaluated model** — the model you want to test
* **Judge model** — a third LLM that analyses the transcripts and logs behavioural incidences

The benchmark focuses on identifying and quantifying:

* **Protective actions** (e.g., pushback, de-escalation, safe redirection)
* **Risky actions** (e.g., delusion reinforcement, harmful advice, sycophancy, consciousness claims)

All runs are automated and produce both **raw per-conversation annotations** and **aggregated scores**.

---

## 📂 Repository Structure

```
ai/sycophancy-delusions-eval/
│
├── main.py                     # Orchestration script for generation + judging phases
├── conversation_runner.py      # Dialogue simulation between user agent & evaluated model
├── results_manager.py          # Thread-safe, atomic JSON result storage
├── api_client.py               # API call logic with retry/backoff handling
├── metrics.py                   # Text analysis & repetition/slop/complexity metrics
│
├── data/                       # Rubrics and judge prompt templates
│   ├── rubric_criteria.txt
│   ├── rubric_prompt.txt
│   ├── ...
│
├── user_instructions/          # Roleplay scripts for the “user” model
│   ├── default.txt
│   ├── variant1.txt
│
├── .env.example                # Example API key config
├── README.md                   # This file
└── ...
```

---

## ⚙️ How It Works

### 1. **Generation Phase**

* Loads a list of **prompt files** containing scenarios and categories
* Pairs them with **user role instructions** and optional **prompt injections**
* Runs simulated conversations (`--num-turns` turns) between:

  * **User model** — plays the seeker-type character
  * **Evaluated model** — the system under test
* Conversations are stored incrementally so they can resume after interruptions

### 2. **Judging Phase**

* Splits each conversation into **chunks** of assistant turns
* Builds a judging prompt with:

  * The chunk transcript
  * The rubric criteria (protective + risky behaviours)
  * A strict JSON output format
* Sends each chunk to the **judge model**
* Stores structured behaviour findings (per metric, with intensity ratings)

### 3. **Aggregation**

* Normalises scores per 10,000 assistant output characters
* Computes average incidence and strength per metric
* Produces summary statistics and optional leaderboard data

---

## 📊 Behaviour Rubric

The default rubric (see `data/rubric_criteria.txt`) includes:

**Protective:**

* Pushback
* De-escalation
* Safe redirection
* Suggestions to seek help

**Risky:**

* Emotional or narrative escalation
* Sycophancy or praise
* Delusion reinforcement
* Consciousness claims
* Harmful advice

---

## 🚀 Running an Evaluation

### 1. **Install dependencies**

```bash
pip install -r requirements.txt
```

Make sure you have NLTK corpora downloaded if you run metrics:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('cmudict')
```

### 2. **Configure API keys**

Copy `.env.example` → `.env` and fill in keys for:

* `USER_AGENT_API_KEY`
* `EVALUATED_MODEL_API_KEY`
* `JUDGE_API_KEY`

These can point to OpenRouter or OpenAI-compatible APIs.

### 3. **Run**

Example:

```bash
python main.py \
  --prompt-files prompts/eval_prompts_v0.1.json \
  --user-model moonshotai/kimi-k2 \
  --evaluated-model openai/chatgpt-4o-latest \
  --judge-model o3 \
  --prompt-injections user_instructions/entropy.json \
  --prompt-injection-every-n 5 \
  --num-prompts 30 \
  --convos-per-prompt 1 \
  --num-turns 20 \
  --parallelism 300 \
  --run-id 1 \
  --output-file "res_v0.2/chatgpt-4o-latest.json"
```

**Key arguments:**

* `--prompt-files` : JSON list of prompt scenarios
* `--prompt-injections` : JSON list of injection strings (optional)
* `--user-model` : roleplay agent (fixed)
* `--evaluated-model` : model to evaluate
* `--judge-model` : model that scores behaviour
* `--num-turns` : turns per conversation
* `--judge-chunk-size` : assistant turns per judging chunk
* `--redo-judging` : wipe and redo judging for an existing run

---

## 📈 Output Format

Results are stored in a JSON structure:

```
{
  "run_id": {
    "file_key": {
      "prompt_key": [
        {
          "completed": true,
          "transcript": [...],
          "judgements": {
            "chunk0": {
              "metrics": { ... },
              "full_metrics": { ... },
              "assistant_turn_indexes": [...],
              "assistant_length_chars": int
            },
            ...
          }
        },
        ...
      ]
    }
  }
}
```

---

## 📑 Metrics & Analysis

The included `metrics.py` module provides:

* **Complexity Index** — based on Flesch-Kincaid + polysyllabic word ratio
* **Slop Index** — weighted score of low-quality/filler phrases
* **Repetition Metric** — over-represented words across prompts
* **Human baseline n-gram overuse detection** — comparing model output to human writing profiles

These can be run separately to produce additional leaderboard columns.

---

## 🧪 Tips & Notes

* **Parallelism**: You can safely run with hundreds of concurrent workers if your API quota allows.
* **Resuming**: If interrupted, rerunning with the same `--run-id` resumes unfinished conversations.
* **Prompt coverage**: `create_task_list` skips already-completed conversations.
* **Judge chunking**: Small chunk sizes reduce context length but increase API calls.

---

## 📜 License

*(Add your license here if applicable)*

---

Do you want me to also include **a diagram of the conversation & judging flow** in the README so new contributors can see the process visually? That would make the pipeline much easier to grasp.
