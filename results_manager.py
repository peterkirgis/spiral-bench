import json
import os
import threading
import logging


class ResultsManager:
    """
    Manages loading and saving conversation results to a JSON file
    in a thread-safe and atomic manner.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.data = self._load()

    # ──────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────

    def is_completed(self, run_id: str, file_key: str, prompt_key: str, convo_index: int) -> bool:
        """
        A conversation is “complete” only if a record exists **and**
        the record’s `completed` flag is True.
        """
        with self.lock:
            try:
                convo = self.data[run_id][file_key][prompt_key][convo_index]
                return convo is not None and convo.get("completed") is True
            except (KeyError, IndexError):
                return False

    def is_judged(self, run_id: str, file_key: str, prompt_key: str, convo_index: int) -> bool:
        """
        Checks if a conversation has a non-null 'judgement' field.
        """
        with self.lock:
            try:
                convo = self.data[run_id][file_key][prompt_key][convo_index]
                return convo is not None and convo.get("judgement") is not None
            except (KeyError, IndexError):
                return False

    def save_result(
        self,
        run_id: str,
        file_key: str,
        prompt_key: str,
        convo_index: int,
        conversation_data: dict,
    ):
        """
        Persist a (possibly partial) transcript. Re-writes the same list slot
        on every call; safe under the class-wide lock. Also used to save judgements.
        """
        with self.lock:
            run_results = self.data.setdefault(run_id, {})
            file_results = run_results.setdefault(file_key, {})
            prompt_results = file_results.setdefault(prompt_key, [])

            while len(prompt_results) <= convo_index:
                prompt_results.append(None)

            prompt_results[convo_index] = conversation_data
            self._atomic_write()
        
        status = "judgement" if "judgement" in conversation_data else "result"
        logging.info(f"Saved {status} for {file_key}/{prompt_key} [convo {convo_index}]")


    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _load(self) -> dict:
        with self.lock:
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    logging.warning(
                        f"Could not decode JSON from {self.filepath}. Starting fresh."
                    )
                    return {}
            return {}

    def _atomic_write(self):
        tmp = self.filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.filepath)