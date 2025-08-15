import random
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict

from api_client import get_completion, APIError

@dataclass
class ConversationResult:
    transcript: List[Dict[str, str]]
    errors: List[Dict[str, str]]
    injections_log: List[str]

SaveCB = Optional[Callable[[ConversationResult], None]]

def run_conversation(
    user_model: str,
    evaluated_model: str,
    user_system_prompt: str,
    evaluated_system_prompt: str,
    canned_prompts: List[Optional[str]],
    num_turns: int,
    user_agent_api_key: str,
    user_agent_base_url: str,
    evaluated_model_api_key: str,
    evaluated_model_base_url: str,
    site_url: str,
    max_retries: int,
    backoff_factor: float,
    save_turn_callback: SaveCB = None,
    injections: Optional[List[str]] = None,
    injection_frequency: int = 5,
    seed: Optional[str] = None,
    resume_state: Optional["ConversationResult"] = None,
) -> ConversationResult:
    """
    Dialogue between two agents. The transcript stores canonical roles,
    but each API call must end with a 'user' role addressed to the callee.
    """
    def _assert_last_is_user(msgs: List[Dict[str, str]]) -> None:
        assert msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"].strip(), (
            "Message list handed to the API must end with a non-empty user message"
        )

    # --- bootstrap or resume ---
    if resume_state and resume_state.transcript:
        transcript     = resume_state.transcript.copy()
        errors         = resume_state.errors.copy()
        injections_log = resume_state.injections_log.copy()
    else:
        initial_user_message = canned_prompts[0]
        if not (isinstance(initial_user_message, str) and initial_user_message.strip()):
            raise AssertionError("Initial canned prompt (index 0) must be a non-empty string.")
        transcript     = [{"role": "user", "content": initial_user_message}]
        errors         = []
        injections_log = [""]  # keep 1:1 with transcript

    # keep injections_log aligned with transcript length (no silent gaps)
    if len(injections_log) < len(transcript):
        injections_log.extend([""] * (len(transcript) - len(injections_log)))
    elif len(injections_log) > len(transcript):
        injections_log = injections_log[:len(transcript)]

    rnd = random.Random(seed)

    # initial save
    if save_turn_callback:
        save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))

    # how many assistant turns already happened
    turns_already_done = sum(1 for m in transcript if m["role"] == "assistant")
    if turns_already_done >= num_turns:
        # conversation already complete from the assistant's perspective
        return ConversationResult(transcript, errors, injections_log)

    # --- RESUME-SPECIFIC PATCH ---
    # If we crashed after an assistant reply (last=assistant), we must do the user-agent move next.
    if transcript and transcript[-1]["role"] == "assistant":
        user_turn_index = turns_already_done  # next user prompt index corresponds to this assistant count
        # 1) scripted reply if present
        if user_turn_index < len(canned_prompts) and isinstance(canned_prompts[user_turn_index], str) and canned_prompts[user_turn_index].strip():
            user_reply = canned_prompts[user_turn_index]
            transcript.append({"role": "user", "content": user_reply})
            injections_log.append("")
        else:
            # 2) generate user-agent reply (with optional injection)
            current_user_system_prompt = user_system_prompt
            injection_this_turn = ""
            if injections and injection_frequency > 0 and rnd.random() < 1 / injection_frequency:
                injection_to_add = rnd.choice(injections)
                current_user_system_prompt = (
                    f"{user_system_prompt}\n\n[Additional instruction for this turn: {injection_to_add}]"
                )
                injection_this_turn = injection_to_add

            # Build user-agent view: system + canonical transcript; ensure last is user for the callee.
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
            _assert_last_is_user(user_msgs_final)

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
        # from here, last is user → normal loop can proceed.

    # --- main loop: assistant then user-agent ---
    for turn_idx in range(turns_already_done, num_turns):
        # assistant move
        try:
            _assert_last_is_user(transcript)
            assistant_reply = get_completion(
                model=evaluated_model,
                messages=transcript,
                api_key=evaluated_model_api_key,
                base_url=evaluated_model_base_url,
                site_url=site_url,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
            )
            transcript.append({"role": "assistant", "content": assistant_reply})
            injections_log.append("")
        except APIError as err:
            errors.append({"turn": turn_idx, "agent": "assistant", "error": str(err)})
            if save_turn_callback:
                save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))
            break
        if save_turn_callback:
            save_turn_callback(ConversationResult(transcript.copy(), errors.copy(), injections_log.copy()))

        # user-agent move (scripted or generated)
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
            _assert_last_is_user(user_msgs_final)

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

    return ConversationResult(transcript, errors, injections_log)
