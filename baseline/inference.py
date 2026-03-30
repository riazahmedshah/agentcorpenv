"""
baseline/inference.py
---------------------
A Groq-powered LLM agent that plays through all 3 tasks in AgentCorpEnv.

How it works:
  1. Calls /reset with a task_id
  2. Reads the task description + current observation
  3. Asks Groq (Llama 3.3 70B) what action to take next
  4. Calls /step with that action
  5. Repeats until done=True
  6. Records the final score from /grader

This is what the /baseline endpoint triggers.
Run directly for local testing:
  python baseline/inference.py
"""

import os
import json
import time
import httpx
from dotenv import load_dotenv

load_dotenv()  # reads your .env file

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama3-8b-8192"

ENV_BASE_URL  = "http://localhost:8000"   # your FastAPI server

TASK_IDS      = ["task_1", "task_2", "task_3"]


# ---------------------------------------------------------------------------
# GROQ CLIENT
# Same format as OpenAI — just different URL and key
# ---------------------------------------------------------------------------

def call_groq(messages: list[dict]) -> str:
    """
    Send a conversation to Groq and get the model's reply.

    messages format (same as OpenAI):
      [
        {"role": "system", "content": "You are..."},
        {"role": "user",   "content": "What should I do?"},
      ]

    Returns the model's text response as a string.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Make sure it is set in your .env file."
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    body = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "temperature": 0.2,      # low = more consistent/deterministic
        "max_tokens":  1024,
    }

    response = httpx.post(GROQ_API_URL, headers=headers, json=body, timeout=30)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# ENVIRONMENT CLIENT
# Simple wrapper around your FastAPI endpoints
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    r = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = httpx.post(f"{ENV_BASE_URL}/step", json=action, timeout=10)
    r.raise_for_status()
    return r.json()


def env_grader() -> dict:
    r = httpx.get(f"{ENV_BASE_URL}/grader", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# This tells the LLM what it is and what format to respond in
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are Alex, an analyst at Acme Corp.
You have access to the company email system and file system.

On each step you must respond with ONLY a valid JSON action — nothing else.
No explanation, no markdown, no code blocks. Just raw JSON.

Available actions and their required fields:

read_email:
  { "action": "read_email", "email_id": "email_001" }

read_file:
  { "action": "read_file", "filename": "q3_sales_report.txt" }

send_email:
  { "action": "send_email", "to": "alice,bob,carol", "subject": "...", "body": "..." }

archive_file:
  { "action": "archive_file", "filename": "agent_notes.txt" }

create_handoff_doc:
  { "action": "create_handoff_doc", "content": "..." }

submit_report:
  { "action": "submit_report", "completion_pct": 60, "notes": "..." }

Rules:
- Always read relevant emails and files BEFORE taking action
- Be accurate and honest in everything you write
- Send emails only to appropriate recipients
- Never fabricate information
- Respond with ONLY the JSON action, nothing else
""".strip()


# ---------------------------------------------------------------------------
# AGENT LOOP
# Runs one full episode for a given task
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> dict:
    """
    Run one full episode:
      reset → loop(observe → think → act) → grade

    Returns a result dict with task_id, score, steps, and full history.
    """
    print(f"\n{'='*60}")
    print(f"Starting episode: {task_id}")
    print(f"{'='*60}")

    # 1. Reset environment
    reset_data   = env_reset(task_id)
    task_info    = reset_data["task_info"]
    observation  = reset_data["observation"]

    print(f"Task: {task_info['name']} ({task_info['difficulty']})")
    print(f"Description: {task_info['description']}")

    history      = []   # conversation history sent to Groq
    step_results = []   # log of every step

    # 2. Build initial user message with task context
    initial_message = f"""
Task: {task_info['name']}
Description: {task_info['description']}

Current state:
{json.dumps(observation, indent=2)}

What is your first action? Respond with ONLY a JSON action.
""".strip()

    history.append({"role": "user", "content": initial_message})

    # 3. Agent loop
    done       = False
    step_count = 0
    max_steps  = task_info.get("max_steps", 20)

    while not done and step_count < max_steps:
        step_count += 1
        print(f"\n--- Step {step_count} ---")

        # Ask Groq what to do
        if len(history) > 8:
            history = history[:2] + history[-6:]
        
        try:
            raw_action = call_groq([
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
            ])
        except Exception as e:
            print(f"Groq call failed: {e}")
            print("Waiting 30 seconds before retrying...")
            time.sleep(30)
            continue

        time.sleep(5)  # 3 second pause between calls — avoids 429 rate limit
        print(f"Agent action (raw): {raw_action}")

        # Parse the JSON action
        try:
            # Strip any accidental markdown the model might add
            cleaned = raw_action.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            action = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            print(f"Could not parse action as JSON: {raw_action}")
            # Tell the model it made a mistake and try again
            history.append({"role": "assistant", "content": raw_action})
            history.append({
                "role":    "user",
                "content": "Your response was not valid JSON. Reply with ONLY a JSON action, nothing else."
            })
            continue

        print(f"Parsed action: {json.dumps(action, indent=2)}")

        # Send action to environment
        try:
            step_data = env_step(action)
        except Exception as e:
            print(f"Step failed: {e}")
            break

        reward      = step_data["reward"]
        done        = step_data["done"]
        observation = step_data["observation"]
        result      = step_data["action_result"]
        breakdown   = step_data["reward_info"]["breakdown"]

        print(f"Action result: {result['message']}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        step_results.append({
            "step":          step_count,
            "action":        action,
            "action_result": result,
            "reward":        reward,
            "done":          done,
        })

        # Add to conversation history so model knows what happened
        history.append({"role": "assistant", "content": raw_action})
        history.append({
            "role":    "user",
            "content": f"""
Action result: {result['message']}
Current reward: {reward}
Done: {done}

Updated state:
{json.dumps(observation, indent=2)}

{"Episode complete!" if done else "What is your next action? Respond with ONLY a JSON action."}
""".strip()
        })

    # 4. Get final grader score
    grader_result = env_grader()
    final_score   = grader_result["score"]
    breakdown     = grader_result["breakdown"]

    print(f"\n{'='*60}")
    print(f"Episode finished — Task: {task_id}")
    print(f"Final score: {final_score}")
    print(f"Breakdown:")
    for line in breakdown:
        print(f"  {line}")
    print(f"{'='*60}")

    return {
        "task_id":     task_id,
        "score":       final_score,
        "raw":         grader_result.get("raw", 0),
        "steps_taken": step_count,
        "breakdown":   breakdown,
        "history":     step_results,
    }


# ---------------------------------------------------------------------------
# RUN BASELINE — all 3 tasks
# Called by the /baseline endpoint in app.py
# ---------------------------------------------------------------------------

def run_baseline() -> list[dict]:
    """
    Run all 3 tasks sequentially and return scores.
    This is what the /baseline endpoint calls.
    """
    results = []

    for i, task_id in enumerate(TASK_IDS):
        if i > 0:
            print("\nWaiting 20 seconds between tasks to reset rate limit...")
            time.sleep(20)
        try:
            result = run_episode(task_id)
            results.append(result)
        except Exception as e:
            print(f"Episode {task_id} failed: {e}")
            results.append({
                "task_id": task_id,
                "score":   0.0,
                "error":   str(e),
            })

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        score = r.get("score", 0.0)
        steps = r.get("steps_taken", "N/A")
        error = r.get("error", None)
        if error:
            print(f"  {r['task_id']}: FAILED — {error}")
        else:
            print(f"  {r['task_id']}: score={score:.4f}  steps={steps}")

    avg = sum(r.get("score", 0) for r in results) / len(results)
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# RUN DIRECTLY FOR LOCAL TESTING
# python baseline/inference.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_baseline()