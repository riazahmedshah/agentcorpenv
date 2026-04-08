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
from mcp import cli
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(override=False)

# CONFIG

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# client = None
# if API_KEY:
#     client = OpenAI(
#         base_url=API_BASE_URL,
#         api_key=API_KEY
#     )

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

TASK_IDS = ["task_1", "task_2", "task_3"]



def get_llm_response(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


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

def run_episode(task_id: str) -> dict:
    """
    Run one full episode:
      reset → loop(observe → think → act) → grade

    Returns a result dict with task_id, score, steps, and full history.
    """
    print(f"[START] task={task_id} env=agentcorpenv model={MODEL_NAME}", flush=True)

    # 1. Reset environment
    reset_data  = env_reset(task_id)
    task_info   = reset_data["task_info"]
    observation = reset_data["observation"]

    history      = []
    step_results = []

    # 2. Build initial user message
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

        # Trim history to avoid token overflow
        # if len(history) > 8:
        #     history = history[:2] + history[-6:]

        # Ask LLM what to do
        try:
            raw_action = get_llm_response([
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
            ])
        except Exception as e:
            print(f"[STEP] step={step_count} action=null reward=0.00 done=false error={str(e)}", flush=True)
            time.sleep(30)
            continue

        time.sleep(5)

        # Parse JSON action
        try:
            cleaned = raw_action.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            action = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            print(f"[STEP] step={step_count} action=null reward=0.00 done=false error=invalid_json", flush=True)
            history.append({"role": "assistant", "content": raw_action})
            history.append({
                "role":    "user",
                "content": "Your response was not valid JSON. Reply with ONLY a JSON action, nothing else."
            })
            continue

        # Send action to environment
        try:
            step_data = env_step(action)
        except Exception as e:
            print(f"[STEP] step={step_count} action={json.dumps(action)} reward=0.00 done=false error={str(e)}", flush=True)
            break

        reward      = step_data["reward"]
        done        = step_data["done"]
        observation = step_data["observation"]
        result      = step_data["action_result"]

        # ✅ Correct [STEP] log format
        print(
            f"[STEP] step={step_count} action={json.dumps(action)} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True
        )

        step_results.append({
            "step":          step_count,
            "action":        action,
            "action_result": result,
            "reward":        reward,
            "done":          done,
        })

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

    # 4. Final grader score
    grader_result = env_grader()
    final_score   = grader_result["score"]
    breakdown     = grader_result["breakdown"]

    # ✅ Correct [END] log format
    rewards_str = ",".join(f"{r['reward']:.2f}" for r in step_results)
    success     = final_score > 0.0
    print(
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True
    )

    return {
        "task_id":     task_id,
        "score":       final_score,
        "raw":         grader_result.get("raw", 0),
        "steps_taken": step_count,
        "breakdown":   breakdown,
        "history":     step_results,
    }


def run_baseline() -> list[dict]:
    results = []

    for i, task_id in enumerate(TASK_IDS):
        # if i > 0:
        #     print("\nWaiting 20 seconds between tasks to reset rate limit...")
        #     time.sleep(20)
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



# RUN DIRECTLY FOR LOCAL TESTING
# python baseline/inference.py

if __name__ == "__main__":
    run_baseline()