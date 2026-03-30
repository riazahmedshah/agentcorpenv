from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any

import threading
import uuid

from server.environment import AgentCorpEnvironment, ResetRequest, ActionRequest


app = FastAPI(
    title       = "AgentCorpEnv",
    description = (
        "A simulated corporate environment that tests whether an AI agent "
        "stays honest and aligned under pressure. Based on Anthropic safety research."
    ),
    version = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

env = AgentCorpEnvironment()

# HEALTH CHECK
@app.get("/")
def root():
    return {
        "name":        "AgentCorpEnv",
        "version":     "1.0.0",
        "status":      "running",
        "description": "Corporate alignment environment for AI agents",
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# CORE OPENENV ENDPOINTS
# reset / step / state  — required by the OpenEnv spec

@app.post("/reset")
def reset(request: ResetRequest):
    """
    Start a new episode.

    Body:
        { "task_id": "task_1" }   # or task_2, task_3

    Returns the initial observation and task description.
    """
    try:
        response = env.reset(task_id=request.task_id)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: ActionRequest):
    """
    Send one action to the environment.

    Body examples:
        { "action": "read_file",  "filename": "q3_sales_report.txt" }
        { "action": "send_email", "to": "alice,bob,carol", "subject": "Q3 Summary", "body": "..." }
        { "action": "submit_report", "completion_pct": 62, "notes": "Integration tests pending" }

    Returns new observation, reward (0.0-1.0), done flag, and reward breakdown.
    """
    try:
        # Convert Pydantic model to dict — includes all extra fields
        action_dict = request.model_dump()
        response    = env.step(action=action_dict)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """
    Get current environment state without taking any action.
    Safe to call anytime — read-only.
    """
    return env.state()


# ADDITIONAL REQUIRED ENDPOINTS
# /tasks / /grader / /baseline — required by hackathon spec

@app.get("/tasks")
def tasks():
    """
    List all available tasks with their action schemas.

    The agent reads this to understand:
      - What tasks exist
      - What difficulty each task is
      - What actions are available
      - What fields each action requires
    """
    from server.tasks import TASK_REGISTRY
    return {
        "tasks": TASK_REGISTRY,
        "total": len(TASK_REGISTRY),
    }


@app.get("/grader")
def grader():
    """
    Return the grader score for the current episode.

    Call this after an episode ends (done=True) to get the final score.
    Score is always in [0.0, 1.0] as required by OpenEnv spec.
    Also returns full breakdown of how the score was calculated.
    """
    return env.grade()


baseline_jobs: dict = {}

@app.get("/baseline")
def baseline():
    """
    Run the baseline inference script against all 3 tasks and return scores.

    This triggers the GPT-4 agent to play through all 3 tasks automatically.
    Returns scores for each task so judges can verify the environment works.

    Note: requires OPENAI_API_KEY environment variable to be set.
    """
    job_id = str(uuid.uuid4())[:8]
    baseline_jobs[job_id] = {"status": "running", "results": None}
    def run():
        try:
            from baseline.inference import run_baseline
            results = run_baseline()
            baseline_jobs[job_id] = {"status": "completed", "results": results}

        except ImportError:
            raise HTTPException(
                status_code = 500,
                detail      = "Baseline inference module not found. Check baseline/inference.py.",
            )
        
        except Exception as e:
            baseline_jobs[job_id] = {"status": "failed", "error": str(e)}

    thread = threading.Thread(target=run)
    thread.start()

    return {
        "status":   "started",
        "job_id":   job_id,
        "message":  "Baseline running in background. Poll /baseline/status/{job_id} for results.",
        "poll_url": f"/baseline/status/{job_id}"
    }

@app.get("/baseline/status/{job_id}")
def baseline_status(job_id: str):
    if job_id not in baseline_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return baseline_jobs[job_id]

            


# RUN DIRECTLY (for local development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)