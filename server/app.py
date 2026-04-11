from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.environment import AgentCorpEnvironment
from server.models import ResetRequest, ActionRequest



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


@app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Start a new episode.

    Body:
        { "task_id": "task_1" }   # or task_2, task_3

    Returns the initial observation and task description.
    """
    try:
        task_id = request.task_id if request else "task_1"
        response = env.reset(task_id=task_id)
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
    Score is always in (0.0, 1.0) as required by OpenEnv spec.
    Also returns full breakdown of how the score was calculated.
    """
    return env.grade()


@app.get("/baseline")
def baseline():
    """
    Run the baseline inference script against all 3 tasks and return scores.

    This triggers the GPT-4 agent to play through all 3 tasks automatically.
    Returns scores for each task so judges can verify the environment works.
    """
    try:
        from inference import run_baseline
        results = run_baseline()
        return {
            "status": "completed",
            "results": results,
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Baseline inference module not found. Check inference.py.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

            
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
