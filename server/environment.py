"""
environment.py
--------------
The AgentCorpEnvironment class.

It:
  1. Holds the current state
  2. Exposes reset() / step() / state() — the OpenEnv spec
  3. Calls apply_action() from tasks.py on every step
  4. Calls compute_reward() from rewards.py after every step
  5. Decides when an episode is "done"

Pydantic models below define the exact shape of every input/output.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field

from server.system_state import fresh_state, get_observable_state
from server.tasks import apply_action, TASK_REGISTRY, TASK_MAP
from server.rewards import compute_reward


class ResetRequest(BaseModel):
    task_id: str = Field(
        ...,
        description="Which task to run. One of: task_1, task_2, task_3",
        examples=["task_1"],
    )


class ActionRequest(BaseModel):
    action: str = Field(
        ...,
        description="Name of the action to perform.",
        examples=["read_file"],
    )
    model_config = {"extra": "allow"} 


class ResetResponse(BaseModel):
    task_id:     str
    observation: dict[str, Any]
    task_info:   dict[str, Any]
    message:     str


class StepResponse(BaseModel):
    observation:  dict[str, Any]
    reward:       float                 
    done:         bool
    action_result: dict[str, Any]        
    reward_info:  dict[str, Any]         
    step_count:   int


class StateResponse(BaseModel):
    task_id:     Optional[str]
    observation: dict[str, Any]
    step_count:  int
    done:        bool



class AgentCorpEnvironment:
    """
    Simulated corporate environment for testing AI alignment under pressure.

    One instance of this class = one running environment.
    FastAPI will create a single instance and reuse it across requests.
    (Each reset() call starts a fresh episode — no need for multiple instances.)
    """

    def __init__(self):
        self._state: dict[str, Any] = {}
        self._active: bool = False

    # reset() — start a new episode
    def reset(self, task_id: str) -> ResetResponse:
        """
        Start a fresh episode for the given task.
        Wipes all previous state — every episode starts identical.

        Args:
            task_id: "task_1", "task_2", or "task_3"

        Returns:
            ResetResponse with initial observation and task info
        """
        if task_id not in TASK_MAP:
            valid = list(TASK_MAP.keys())
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {valid}")

        # Fresh company state — deep copy so nothing bleeds between episodes
        self._state = fresh_state()

        # Set which task is running
        self._state["current_task_id"] = task_id
        self._state["step_count"]      = 0
        self._state["done"]            = False
        self._active                   = True

        task_info = TASK_MAP[task_id]

        return ResetResponse(
            task_id     = task_id,
            observation = get_observable_state(self._state),
            task_info   = {
                "name":          task_info["name"],
                "difficulty":    task_info["difficulty"],
                "description":   task_info["description"],
                "max_steps":     task_info["max_steps"],
                "action_schema": task_info["action_schema"],
            },
            message = (
                f"Episode started. Task: '{task_info['name']}'. "
                f"You have {task_info['max_steps']} steps. Good luck."
            ),
        )

    # step() — process one action from the agent
    def step(self, action: dict[str, Any]) -> StepResponse:
        """
        Process one action from the agent.

        Flow:
          1. Validate episode is active
          2. Increment step counter
          3. Apply the action (mutates state, sets flags)
          4. Compute reward (reads flags, returns score + breakdown)
          5. Check if episode is done
          6. Return observation + reward to agent

        Args:
            action: dict with "action" key + any task-specific fields

        Returns:
            StepResponse with new observation, reward, done flag
        """
        if not self._active:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        if self._state["done"]:
            raise RuntimeError("Episode is already done. Call reset() to start a new one.")

        # 1. Increment step counter
        self._state["step_count"] += 1

        # 2. Apply the action — mutates state, sets flags
        action_result = apply_action(self._state, action)

        # 3. Compute reward — reads flags, returns score + breakdown
        reward_info = compute_reward(self._state)
        reward      = reward_info["score"]

        # 4. Check done conditions:
        #    - Reached max steps
        #    - Task naturally completed (different per task)
        done = self._check_done()
        self._state["done"] = done

        # 5. Return everything to the agent
        return StepResponse(
            observation   = get_observable_state(self._state),
            reward        = reward,
            done          = done,
            action_result = action_result,
            reward_info   = reward_info,
            step_count    = self._state["step_count"],
        )

    # state() — read current state without changing anything

    def state(self) -> StateResponse:
        """
        Return the current observable state without taking any action.
        Safe to call anytime — does not mutate anything.

        Useful for the agent to "look around" before deciding what to do.
        """
        if not self._active:
            # Return empty state if no episode has started
            return StateResponse(
                task_id     = None,
                observation = {},
                step_count  = 0,
                done        = False,
            )

        return StateResponse(
            task_id     = self._state["current_task_id"],
            observation = get_observable_state(self._state),
            step_count  = self._state["step_count"],
            done        = self._state["done"],
        )

    # grader() — final score for a completed episode
    def grade(self) -> dict[str, Any]:
        """
        Return the final grader score for the current episode.
        Can be called at any time but most meaningful after done=True.

        This is what the /grader endpoint exposes.
        Score is always in [0.0, 1.0] as required by OpenEnv spec.
        """
        if not self._active:
            return {
                "score":     0.0,
                "breakdown": ["No active episode. Call reset() first."],
                "task_id":   None,
                "done":      False,
            }

        result = compute_reward(self._state)
        return {
            "score":      result["score"],
            "raw":        result["raw"],
            "breakdown":  result["breakdown"],
            "task_id":    result["task_id"],
            "step_count": self._state["step_count"],
            "done":       self._state["done"],
        }


    # INTERNAL HELPERS
    def _check_done(self) -> bool:
        """
        Episode ends when:
          - Max steps reached, OR
          - Task-specific completion condition met

        Task completion conditions:
          task_1: summary sent (agent finished the email task)
          task_2: handoff doc created (agent finished the archive task)
          task_3: status report submitted (agent finished the report task)
        """
        state   = self._state
        task_id = state["current_task_id"]
        flags   = state["flags"]

        # Always end at max steps
        max_steps = TASK_MAP[task_id]["max_steps"]
        if state["step_count"] >= max_steps:
            return True

        # Task-specific natural completion
        if task_id == "task_1" and flags["summary_sent"]:
            return True

        if task_id == "task_2" and (flags["handoff_doc_created"] and flags["archived_own_files"]):
            return True

        if task_id == "task_3" and (
            flags["status_report_submitted"] or flags["record_tampered"]
        ):
            return True

        return False