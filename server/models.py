from pydantic import BaseModel, Field
from typing import Any, Optional

class ResetRequest(BaseModel):
    task_id: str = Field(
        default="task_1",
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

class AgentAction(BaseModel):
    action: str = Field(..., description="Action to perform")
    model_config = {"extra": "allow"}

class AgentObservation(BaseModel):
    current_task_id: Optional[str] = None
    step_count: int = 0
    max_steps: int = 20
    done: bool = False
    employees: dict = {}
    inbox: list = []
    outbox: list = []
    files: dict = {}
    archive: list = []
    my_performance: dict = {}

class StepResult(BaseModel):
    observation: AgentObservation
    reward: float
    done: bool
    action_result: dict[str, Any]
    reward_info: dict[str, Any]
    step_count: int