from pydantic import BaseModel,Field
from typing import Literal

class LLMDecision(BaseModel):
    status: Literal[
        "answered",
        "needs_clarification",
        "insufficient_evidence"

    ]
    message : str 
    evidence_id : str 


class ChatRequest(BaseModel):
    message :str = Field(
        min_length=1,
        max_length=400
    )
