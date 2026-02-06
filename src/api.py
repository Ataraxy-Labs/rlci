"""
FastAPI Deployment Server for CI/CD Pipeline RL Agent.

Exposes the trained policy via API endpoints that CI/CD tools can query
in YAML workflows to determine test scope dynamically.

Reference: Section III-F - Real-World Deployment Consideration
"""

import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import DQNAgent
from .environment import ACTION_NAMES, STATE_DIM

app = FastAPI(
    title="RL CI/CD Pipeline Optimizer",
    description="API for dynamically determining test scope using a trained RL agent.",
    version="1.0.0",
)

# Global agent instance
_agent: DQNAgent | None = None
_model_path: str = "models/dqn_agent.pt"


class CommitFeatures(BaseModel):
    """Input features for a commit."""
    diff_size: float = Field(..., ge=0, le=1, description="Normalized diff size")
    developer_id: float = Field(..., ge=0, le=1, description="Normalized developer ID")
    file_types_modified: float = Field(..., ge=0, le=1, description="File type ratio")
    historical_defect_rate: float = Field(..., ge=0, le=1, description="Historical defect rate")
    prior_test_pass_rate: float = Field(..., ge=0, le=1, description="Prior test pass rate")
    time_since_last_commit: float = Field(..., ge=0, le=1, description="Normalized time gap")
    num_files_changed: float = Field(..., ge=0, le=1, description="Normalized file count")
    is_merge_commit: float = Field(..., ge=0, le=1, description="Is merge commit (0 or 1)")
    branch_depth: float = Field(..., ge=0, le=1, description="Branch depth (normalized)")
    code_complexity: float = Field(..., ge=0, le=1, description="Code complexity (normalized)")


class TestScopeResponse(BaseModel):
    """Response with recommended test scope."""
    action: str = Field(..., description="Recommended test scope")
    action_id: int = Field(..., description="Action index (0=full, 1=partial, 2=skip)")
    q_values: list[float] = Field(..., description="Q-values for each action")
    confidence: float = Field(..., description="Confidence in recommendation")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


def get_agent() -> DQNAgent:
    """Get or load the global agent instance."""
    global _agent
    if _agent is None:
        _agent = DQNAgent(device="cpu")
        model_file = Path(_model_path)
        if model_file.exists():
            _agent.load(str(model_file))
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {_model_path}. Train the agent first.",
            )
    return _agent


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    model_loaded = Path(_model_path).exists()
    return HealthResponse(status="healthy", model_loaded=model_loaded)


@app.post("/predict", response_model=TestScopeResponse)
async def predict_test_scope(commit: CommitFeatures):
    """
    Predict optimal test scope for a commit.

    The CI/CD pipeline sends commit metadata and receives a recommendation
    for which test scope to use (full, partial, or skip).
    """
    agent = get_agent()

    state = np.array([
        commit.diff_size,
        commit.developer_id,
        commit.file_types_modified,
        commit.historical_defect_rate,
        commit.prior_test_pass_rate,
        commit.time_since_last_commit,
        commit.num_files_changed,
        commit.is_merge_commit,
        commit.branch_depth,
        commit.code_complexity,
    ], dtype=np.float32)

    action = agent.select_action(state, training=False)
    q_values = agent.get_q_values(state)

    # Confidence: softmax-like measure of how dominant the chosen action is
    q_exp = np.exp(q_values - np.max(q_values))
    q_softmax = q_exp / q_exp.sum()
    confidence = float(q_softmax[action])

    return TestScopeResponse(
        action=ACTION_NAMES[action],
        action_id=action,
        q_values=q_values.tolist(),
        confidence=confidence,
    )


@app.post("/batch_predict")
async def batch_predict(commits: list[CommitFeatures]):
    """Predict test scope for multiple commits at once."""
    agent = get_agent()
    results = []

    for commit in commits:
        state = np.array([
            commit.diff_size,
            commit.developer_id,
            commit.file_types_modified,
            commit.historical_defect_rate,
            commit.prior_test_pass_rate,
            commit.time_since_last_commit,
            commit.num_files_changed,
            commit.is_merge_commit,
            commit.branch_depth,
            commit.code_complexity,
        ], dtype=np.float32)

        action = agent.select_action(state, training=False)
        q_values = agent.get_q_values(state)

        q_exp = np.exp(q_values - np.max(q_values))
        q_softmax = q_exp / q_exp.sum()
        confidence = float(q_softmax[action])

        results.append(TestScopeResponse(
            action=ACTION_NAMES[action],
            action_id=action,
            q_values=q_values.tolist(),
            confidence=confidence,
        ))

    return results


def set_model_path(path: str):
    """Set the model path for the API."""
    global _model_path, _agent
    _model_path = path
    _agent = None  # Force reload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
