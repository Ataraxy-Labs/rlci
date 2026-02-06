# RL-based CI/CD Pipeline Optimizer

Implementation of [arXiv:2601.11647](https://arxiv.org/abs/2601.11647) — *Reinforcement Learning for Dynamic Workflow Optimization in CI/CD Pipelines*.

A DQN agent learns to dynamically select test scope (full, partial, or skip) for each commit based on its metadata, optimizing the tradeoff between pipeline throughput and defect detection.

## Results

Verified on Modal cloud (2000 episodes, 100 commits/episode):

| Metric | Paper Target | Our Result |
|--------|-------------|-----------|
| Throughput Improvement | ~30% | **32.1%** |
| Test Time Savings | ~25% | **24.3%** |
| Defect Miss Rate | <5% | **3.0%** |

The agent learns a state-dependent policy: **68% full tests** on risky commits, **25% partial** on safe commits, **7% skip** on trivially safe commits.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Ataraxy-Labs/rlci.git
cd rlci
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Quick training (reduced parameters, ~2 min)
python main.py --quick

# Full training (paper parameters, ~30 min)
python main.py
```

## Project Structure

```
src/
  environment.py   # MDP simulation (10-dim state, 3 actions, reward function)
  agent.py         # DQN with replay buffer and target network
  baselines.py     # Static, Heuristic, and Supervised Classifier baselines
  train.py         # Training pipeline with convergence checking
  evaluate.py      # Evaluation metrics (TP, DMR, TTS, SI) and comparison
  api.py           # FastAPI server for policy deployment
  visualize.py     # Matplotlib visualization plots
main.py            # CLI entry point (full pipeline)
modal_train.py     # Cloud training on Modal
tests/             # 85 tests across 6 test files
```

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | 10-dimensional normalized vector: diff size, developer ID, file types, historical defect rate, prior pass rate, time gap, files changed, merge flag, branch depth, complexity |
| **Actions** | `full_test` (10 min, 100% detection), `partial_test` (3 min, 70%), `skip_test` (0 min, 0%) |
| **Reward** | R = -(t_exec / T_full) - &beta; &middot; I_bug_escaped |
| **Discount** | &gamma; = 0.99 |

## Training

### Local

```bash
# Default: 2000 episodes, beta=20, seed=42
python main.py

# Custom parameters
python main.py --episodes 2000 --beta 20 --seed 42 --commits 100

# Quick mode for testing
python main.py --quick
```

### Modal (Cloud)

```bash
pip install modal
modal run modal_train.py
```

Runs full training + evaluation (baselines, beta sensitivity, adversarial, 5-run stats) on Modal's cloud infrastructure. Takes ~1 hour.

## Evaluation

The pipeline runs 5 evaluation stages:

1. **DQN Training** - 2000 episodes with epsilon decay (1.0 &rarr; 0.1)
2. **Baseline Comparison** - RL agent vs Static, Heuristic, and Supervised Classifier over 5 independent runs
3. **&beta; Sensitivity Study** - Sweep &beta; &isin; {1, 3, 5, 10, 15, 20, 30, 40} showing the safety-speed tradeoff
4. **Adversarial Robustness** - Tests against deceptive low-diff commits with hidden bugs
5. **Results & Plots** - Saves metrics to `results/` and plots to `plots/`

## API Deployment

```bash
# Start the API server
uvicorn src.api:app --reload

# Query the policy
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "diff_size": 0.3,
    "developer_id": 0.5,
    "file_types_modified": 0.4,
    "historical_defect_rate": 0.1,
    "prior_test_pass_rate": 0.9,
    "time_since_last_commit": 0.2,
    "num_files_changed": 0.15,
    "is_merge_commit": 0.0,
    "branch_depth": 0.1,
    "code_complexity": 0.2
  }'
```

Response:
```json
{
  "action": "partial_test",
  "action_id": 1,
  "q_values": [-0.98, -0.31, -2.45],
  "confidence": 0.87
}
```

## Key Findings

- **&beta; controls the safety-speed tradeoff**: Low &beta; (&lt;10) leads to aggressive skipping with high defect miss rates. High &beta; (&gt;40) leads to near-100% full testing with minimal time savings. The sweet spot is &beta;=15-20.
- **Bimodal commit distribution**: Real CI/CD pipelines have ~35% trivially safe commits (config, docs) and ~65% substantive code changes. The agent learns to exploit this structure.
- **Reward normalization is critical**: Normalizing execution time by T_full keeps the reward components on comparable scales, allowing &beta; to effectively control the tradeoff.

## &beta; Sensitivity

```
 beta |  TP_imp |   TTS |  DMR | Policy
------+---------+-------+------+---------------------------
    1 | 75658%  | 99.9% | 99.4%| 100% skip
    5 |   400%  | 80.0% | 37.3%| 67% partial, 33% skip
   10 |   116%  | 53.7% | 19.8%| 31% full, 51% partial
   15 |    33%  | 24.6% |  2.9%| 66% full, 32% partial
   20 |    33%  | 24.5% |  3.1%| 68% full, 25% partial
   30 |    29%  | 22.5% |  2.2%| 68% full, 30% partial
   40 |    17%  | 14.5% |  1.2%| 79% full, 20% partial
```

## Citation

```bibtex
@article{rlcicd2025,
  title={Reinforcement Learning for Dynamic Workflow Optimization in CI/CD Pipelines},
  journal={arXiv preprint arXiv:2601.11647},
  year={2025}
}
```

## License

MIT
