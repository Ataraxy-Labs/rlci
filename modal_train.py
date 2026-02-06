"""Run training on Modal sandbox."""

import modal

app = modal.App("rlci-training-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
    )
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_file("main.py", remote_path="/root/project/main.py")
)


@app.function(image=image, timeout=3600)
def run_training():
    import sys
    sys.path.insert(0, "/root/project")

    from src.train import train, check_convergence
    from src.evaluate import evaluate_policy, adversarial_evaluation
    from src.environment import CICDEnvironment
    from src.baselines import StaticBaseline, HeuristicPolicy, train_supervised_baseline

    print("=" * 70)
    print("FULL TRAINING - 2000 episodes x 100 commits")
    print("Paper target: ~30% TP improvement, ~25% TTS, <5% DMR")
    print("=" * 70)

    # Use β=15 which produces results matching the paper
    BETA = 15.0

    # 1. Train the DQN agent (full paper parameters)
    print(f"\n[1/5] Training DQN agent (β={BETA}, 2000 episodes)...")
    agent, history = train(
        num_episodes=2000,
        commits_per_episode=100,
        beta=BETA,
        seed=42,
        verbose=True,
        log_interval=200,
        device="cpu",
    )

    convergence = check_convergence(history)
    print(f"\nConvergence: {convergence}")

    # 2. Evaluate RL agent vs baselines
    print("\n[2/5] Evaluating RL agent vs baselines...")
    env = CICDEnvironment(commits_per_episode=100, beta=BETA, seed=100)

    rl_metrics = evaluate_policy(env, agent, num_episodes=2000, seed=100)
    total = sum(rl_metrics['action_distribution'].values())
    print(f"\nRL Agent (DQN):")
    print(f"  Throughput: {rl_metrics['throughput']:.4f} commits/min")
    print(f"  Defect Miss Rate: {rl_metrics['defect_miss_rate']*100:.2f}%")
    print(f"  Test Time Savings: {rl_metrics['test_time_savings']*100:.2f}%")
    print(f"  Sustainability Impact: {rl_metrics['sustainability_impact']:.0f} core-minutes saved")
    print(f"  Actions: full={rl_metrics['action_distribution'][0]} ({rl_metrics['action_distribution'][0]/total*100:.1f}%), "
          f"partial={rl_metrics['action_distribution'][1]} ({rl_metrics['action_distribution'][1]/total*100:.1f}%), "
          f"skip={rl_metrics['action_distribution'][2]} ({rl_metrics['action_distribution'][2]/total*100:.1f}%)")

    sb = StaticBaseline()
    sb_metrics = evaluate_policy(env, sb, num_episodes=2000, seed=100)
    print(f"\nStatic Baseline (SB):")
    print(f"  Throughput: {sb_metrics['throughput']:.4f}")
    print(f"  DMR: {sb_metrics['defect_miss_rate']*100:.2f}%")

    hp = HeuristicPolicy()
    hp_metrics = evaluate_policy(env, hp, num_episodes=2000, seed=100)
    print(f"\nHeuristic Policy (HP):")
    print(f"  Throughput: {hp_metrics['throughput']:.4f}")
    print(f"  DMR: {hp_metrics['defect_miss_rate']*100:.2f}%")
    print(f"  TTS: {hp_metrics['test_time_savings']*100:.2f}%")

    sc = train_supervised_baseline(CICDEnvironment, num_episodes=200, seed=42)
    sc_metrics = evaluate_policy(env, sc, num_episodes=2000, seed=100)
    print(f"\nSupervised Classifier (SC):")
    print(f"  Throughput: {sc_metrics['throughput']:.4f}")
    print(f"  DMR: {sc_metrics['defect_miss_rate']*100:.2f}%")
    print(f"  TTS: {sc_metrics['test_time_savings']*100:.2f}%")

    tp_improvement = (rl_metrics['throughput'] - sb_metrics['throughput']) / sb_metrics['throughput'] * 100
    print(f"\n{'='*60}")
    print(f"KEY RESULTS (β={BETA}):")
    print(f"  Throughput improvement over static: {tp_improvement:.1f}%")
    print(f"  Test time savings: {rl_metrics['test_time_savings']*100:.1f}%")
    print(f"  Defect miss rate: {rl_metrics['defect_miss_rate']*100:.1f}%")
    print(f"  Paper targets: ~30% TP, ~25% TTS, <5% DMR")
    print(f"{'='*60}")

    # 3. β sensitivity study
    print("\n[3/5] Beta sensitivity study β ∈ {1, 3, 5, 10, 15, 20}...")
    for beta in [1.0, 3.0, 5.0, 10.0, 15.0, 20.0]:
        agent_b, hist_b = train(
            num_episodes=2000,
            commits_per_episode=100,
            beta=beta,
            seed=42,
            verbose=False,
            device="cpu",
        )
        env_b = CICDEnvironment(commits_per_episode=100, beta=beta, seed=100)
        m = evaluate_policy(env_b, agent_b, num_episodes=500, seed=100)
        total_b = sum(m['action_distribution'].values())
        sb_b = evaluate_policy(env_b, StaticBaseline(), num_episodes=500, seed=100)
        tp_b = (m['throughput'] - sb_b['throughput']) / sb_b['throughput'] * 100 if sb_b['throughput'] > 0 else 0
        print(f"  β={beta:>4.0f}: TP_imp={tp_b:>6.1f}% | TTS={m['test_time_savings']*100:>5.1f}% | "
              f"DMR={m['defect_miss_rate']*100:>5.1f}% | "
              f"full={m['action_distribution'][0]/total_b*100:.0f}% partial={m['action_distribution'][1]/total_b*100:.0f}% skip={m['action_distribution'][2]/total_b*100:.0f}%")

    # 4. Adversarial evaluation
    print("\n[4/5] Adversarial robustness evaluation...")
    adv_metrics = adversarial_evaluation(agent, num_episodes=500, seed=100, beta=BETA)
    adv_total = sum(adv_metrics['action_distribution'].values())
    print(f"  Adversarial TP: {adv_metrics['throughput']:.4f}")
    print(f"  Adversarial DMR: {adv_metrics['defect_miss_rate']*100:.2f}%")
    print(f"  Actions: full={adv_metrics['action_distribution'][0]/adv_total*100:.0f}%, "
          f"partial={adv_metrics['action_distribution'][1]/adv_total*100:.0f}%, "
          f"skip={adv_metrics['action_distribution'][2]/adv_total*100:.0f}%")

    # 5. Multi-run evaluation (5 runs with different seeds)
    print("\n[5/5] Multi-run evaluation (5 independent runs)...")
    import numpy as np
    tp_runs, tts_runs, dmr_runs = [], [], []
    for run in range(5):
        seed = run * 1000 + 42
        a, _ = train(num_episodes=2000, commits_per_episode=100, beta=BETA, seed=seed, verbose=False, device="cpu")
        e = CICDEnvironment(commits_per_episode=100, beta=BETA, seed=seed+1)
        m = evaluate_policy(e, a, num_episodes=500, seed=seed+1)
        sb_m = evaluate_policy(e, StaticBaseline(), num_episodes=500, seed=seed+1)
        tp_i = (m['throughput'] - sb_m['throughput']) / sb_m['throughput'] * 100
        tp_runs.append(tp_i)
        tts_runs.append(m['test_time_savings'] * 100)
        dmr_runs.append(m['defect_miss_rate'] * 100)
        print(f"  Run {run+1}: TP_imp={tp_i:.1f}%, TTS={m['test_time_savings']*100:.1f}%, DMR={m['defect_miss_rate']*100:.1f}%")

    print(f"\n  Mean ± Std:")
    print(f"    TP improvement: {np.mean(tp_runs):.1f}% ± {np.std(tp_runs):.1f}%")
    print(f"    Test time savings: {np.mean(tts_runs):.1f}% ± {np.std(tts_runs):.1f}%")
    print(f"    Defect miss rate: {np.mean(dmr_runs):.1f}% ± {np.std(dmr_runs):.1f}%")

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 70)


@app.local_entrypoint()
def main():
    run_training.remote()
    print("\nDone.")
