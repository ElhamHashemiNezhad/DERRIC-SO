import os
import time
import json
import pickle
import csv
import ray

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from ray.rllib.models import ModelCatalog

from envs.MultiAgentEnvironment import MultiAgentEnvironment
from custom_model import SigmoidModel

# ---------- Register custom model & env ----------
ModelCatalog.register_custom_model("sigmoid_head", SigmoidModel)

def env_creator(config):
    return MultiAgentEnvironment(config)

register_env("MultiAgentEnvironment-v0", env_creator)


# ---------- Simple evaluation ----------
def run_evaluation_episode(algo, env_config, env_seed, num_episodes=5):
    policy_ids = ["orchestrator_policy", "orchcont_policy", "controller_policy"]
    eval_env_config = dict(env_config)
    eval_env_config["seed"] = env_seed
    env = MultiAgentEnvironment(eval_env_config)

    totals = {p: 0.0 for p in policy_ids}
    lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        ep_rewards = {p: 0.0 for p in policy_ids}
        steps = 0

        while not done["__all__"]:
            actions = {}
            for agent_id, agent_obs in obs.items():
                if   agent_id.startswith("orch_"):      pid = "orchestrator_policy"
                elif agent_id.startswith("orchcont_"):  pid = "orchcont_policy"
                elif agent_id.startswith("ctrl_"):      pid = "controller_policy"
                else:
                    continue
                actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=pid, explore=False)

            obs, rewards, done, truncated, infos = env.step(actions)

            for aid, r in rewards.items():
                if   aid.startswith("orch_"):     ep_rewards["orchestrator_policy"] += r
                elif aid.startswith("orchcont_"): ep_rewards["orchcont_policy"] += r
                elif aid.startswith("ctrl_"):     ep_rewards["controller_policy"] += r

            steps += 1

        for p in totals:
            totals[p] += ep_rewards[p]
        lengths.append(steps)

    avg_rewards = {p: totals[p] / num_episodes for p in totals}
    avg_len = sum(lengths) / len(lengths)
    return {"eval_rewards": avg_rewards, "eval_episode_length": avg_len}


def main():
    # ---------- Paths & setup ----------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("results", f"ppo_training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # ---------- Env config (no phase) ----------
    env_seed = 42
    num_iterations = 100
    env_config = {
        "num_hosts": 34,
        "num_orchestrators": 2,
        "num_controllers": 4,
        "num_users": 200,
        "num_base_stations": 15,
        "seed": env_seed,
        "num_iterations": num_iterations,
        "episodes_per_iteration": 1,
        "time_step_duration": 1.0,
        "episode_length": 20,
        "output_dir": output_dir,
    }

    # Small local env to read spaces/ids
    env0 = MultiAgentEnvironment(env_config)
    orchestrator_agent = list(env0.orchestrator_agents)[0]
    orchcont_agent    = list(env0.orchcont_agents)[0]
    controller_agent  = list(env0.controller_agents)[0]

    # ---------- Policies ----------
    policies = {
        "orchestrator_policy": (
            None,
            env0.observation_orchestrator_spaces[orchestrator_agent],
            env0.action_orchestrator_spaces[orchestrator_agent],
            {"model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"}},
        ),
        "orchcont_policy": (
            None,
            env0.observation_orchcont_spaces[orchcont_agent],
            env0.action_orchcont_spaces[orchcont_agent],
            {"model": {"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"}},
        ),

        "controller_policy": (
            None,
            env0.observation_controller_spaces[controller_agent],
            env0.action_controller_spaces[controller_agent],
            {"model": {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}},
        ),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if   agent_id.startswith("orch_"):     return "orchestrator_policy"
        elif agent_id.startswith("orchcont_"): return "orchcont_policy"
        elif agent_id.startswith("ctrl_"):     return "controller_policy"
        raise ValueError(f"Unknown agent_id: {agent_id}")

    # ---------- PPO config ----------
    ppo_config = (
        PPOConfig()
        .environment(env="MultiAgentEnvironment-v0", env_config=env_config)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[
                "orchestrator_policy",
                "orchcont_policy",
                "controller_policy",
            ],
        )
        .training(
            lr=0.0001,
            entropy_coeff=0.01,
            train_batch_size=64,
            sgd_minibatch_size=8,
            num_sgd_iter=10,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            gamma=0.99,
            use_gae=True,
        )
        .rollouts(
            num_rollout_workers=1,
            rollout_fragment_length=64,
        )
        .framework("torch")
        .debugging(log_level="INFO")
    )
    ppo_config.seed = env_seed

    algo = ppo_config.build()


    # ---------- CSV logging setup ----------
    policy_csv_path = os.path.join(output_dir, "policy_rewards.csv")
    with open(policy_csv_path, "w", newline="") as policy_csv:
        policy_writer = csv.writer(policy_csv)
        policy_writer.writerow([
            "iteration",
            "orchestrator_policy",
            "orchcont_policy",
            "controller_policy",
        ])

        best = {pid: {"reward": float("-inf"), "iter": -1, "path": None} for pid in policies}
        eval_every = 5

        # ---------- Training loop ----------
        for it in range(1, num_iterations + 1):
            result = algo.train()
            means = result.get("policy_reward_mean", {})

            # Log rewards per iteration
            policy_writer.writerow([
                it,
                float(means.get("orchestrator_policy", 0.0)),
                float(means.get("orchcont_policy", 0.0)),
                float(means.get("controller_policy", 0.0)),
            ])
            policy_csv.flush()

            print(f"\n== Iter {it}/{num_iterations} ==")
            for pid in policies:
                print(f"{pid}: {means.get(pid, 0.0):.4f}")

        # --- Periodic evaluation ---
        if it % eval_every == 0 or it == num_iterations:
            eval_metrics = run_evaluation_episode(
                algo=algo, env_config=env_config, env_seed=env_seed, num_episodes=10
            )
            print(f"Eval avg len: {eval_metrics['eval_episode_length']:.2f}")
            for pid, r in eval_metrics["eval_rewards"].items():
                print(f"Eval {pid}: {r:.4f}")
                if r > best[pid]["reward"]:
                    best[pid]["reward"] = r
                    best[pid]["iter"] = it

    # ---------- Final checkpoint & summary ----------
    final_checkpoint = algo.save(os.path.join(output_dir, "final_checkpoint"))
    final_checkpoint = str(final_checkpoint)
    print(f"\nTraining complete. Final checkpoint: {final_checkpoint}")

    with open(os.path.join(output_dir, "best_checkpoint_info.json"), "w") as f:
        json.dump(
            {
                "final_checkpoint": final_checkpoint,
                "best": best,
                "env_config": env_config,
                "iterations": num_iterations,
            },
            f,
            indent=2,
        )
    policy_csv.close()


    ray.shutdown()


if __name__ == "__main__":
    main()
