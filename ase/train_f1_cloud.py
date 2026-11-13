"""Cloud entry for F1 AMP/ASE; invoked by gm-run from any working directory.

Example: gm-run F1_ase/ase/train_f1_cloud.py --install-deps --max_iterations 200
Keeps the image's PyTorch/Isaac Gym and caches legacy ASE dependencies outside
the platform's /workspace artifact scanner.
"""
import argparse
from collections import deque
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "numpy": "1.23.5", "rl-games": "1.1.4", "gym": "0.23.1",
    "ray": "1.13.0", "protobuf": "3.20.3", "tensorboardX": "2.6.2.2",
    "termcolor": "1.1.0", "PyYAML": "6.0.1", "gym-notices": "0.0.8",
    "tensorboard": "2.11.2", "psutil": "5.9.8", "setproctitle": "1.3.3",
}


def format_training_log(stats):
    """Render measured iteration statistics without importing the simulator."""
    def duration(seconds):
        seconds = max(0, int(seconds))
        return "{:02d}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)

    def row(label, value):
        return "{:>43} {}".format(label + ":", value)

    width = 88
    lines = ["#" * width,
             "Learning iteration {}/{}".format(stats["epoch"], stats["max_epochs"]).center(width),
             row("Computation", "{:.0f} steps/s (collection: {:.3f}s, learning: {:.3f}s)".format(
                 stats["fps"], stats["play_time"], stats["update_time"]))]
    for label, value in stats["metrics"].items():
        display = "{:.2e}".format(value) if label == "PPO/learning_rate" else "{:.4f}".format(value)
        lines.append(row(label, display))
    for name, count in stats["terminations"].items():
        lines.append(row("Episode_Termination/" + name, "{} / rollout".format(count)))
    lines += ["-" * width,
              row("Total timesteps", str(stats["frame"])),
              row("Iteration time", "{:.3f}s".format(stats["iteration_time"])),
              row("Time elapsed", duration(stats["elapsed_s"])),
              row("ETA", duration(stats["eta_s"]))]
    return "\n".join(lines)


def mean_metric(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    return sum(float(v.detach().float().mean()) if hasattr(v, "detach") else float(v)
               for v in values) / len(values)


def discriminator_log_metrics(train_info):
    """Measured minibatch diagnostics shared by console logging and the smoke check."""
    labels = {"disc_agent_acc": "policy_accuracy", "disc_demo_acc": "demo_accuracy",
              "disc_agent_logit": "policy_logit", "disc_demo_logit": "demo_logit",
              "disc_grad_penalty": "gradient_penalty"}
    return {"Discriminator/" + label: mean_metric(train_info[key])
            for key, label in labels.items() if key in train_info}


def prepare_dependencies(install):
    deps = Path(os.environ.get("F1_DEPS_DIR", str(Path(tempfile.gettempdir()) / "f1_ase_deps")))
    if install:
        installed = {d.metadata["Name"].lower(): d.version
                     for d in metadata.distributions(path=[str(deps)])}
        if any(installed.get(name.lower()) != version for name, version in PACKAGES.items()):
            pip = [
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "--upgrade", "--target", str(deps), "--timeout", "30", "--retries", "2",
            ]
            # Resolve Ray/Gym dependencies, including msgpack. Install rl-games
            # separately so its broad torch requirement cannot replace the image.
            groups = [
                [name + "==" + version for name, version in PACKAGES.items() if name != "rl-games"],
                ["--no-deps", "rl-games==" + PACKAGES["rl-games"]],
            ]
            for group in groups:
                for index in ("https://pypi.org/simple", "https://mirrors.aliyun.com/pypi/simple/"):
                    print("F1_DEPENDENCIES installing from " + index, flush=True)
                    result = subprocess.run(pip + ["--index-url", index] + group)
                    if result.returncode == 0:
                        break
                else:
                    raise RuntimeError("F1 dependency installation failed")
    if deps.is_dir():
        sys.path.insert(0, str(deps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="output")
    parser.add_argument("--algorithm", choices=("ase", "amp"), default="ase")
    parser.add_argument("--task", choices=("HumanoidF1", "HumanoidF1Velocity"), default="HumanoidF1")
    parser.add_argument("--velocity-stage", choices=("A", "B", "C"),
                        help="Override curriculum; otherwise resume saved stage (fresh ASE: A, AMP: B)")
    parser.add_argument("--cfg-env", help="Environment YAML; defaults to the selected F1 task")
    parser.add_argument("--cfg-train", help="Training YAML; defaults to the selected task/algorithm")
    parser.add_argument("--motion-file", help="Reference NPZ or YAML; defaults to the selected F1 task")
    parser.add_argument("--json-logs", action="store_true", help="Also emit F1_TRAIN JSON lines for automation")
    args, extra = parser.parse_known_args()
    velocity_task = args.task == "HumanoidF1Velocity"
    if args.max_iterations is None:
        args.max_iterations = 500 if velocity_task else 200
    if args.max_iterations < 1 or args.num_envs < 1:
        parser.error("max_iterations and num_envs must be positive")
    if args.velocity_stage is not None and not velocity_task:
        parser.error("--velocity-stage requires HumanoidF1Velocity")
    velocity_env = "humanoid_f1_velocity_amp.yaml" if args.algorithm == "amp" else "humanoid_f1_velocity.yaml"
    args.cfg_env = args.cfg_env or ("ase/data/cfg/" + velocity_env if velocity_task else
                                     "ase/data/cfg/humanoid_f1.yaml")
    args.cfg_train = args.cfg_train or ("ase/data/cfg/train/rlg/" + args.algorithm + "_f1_velocity.yaml" if velocity_task else
                                         "ase/data/cfg/train/rlg/" + args.algorithm + "_f1.yaml")
    args.motion_file = args.motion_file or ("ase/data/motions/f1/dataset_f1_walk_stop.yaml" if velocity_task else
                                             "ase/data/motions/f1/dataset_f1.yaml")

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    prepare_dependencies(args.install_deps)
    os.chdir(str(ROOT))
    sys.path.insert(0, str(ROOT / "ase"))

    # Isaac Gym must load before torch (run imports it through utils.config).
    import run as entry
    import torch

    class CloudObserver(entry.RLGPUAlgoObserver):
        def process_infos(self, infos, done_indices):
            super().process_infos(infos, done_indices)
            if not hasattr(self, "_termination_counts"):
                self._termination_counts = {}
            for name, mask in infos.get("termination_reasons", {}).items():
                count = mask.sum().detach()
                self._termination_counts[name] = self._termination_counts.get(name, 0) + count
            if not hasattr(self, "_motion_totals"):
                self._motion_totals = {}
            for name, values in infos.get("motion_metrics", {}).items():
                if name == "completed_displacement_m":
                    values = values[done_indices.reshape(-1)]
                if values.numel():
                    total, count = self._motion_totals.get(name, (0, 0))
                    self._motion_totals[name] = (total + values.sum().detach(), count + values.numel())
            if not hasattr(self, "_component_totals"):
                self._component_totals = {}
            for group in ("reward_components", "event_metrics", "gait_events"):
                for name, values in infos.get(group, {}).items():
                    if values.numel():
                        tag = ("Task_Reward/" if group == "reward_components" else "Metrics/event/") + name
                        total, count = self._component_totals.get(tag, (0, 0))
                        self._component_totals[tag] = (total + values.sum().detach(), count + values.numel())

        def after_print_stats(self, frame, epoch_num, total_time):
            super().after_print_stats(frame, epoch_num, total_time)
            result = {"epoch": epoch_num, "max_epochs": self.algo.max_epochs,
                      "elapsed_s": round(total_time, 2),
                      "mean_reward": self.algo._mean_training_reward, "reward_unit": "per_step"}
            self.writer.add_scalar("Train/mean_reward", result["mean_reward"], epoch_num)
            self.writer.add_scalar("Train/mean_reward_per_step", result["mean_reward"], epoch_num)
            if self.algo.game_rewards.current_size > 0:
                reward = float(self.algo._get_mean_rewards().mean())
                length = float(self.algo.game_lengths.get_mean())
                result.update(env_episode_return=reward, mean_episode_length=length,
                              survival_s=round(length * self.algo.vec_env.env.task.dt, 3))
                self.writer.add_scalar("Env/mean_episode_return", reward, epoch_num)
                self.writer.add_scalar("Train/mean_episode_length", length, epoch_num)
            self._last_summary = result
            self._last_termination_counts = {
                name: int(count) for name, count in getattr(self, "_termination_counts", {}).items()
            }
            for name, count in self._last_termination_counts.items():
                self.writer.add_scalar("Episode_Termination/" + name, count, frame)
            self._termination_counts = {}
            self._last_motion_metrics = {
                "Metrics/motion/" + name: float(total / count)
                for name, (total, count) in getattr(self, "_motion_totals", {}).items()
            }
            for name, value in self._last_motion_metrics.items():
                self.writer.add_scalar(name, value, frame)
            self._motion_totals = {}
            self._last_components = {name: float(total / count)
                                     for name, (total, count) in getattr(self, "_component_totals", {}).items()}
            for name, value in self._last_components.items():
                self.writer.add_scalar(name, value, frame)
            self._component_totals = {}
            result["motion"] = self._last_motion_metrics

        def format_train_stats(self, train_info, epoch_num, frame, total_time):
            if not hasattr(self, "_iteration_times"):
                self._iteration_times = deque(maxlen=20)
            self._iteration_times.append(train_info["total_time"])
            metrics = {}
            metrics.update(self._last_motion_metrics)
            metrics.update(self._last_components)
            metrics.update({"Reward/weighted_" + name: value for name, value in
                            self.algo._training_reward_components.items()})
            metrics.update(discriminator_log_metrics(train_info))
            metrics.update(getattr(self.algo, "_discriminator_phase_metrics", {}))
            sigmas = self.algo.experience_buffer.tensor_dict.get("sigmas")
            if sigmas is not None:
                metrics["Mean action noise std"] = mean_metric(sigmas)
            for label, key in (("Mean value_function loss", "critic_loss"),
                               ("Mean surrogate loss", "actor_loss"),
                               ("Mean policy entropy", "entropy"),
                               ("Mean bounds loss", "b_loss"),
                               ("Mean discriminator loss", "disc_loss"),
                               ("Mean encoder loss", "enc_loss")):
                if key in train_info:
                    metrics[label] = mean_metric(train_info[key])
            summary = self._last_summary
            metrics["Mean reward (training, per step)"] = summary["mean_reward"]
            if "mean_episode_length" in summary:
                metrics.update({"Mean episode length (steps)": summary["mean_episode_length"],
                                "Mean episode duration (s)": summary["survival_s"]})
            disc_reward = mean_metric(train_info["disc_rewards"])
            enc_reward = mean_metric(train_info["enc_rewards"]) if "enc_rewards" in train_info else 0.0
            reward_prefix = args.algorithm.upper() + "_Reward/"
            metrics.update({
                reward_prefix + "discriminator": disc_reward,
                "PPO/learning_rate": self.algo.last_lr,
                "PPO/clip_fraction": mean_metric(train_info["actor_clip_frac"]),
            })
            if "enc_rewards" in train_info:
                metrics[reward_prefix + "encoder"] = enc_reward
            if "enc_rewards_raw" in train_info:
                metrics[reward_prefix + "encoder_raw"] = mean_metric(train_info["enc_rewards_raw"])
            block = format_training_log({
                "epoch": epoch_num, "max_epochs": self.algo.max_epochs, "frame": frame,
                "fps": self.algo.curr_frames / train_info["total_time"],
                "play_time": train_info["play_time"], "update_time": train_info["update_time"],
                "iteration_time": train_info["total_time"], "elapsed_s": total_time,
                "eta_s": max(0, self.algo.max_epochs - epoch_num)
                    * sum(self._iteration_times) / len(self._iteration_times),
                "metrics": metrics, "terminations": self._last_termination_counts,
            })
            if args.json_logs:
                block += "\nF1_TRAIN " + json.dumps(summary)
            return block

    entry.RLGPUAlgoObserver = CloudObserver
    print("F1_RUNTIME " + json.dumps({
        "python": sys.version.split()[0], "torch": torch.__version__,
        "num_envs": args.num_envs, "max_iterations": args.max_iterations,
        "seed": args.seed, "control_hz": 50,
        "algorithm": args.algorithm, "motion_file": args.motion_file,
        "task": args.task, "cfg_env": args.cfg_env, "cfg_train": args.cfg_train,
        "velocity_stage_override": args.velocity_stage,
    }), flush=True)
    sys.argv = [
        str(ROOT / "ase/run.py"), "--task", args.task, "--headless",
        "--sim_device", "cuda:0", "--pipeline", "gpu", "--rl_device", "cuda:0",
        "--cfg_env", args.cfg_env,
        "--cfg_train", args.cfg_train,
        "--motion_file", args.motion_file,
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iterations),
        "--seed", str(args.seed), "--output_path", args.output_path,
        *(["--velocity_stage", args.velocity_stage] if args.velocity_stage is not None else []), *extra,
    ]
    entry.main()


if __name__ == "__main__":
    main()
