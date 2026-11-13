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
import hashlib
import shutil
import urllib.request


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
    """Policy samples are negative, reference samples are positive."""
    return {"Discriminator/" + label: mean_metric(train_info[key])
            for label, key in (("policy_accuracy", "disc_agent_acc"),
                               ("demo_accuracy", "disc_demo_acc"),
                               ("policy_logit", "disc_agent_logit"),
                               ("demo_logit", "disc_demo_logit"),
                               ("gradient_penalty", "disc_grad_penalty"))
            if key in train_info}


def reward_log_metrics(agent, train_info):
    metrics = {"Reward/weighted_discriminator":
               agent._disc_reward_w * mean_metric(train_info["disc_rewards"])}
    if "enc_rewards" in train_info:
        metrics["Reward/weighted_encoder"] = agent._enc_reward_w * mean_metric(train_info["enc_rewards"])
    return metrics


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


def prepare_llc(checkpoint, source_url=None, expected_sha256=None):
    """Materialize an immutable LLC dependency before importing the simulator.

    Signed URLs stay out of logs and checkpoints. Downloads are verified before
    replacing the destination, and never written into the training repository.
    """
    checkpoint = Path(checkpoint)
    def digest(path):
        hasher = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024*1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
    if source_url and not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(checkpoint.name+".download")
        try:
            with urllib.request.urlopen(source_url, timeout=120) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            if expected_sha256 and digest(temporary) != expected_sha256:
                raise RuntimeError("Downloaded LLC SHA256 mismatch")
            temporary.replace(checkpoint)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("LLC download failed; verify source availability") from None
    if not checkpoint.is_file():
        raise FileNotFoundError("LLC checkpoint is unavailable")
    actual = digest(checkpoint)
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError("LLC checkpoint SHA256 mismatch")
    print("F1_LLC_READY " + json.dumps(dict(bytes=checkpoint.stat().st_size, sha256=actual)), flush=True)
    return checkpoint


def cloud_observer_class(entry, args):
    class CloudObserver(entry.RLGPUAlgoObserver):
        def after_init(self, algo):
            super().after_init(algo)
            task = algo.vec_env.env.task
            task.training_seed = entry.cfg_train["params"]["seed"]
            metadata = task.get_training_metadata()
            print("F1_RUNTIME_EFFECTIVE " + json.dumps({
                "algorithm": args.algorithm, "seed": entry.cfg_train["params"]["seed"],
                "control_config": task.get_control_config(),
                "actor_dim": metadata["actor_dim"], "critic_dim": metadata["critic_dim"],
                "amp_dim": metadata["amp_dim"], "motions": metadata["motions"],
            }), flush=True)

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
            for name, values in infos.get("episode_metrics", {}).items():
                if values.numel():
                    total, count = self._motion_totals.get(name, (0, 0))
                    self._motion_totals[name] = (total + values.sum().detach(), count + values.numel())

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
            result["motion"] = self._last_motion_metrics

        def format_train_stats(self, train_info, epoch_num, frame, total_time):
            if not hasattr(self, "_iteration_times"):
                self._iteration_times = deque(maxlen=20)
            self._iteration_times.append(train_info["total_time"])
            metrics = {}
            metrics.update(self._last_motion_metrics)
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
            diagnostics = discriminator_log_metrics(train_info)
            diagnostics.update(reward_log_metrics(self.algo, train_info))
            metrics.update(diagnostics)
            if args.algorithm == "hrl":
                metrics["Control/LLC_hz"] = 1/self.algo.vec_env.env.task.dt
                metrics["Control/HLC_hz"] = 1/(self.algo.vec_env.env.task.dt*self.algo._llc_steps)
                metrics["Reward/weighted_task"] = self.algo._task_reward_w * mean_metric(train_info["task_rewards"])
            for label, value in diagnostics.items():
                self.writer.add_scalar(label, value, frame)
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

    return CloudObserver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="output")
    parser.add_argument("--algorithm", choices=("ase", "amp", "hrl"), default="ase")
    parser.add_argument("--llc-checkpoint", type=Path, help="Frozen ASE model required by F1 HLC")
    parser.add_argument("--llc-checkpoint-url", help="Optional download URL for a cloud LLC dependency")
    parser.add_argument("--llc-sha256", help="Expected immutable LLC SHA256")
    parser.add_argument("--motion-file", default="ase/data/motions/f1/dataset_f1.yaml")
    parser.add_argument("--json-logs", action="store_true", help="Also emit F1_TRAIN JSON lines for automation")
    args, extra = parser.parse_known_args()
    if args.max_iterations < 1 or args.num_envs < 1:
        parser.error("max_iterations and num_envs must be positive")
    if args.algorithm == "hrl" and not args.llc_checkpoint:
        parser.error("hrl requires --llc-checkpoint")
    if args.llc_checkpoint:
        args.llc_checkpoint = args.llc_checkpoint.resolve()
    if args.algorithm == "hrl":
        args.llc_checkpoint = prepare_llc(args.llc_checkpoint, args.llc_checkpoint_url, args.llc_sha256)

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

    CloudObserver = cloud_observer_class(entry, args)
    entry.RLGPUAlgoObserver = CloudObserver
    print("F1_RUNTIME " + json.dumps({
        "python": sys.version.split()[0], "torch": torch.__version__,
        "num_envs": args.num_envs, "max_iterations": args.max_iterations,
        "seed": args.seed, "control_hz": 50,
        "algorithm": args.algorithm, "motion_file": args.motion_file,
    }), flush=True)
    sys.argv = [
        str(ROOT / "ase/run.py"), "--task", "HumanoidF1HLC" if args.algorithm == "hrl" else "HumanoidF1", "--headless",
        "--sim_device", "cuda:0", "--pipeline", "gpu", "--rl_device", "cuda:0",
        "--cfg_env", "ase/data/cfg/humanoid_f1_hlc.yaml" if args.algorithm == "hrl" else "ase/data/cfg/humanoid_f1.yaml",
        "--cfg_train", "ase/data/cfg/train/rlg/" + args.algorithm + "_f1.yaml",
        "--motion_file", args.motion_file,
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iterations),
        "--seed", str(args.seed), "--output_path", args.output_path,
        *extra,
    ]
    if args.algorithm == "hrl":
        sys.argv += ["--llc_checkpoint", str(args.llc_checkpoint)]
    entry.main()


if __name__ == "__main__":
    main()
