"""Cloud entry for F1 ASE; invoked by gm-run from any working directory.

Example: gm-run F1_ase/ase/train_f1_cloud.py --install-deps --max_iterations 200
Keeps the image's PyTorch/Isaac Gym and caches legacy ASE dependencies outside
the platform's /workspace artifact scanner.
"""
import argparse
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
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="output")
    args, extra = parser.parse_known_args()
    if args.max_iterations < 1 or args.num_envs < 1:
        parser.error("max_iterations and num_envs must be positive")

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
        def after_print_stats(self, frame, epoch_num, total_time):
            super().after_print_stats(frame, epoch_num, total_time)
            result = {"epoch": epoch_num, "max_epochs": args.max_iterations,
                      "elapsed_s": round(total_time, 2)}
            if self.algo.game_rewards.current_size > 0:
                reward = float(self.algo._get_mean_rewards().mean())
                length = float(self.algo.game_lengths.get_mean())
                result.update(mean_reward=reward, mean_episode_length=length,
                              survival_s=round(length * 0.02, 3))
                self.writer.add_scalar("Train/mean_reward", reward, epoch_num)
                self.writer.add_scalar("Train/mean_episode_length", length, epoch_num)
            print("F1_TRAIN " + json.dumps(result), flush=True)

    entry.RLGPUAlgoObserver = CloudObserver
    print("F1_RUNTIME " + json.dumps({
        "python": sys.version.split()[0], "torch": torch.__version__,
        "num_envs": args.num_envs, "max_iterations": args.max_iterations,
        "seed": args.seed, "control_hz": 50,
    }), flush=True)
    sys.argv = [
        str(ROOT / "ase/run.py"), "--task", "HumanoidF1", "--headless",
        "--sim_device", "cuda:0", "--pipeline", "gpu", "--rl_device", "cuda:0",
        "--cfg_env", "ase/data/cfg/humanoid_f1.yaml",
        "--cfg_train", "ase/data/cfg/train/rlg/ase_f1.yaml",
        "--motion_file", "ase/data/motions/f1/dataset_f1.yaml",
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iterations),
        "--seed", str(args.seed), "--output_path", args.output_path,
        *extra,
    ]
    entry.main()


if __name__ == "__main__":
    main()
