import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episode: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, render_mode="human")])
    model = Model(envs).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episode:
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                )
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    # from huggingface_hub import hf_hub_download

    from DQN_train import QNetwork
    from utils import make_env

    game = "ALE/Assault-v5"

    # model_path = hf_hub_download(repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="dqn.cleanrl_model")
    # model_path = ".pth"

    model_path = (
        "models/DQN/ALE-Assault-v5__production__66/2024-08-03/225549/episode#7000.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate(
        model_path,
        make_env,
        game,
        eval_episode=10,
        run_name=f"eval",
        Model=QNetwork,
        device=device,
        capture_video=False,
    )