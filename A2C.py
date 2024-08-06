from datetime import datetime
import gymnasium as gym
from stable_baselines3 import A2C
import os
from utils import make_env

game = "ALE/Assault-v5"
seed = 66
algorithm_name = "A2C"

now = datetime.today().strftime("%Y-%m-%d/%H%M%S")
run_name = f"{algorithm_name}/{game.replace('/', '-')}__production__{seed}__{now}"

# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3():

    # Where to store trained model and logs
    model_dir = "models"
    log_dir = f"runs/{algorithm_name}/{run_name}"
    num_envs = 1

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    envs = gym.vector.SyncVectorEnv(
        [make_env(game, seed + i, i, True, run_name) for i in range(num_envs)]
    )

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = A2C(
        "MlpPolicy",
        envs,
        verbose=0,
        device="cuda",
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        gamma=0.99,
    )

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 2_000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)  # train
        if iters % 10 == 0:
            model.save(f"{model_dir}/{algorithm_name}/episode#{TIMESTEPS*iters}")


# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3():

    env = gym.vector.SyncVectorEnv([make_env(game, 0, 0, False, run_name, render_mode="human")])

    # Load model
    path = "models\A2C\old\episode#11060000"
    # model = A2C.load(f"models/{algorithm_name}/old/episode#11060000", env=env)
    model = A2C.load(path, env=env)
    eval_episode = 10

    obs, _ = env.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episode:
        action, _ = model.predict(
            observation=obs, deterministic=True
        )
        
        next_obs, _, _, _, infos = env.step(action)
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
    # Train/test using StableBaseline3
    # train_sb3()
    test_sb3()
