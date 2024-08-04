from datetime import datetime
import gymnasium as gym
from stable_baselines3 import A2C
import os
from utils import make_env

game = "ALE/Assault-v5"
seed = 66
algorithm_name = "A2C"

# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3():
    now = datetime.today().strftime("%Y-%m-%d/%H%M%S")
    run_name = f"{algorithm_name}/{game.replace('/', '-')}__production__{seed}__{now}"
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
def test_sb3(render=True):

    env = gym.make(game, render_mode="human" if render else None)

    # Load model
    model = A2C.load(f"models/{algorithm_name}_40000", env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        action, _ = model.predict(
            observation=obs, deterministic=True
        )  # Turn on deterministic, so predict always returns the same behavior
        obs, _, terminated, _, _ = env.step(action)

        if terminated:
            break


if __name__ == "__main__":

    # Train/test using Q-Learning
    # run_q(1000, is_training=True, render=False)
    # run_q(1, is_training=False, render=True)

    # Train/test using StableBaseline3
    train_sb3()
    # test_sb3()
