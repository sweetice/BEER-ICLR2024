import numpy as np
import torch
import gym
import argparse

import utils
from BEER import BEER
import time
import random
import dm_control
import dmc2gym
from tqdm import trange

torch.set_default_device("cuda")

def eval_policy(current_t, policy, domain_name, task_name, seed=0, eval_episodes=10):
    evaluations = []
    eval_env = make_env(domain_name, task_name, seed=seed)
    for _ in range(eval_episodes):
        avg_reward = 0.
        state, done = eval_env.reset(), False
        reward_seq = []
        while not done:
            action = policy.select_action(np.array(state, dtype=np.float32))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            reward_seq.append(reward)
        evaluations.append(avg_reward)
    eval_env.close()

    evaluations = np.array(evaluations)
    avg_evaluation = np.mean(evaluations)
    print("---------------------------------------")
    print(f"Current Timesteps: {current_t}, Algo: BEER, Domain: {domain_name}, Task: {task_name}, seed: {args.seed}, Evaluation over {eval_episodes} episodes: {avg_evaluation:.3f}")




def make_env(domain_name, task_name, seed):
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed, from_pixels=False,  visualize_reward=False)
    return env


if __name__ == "__main__":

    begin_time = time.asctime(time.localtime(time.time()))
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='Pendulum-v0')   # OpenAI gym env runniironment name
    parser.add_argument("--domain_name", default="humanoid") # humanoid-stand
    parser.add_argument("--task_name", default="stand") #
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    args = parser.parse_args()

    env = make_env(domain_name=args.domain_name, task_name=args.task_name, seed=args.seed)

    # Set seeds
    random.seed(args.seed)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    policy = BEER(**kwargs)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    start_time_steps = 0

    for t in trange(start_time_steps, int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state, dtype='float32'))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action).astype(np.float32)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:

            eval_policy(current_t=t+1, policy=policy, domain_name=args.domain_name,
                                                    task_name=args.task_name, seed=args.seed)