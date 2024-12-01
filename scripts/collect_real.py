import argparse
import datetime
import os
import time

import numpy as np

from research.datasets import ReplayBuffer
from research.utils.config import Config
from research.utils.evaluate import EvalMetricTracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--num-ep", type=int, default=np.inf)
    parser.add_argument("--num-steps", type=int, default=np.inf)
    parser.add_argument(
        "--shard", action="store_true", default=False, help="Whether or not to shard the dataset into episodes."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--override",
        metavar="KEY=VALUE",
        nargs="+",
        default=[],
        help="Set kv pairs used as args for the entry point script.",
    )
    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True)

    config = Config.load(os.path.dirname(args.checkpoint) if args.checkpoint.endswith(".pt") else args.checkpoint)
    config["checkpoint"] = None  # Set checkpoint to None

    # Overrides
    print("Overrides:")
    for override in args.override:
        print(override)

    for override in args.override:
        items = override.split("=")
        key, value = items[0].strip(), "=".join(items[1:])
        config_path = key.split(".")
        config_dict = config
        while len(config_path) > 1:
            config_dict = config_dict[config_path[0]]
            config_path.pop(0)
        config_dict[config_path[0]] = value

    config = config.parse()

    # Get the environment
    env = config.get_train_env_fn()()
    if env is None:
        env = config.get_eval_env_fn()()

    model = config.get_model(observation_space=env.observation_space, action_space=env.action_space, device=args.device)

    capacity = (env._max_episode_steps + 2) * args.num_ep if args.num_ep < np.inf else args.num_steps

    capacity = 2 if args.shard else capacity  # Set capacity to a small value if we are saving eps to disk directly.
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, capacity=capacity, cleanup=not args.shard, distributed=args.shard
    )

    # Track data collection
    num_steps = 0
    num_ep = 0
    finished_data_collection = False
    # Episode metrics
    metric_tracker = EvalMetricTracker()
    start_time = time.time()

    ep_steps = 0

    try:
        while not finished_data_collection:
            progress = num_ep / args.num_ep if args.num_ep != np.inf else num_steps / args.num_steps
            ep_steps = 0
            # Collect an episode 
            done = False
            ep_length = 0
            first_obs = {}
            obs = env.reset()
            first_obs = obs
            metric_tracker.reset()
            print("====Collecting new episode====")

            trajectory_dict = {"obs": {}}
            action = np.zeros(7)

            while not done:
                action = model.predict(dict(obs=obs))
                obs, reward, done, info = env.step(action)

                x = model.get_info()
                if x["success"] or x["failure"]:
                    done = True
                    trajectory_dict["done"][-1] = True
                    break

                if ep_length == 0:
                    for k, v in obs.items():
                        trajectory_dict["obs"][k] = np.expand_dims(v, 0)
                    trajectory_dict["reward"] = np.reshape(reward, (1))
                    trajectory_dict["done"] = np.reshape(done, (1))
                    trajectory_dict["discount"] = np.reshape(info["discount"], (1))
                    trajectory_dict["action"] = np.expand_dims(action, 0)
                else:
                    for k in obs.keys():
                        trajectory_dict["obs"][k] = np.concatenate(
                            (trajectory_dict["obs"][k], np.expand_dims(obs[k], 0))
                        )
                    trajectory_dict["reward"] = np.concatenate((trajectory_dict["reward"], np.reshape(reward, (1))))
                    trajectory_dict["done"] = np.concatenate((trajectory_dict["done"], np.reshape(done, (1))))
                    trajectory_dict["discount"] = np.concatenate(
                        (trajectory_dict["discount"], np.reshape(info["discount"], (1)))
                    )
                    trajectory_dict["action"] = np.concatenate((trajectory_dict["action"], np.expand_dims(action, 0)))

                metric_tracker.step(reward, info)
                ep_length += 1
                if hasattr(env, "_max_episode_steps") and ep_length == env._max_episode_steps:
                    done = True
                num_steps += 1
            # If successful, add to the replay buffer
            if model.get_info()["success"]: 
                replay_buffer.add(obs=first_obs)
                replay_buffer.extend(**trajectory_dict)
            trajectory_dict = {}
            num_ep += 1
            # Determine if we should stop data collection
            finished_data_collection = num_steps >= args.num_steps or num_ep >= args.num_ep
    except KeyboardInterrupt:
        print("Demo ended")

    end_time = time.time()
    print("Finished", num_ep, "episodes in", num_steps, "steps.")
    print("It took", (end_time - start_time) / num_steps, "seconds per step")

    replay_buffer.save(args.path)
    fname = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    # Write the metrics
    metrics = metric_tracker.export()
    print("Metrics:")
    print(metrics)
    with open(os.path.join(args.path, "metrics.txt"), "a") as f:
        f.write("Collected data: " + str(fname) + "\n")
        for k, v in metrics.items():
            f.write(k + ": " + str(v) + "\n")
