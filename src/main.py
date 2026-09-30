import os
import json
import argparse
import sys
from datetime import datetime as dt

import numpy as np
import yaml

from TRL.common.data_manager import DataManager
from TRL.common.utils import init_env, load, env_reset, env_step
from TRL.common.logger import Logger
from TRL.agent import Agent
from TRL.common.checkpoint import save_checkpoint, load_checkpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_test_episode(env, agent: Agent, time_limit: int, suite):
    print('\n', '=' * 10, 'TEST EPISODE', '=' * 10, '\n')
    episode_step = 1
    episode_reward = 0
    current_observation = env_reset(env, suite)
    done = False
    while not done:
        action, trajectory, time_transition  = agent.select_action(current_observation, episode_step)
        observation, reward, done = env_step(env, suite, action)
        def _fmt(obs):
            # one-hot obs (DeepSea) -> cell index; anything else -> raw vector
            lst = obs.tolist()
            return lst.index(1) if 1 in lst else [round(x, 3) for x in lst]
        print(
            '\nstate:', _fmt(current_observation),
            '\taction:', action.item(),
            '\treward:', round(reward, 4),
            '\tnext_state:', _fmt(observation)
        )
        episode_reward += reward
        current_observation = observation
        episode_step += 1
        done = done or episode_step == time_limit

    return episode_reward


def run_experiment(env, agent: Agent, logger: Logger, test_env,
        config, save: bool, save_freq: int, iter:int, start_step: int = 1, start_ep: int = 0):
    start_time = dt.now()
    ep = start_ep
    experiment_step = start_step # controls when the model gets updated (training)

    while experiment_step <= config['steps']: # controls the number of interactions/episodes with the environment
        episode_step = 1 # controls the number of steps agent takes in the env before episode concludes
        episode_reward = 0

        current_observation = env_reset(env, config['suite'])

        done = False
        while not done:
            # print('\n\nEpisode:',ep,'\tEpisode step',episode_step,'\tTime limit:',time_limit)
            # Select action according to BTRL and step into the environment until episode ends
            if config['test'] and experiment_step % config['test_freq'] == 0:
                # print('\nTesting')
                test_reward = run_test_episode(test_env, agent, config['time_limit'], config['suite'])
                logger.log_episode(
                    experiment_step, train_reward=np.nan, test_reward=test_reward
                )
                print(f'Episode {ep}, Experiment Timestep {experiment_step}, Test Reward {test_reward}')
                print('\n', '=' * 10, 'TEST COMPLETE', '=' * 10, '\n')
                print('Iteration:',iter, '\tStarted @',start_time)

            # print('Current state:\n',current_observation)
            # print('episode_step:',episode_step)
            action, trajectory, time_transition = agent.select_action(current_observation, episode_step)
            # print('\tmain() action selected:',action.item())

            next_state, reward, done = env_step(env, config['suite'], action)
            # print('Done?',done,'\tExperiment Timestep:',experiment_step)
            done = done or episode_step == config['time_limit']
            agent.update(
                trajectory,
                action,
                reward,
                next_state,
                done,
                ep,
                experiment_step,
                time_transition
            )

            episode_reward += reward
            current_observation = next_state
            episode_step += 1
            experiment_step += 1

            if ep and save and experiment_step % save_freq == 0:
                logger.data_manager.save(agent, experiment_step)
                save_checkpoint(agent, experiment_step, ep,
                    logger.data_manager.logdir + "checkpoints/" + str(experiment_step) + "/resume.pt")
        # print(f'Episode: {ep}, Timestep: {experiment_step}, Train Reward: {episode_reward}')

        ep += 1
        logger.log_episode(
            experiment_step, train_reward=episode_reward, test_reward=np.nan
        )

    logger.data_manager.save(agent, experiment_step)
    print('\nFinal performance:')
    test_reward = run_test_episode(test_env, agent, config['time_limit'], config['suite'])
    print(f'Episode {ep}, Experiment Timestep {experiment_step}, Test Reward {test_reward}')


def main(config: dict, iter:int = 10):
    data_manager = DataManager(config)
    logger = Logger(data_manager)

    env, actions, test_env, obs_space = init_env(config["experiment"]["suite"], config["experiment"]["env"],
                                                 config["experiment"]["test"],
        deterministic=config["experiment"]["deterministic"]
    )

    agent = Agent(config, actions, logger, obs_space, config["experiment"]["seed"])

    start_step, start_ep = 1, 0
    if config.get("resume_from"):
        start_step, start_ep = load_checkpoint(agent, config["resume_from"])
        print(f"Resumed from {config['resume_from']} at step {start_step}, episode {start_ep}")

    run_experiment(
        env, agent,
        logger,
        test_env,
        config["experiment"],
        config["save"], config["save_freq"],
        iter,
        start_step, start_ep,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/config_vector-setting1.yaml")
    parser.add_argument("--suite", type=str, default="bsuite", help="Bsuite is implemented.")
    parser.add_argument("--env", type=str, default='',
                        help="In the case of bsuite, enter a single integer for a DeepSea environment.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_experiments", type=int, default=10)
    # parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--load", type=bool, default=False)  # load model from last checkpoint or fresh run
    parser.add_argument("--load_timestep", type=str, default=None)  # load model from last checkpoint or fresh run
    parser.add_argument("--agent", type=str, default=None)  # load model from last checkpoint or fresh run
    parser.add_argument("--hidden_dims", type=int, default=None)  # load model from last checkpoint or fresh run

    args = parser.parse_args()
    # print('args:', args)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

        if args.env:
            config["experiment"]["env"] = args.env
        config["experiment"]["seed"] = args.seed
        config["load"] = args.load

        config["replay"]["sequence_length"] = int(config["experiment"]["env"])+1
        # config['transition']['hidden_dim'] = 512
        if args.hidden_dims:
            config['transition']['hidden_dim'] = config['value']['hidden_dim'] = config['terminal']['hidden_dim'] = args.hidden_dims
            config["experiment"]["name"] = 'hidden_dims' + str(args.hidden_dims)
        else:
            config["experiment"]["name"] = (
                'vn_dims' + str(config['value']['hidden_dim']) + '-x-' +
                str(config['value'].get('hidden_layers', 2)) +
                '-tn_dims' + str(config['transition']['hidden_dim']) +
                'x' + str(config['transition']['num_encoder_layers']) +
                '-context_length' + str(config['transition']['context_length'])
            )

        config["experiment"]["time_limit"] = int(config["experiment"]["env"])+1

        if config["load"]:
            config["experiment"]["seed"] = args.seed
            ckpt_dir = './logdir/%s/BTRL-%s/%s/checkpoints/' % (
                config["experiment"]["env"], config["experiment"]["name"], args.seed)
            config["experiment"]["log_dir"] = ckpt_dir
            print(ckpt_dir)
            if args.load_timestep:
                config["load_dir"] = ckpt_dir + args.load_timestep
            else:
                ckpts = [int(i) for i in os.listdir(ckpt_dir)[1:]]
                config["load_dir"] = ckpt_dir + str(max(ckpts))

    print('\n',json.dumps(config,indent=2),'\n')

    for i in range(args.num_experiments):
        begin = dt.now()
        print('\n\n', '*' * 10,'Starting experiment',i,'/',args.num_experiments, '@',begin,'*' * 10,'\n')
        main(config, i)
        print('Time taken:',(dt.now()-begin).total_seconds()//60,'minutes\n')
