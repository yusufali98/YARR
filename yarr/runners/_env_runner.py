import copy
import logging
import os
import time
import pandas as pd

from multiprocessing import Process, Manager
from multiprocessing import get_start_method, set_start_method
from typing import Any

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.log_writer import LogWriter
from yarr.utils.process_str import change_case

try:
    if get_start_method() != 'spawn':
        set_start_method('spawn', force=True)
except RuntimeError:
    pass


class _EnvRunner(object):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 train_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_seed: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 num_eval_episodes_signal: Any,
                 eval_epochs_signal: Any,
                 eval_report_signal: Any,
                 log_freq: int,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 weightsdir: str = None,
                 logdir: str = None,
                 env_device: torch.device = None,
                 previous_loaded_weight_folder: str = '',
                 num_eval_runs: int = 1,
                 ):
        self._train_env = train_env
        self._eval_env = eval_env
        self._agent = agent
        self._train_envs = train_envs
        self._eval_envs = eval_envs
        self._train_episodes = train_episodes
        self._eval_episodes = eval_episodes
        self._training_iterations = training_iterations
        self._num_eval_runs = num_eval_runs
        self._eval_from_seed = eval_from_seed
        self._episode_length = episode_length
        self._rollout_generator = rollout_generator
        self._weightsdir = weightsdir
        self._logdir = logdir
        self._env_device = env_device
        self._previous_loaded_weight_folder = previous_loaded_weight_folder

        self._timesteps = timesteps

        self._p_args = {}
        self.p_failures = {}
        manager = Manager()
        self.write_lock = manager.Lock()
        self.stored_transitions = manager.list()
        self.agent_summaries = manager.list()
        self._kill_signal = kill_signal
        self._step_signal = step_signal
        self._num_eval_episodes_signal = num_eval_episodes_signal
        self._eval_epochs_signal = eval_epochs_signal
        self._eval_report_signal = eval_report_signal
        self._save_load_lock = save_load_lock
        self._current_replay_ratio = current_replay_ratio
        self._target_replay_ratio = target_replay_ratio
        self._log_freq = log_freq

        self._new_weights = False

    def restart_process(self, name: str):
        run_fn = self._run_eval if eval else self._run_train
        p = Process(target=run_fn, args=self._p_args[name], name=name)
        p.start()
        return p

    def spin_up_envs(self, name: str, num_envs: int, eval: bool):

        ps = []
        for i in range(num_envs):
            n = name + str(i)
            self._p_args[n] = (n, eval)
            self.p_failures[n] = 0
            run_fn = self._run_eval if eval else self._run_train
            p = Process(target=run_fn, args=self._p_args[n], name=n)
            p.start()
            ps.append(p)
        return ps

    def _load_save(self):
        if self._weightsdir is None:
            print("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # if self._previous_loaded_weight_folder == weight_folders[-1]:
                    #     time.sleep(1)
                    #     continue
                    # Only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # Rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        print('Agent %s: Loaded weights: %s' % (self._name, d))
                        # print('Agent %s: Loaded weights: %s' % (self._name, d))
                        self._new_weights = True
                    else:
                        self._new_weights = False
                    break
            print('Waiting for weights to become available.')
            time.sleep(1)

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def _run_train(self, name: str, eval: bool):

        self._name = name

        self._agent = copy.deepcopy(self._agent)

        self._agent.build(training=False, device=self._env_device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._train_env
        env.eval = eval
        env.launch()
        for ep in range(self._train_episodes):
            self._load_save()
            logging.info('%s: Starting episode %d.' % (name, ep))
            # print("Train %s: Starting episode %d." % (name, ep))
            episode_rollout = []
            generator = self._rollout_generator.generator(
                self._step_signal, env, self._agent,
                self._episode_length, self._timesteps, eval)
            try:
                for replay_transition in generator:
                    while True:
                        if self._kill_signal.value:
                            env.shutdown()
                            return
                        if (eval or self._target_replay_ratio is None or
                                self._step_signal.value <= 0 or (
                                        self._current_replay_ratio.value >
                                        self._target_replay_ratio)):
                            break
                        time.sleep(1)
                        logging.debug(
                            'Agent. Waiting for replay_ratio %f to be more than %f' %
                            (self._current_replay_ratio.value, self._target_replay_ratio))

                    with self.write_lock:
                        if len(self.agent_summaries) == 0:
                            # Only store new summaries if the previous ones
                            # have been popped by the main env runner.
                            for s in self._agent.act_summaries():
                                self.agent_summaries.append(s)
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                env.shutdown()
                raise e

            with self.write_lock:
                for transition in episode_rollout:
                    self.stored_transitions.append((name, transition, eval))
        env.shutdown()

    def _run_eval(self, name: str, eval: bool):

        self._name = name

        self._agent = copy.deepcopy(self._agent)

        self._agent.build(training=False, device=self._env_device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env
        env.eval = eval
        env.launch()

        while self._unevaluated_weights():
            self._load_save()
            for n_eval in range(self._num_eval_runs):
                for ep in range(self._eval_episodes):
                    eval_demo_seed = ep + self._eval_from_seed
                    logging.info('%s: Starting episode %d, seed %d.' % (name, ep, eval_demo_seed))
                    # print("Eval: %s: Starting episode %d, seed %d." % (name, ep, eval_demo_seed))
                    episode_rollout = []
                    generator = self._rollout_generator.generator(
                        self._step_signal, env, self._agent,
                        self._episode_length, self._timesteps, eval,
                        eval_demo_seed=eval_demo_seed)
                    try:
                        for replay_transition in generator:
                            while True:
                                if self._kill_signal.value:
                                    env.shutdown()
                                    return
                                if (eval or self._target_replay_ratio is None or
                                        self._step_signal.value <= 0 or (
                                                self._current_replay_ratio.value >
                                                self._target_replay_ratio)):
                                    break
                                time.sleep(1)
                                logging.debug(
                                    'Agent. Waiting for replay_ratio %f to be more than %f' %
                                    (self._current_replay_ratio.value, self._target_replay_ratio))

                            with self.write_lock:
                                if len(self.agent_summaries) == 0:
                                    # Only store new summaries if the previous ones
                                    # have been popped by the main env runner.
                                    for s in self._agent.act_summaries():
                                        self.agent_summaries.append(s)
                            episode_rollout.append(replay_transition)
                    except StopIteration as e:
                        continue
                    except Exception as e:
                        env.shutdown()
                        raise e

                    with self.write_lock:
                        for transition in episode_rollout:
                            self.stored_transitions.append((name, transition, eval))

                    self._num_eval_episodes_signal.value += 1
                self._eval_report_signal.value = True

            if self._new_weights:
                self._eval_epochs_signal.value += 1

        env.shutdown()

    def _run_eval_independent(self, name: str,
                              stats_accumulator,
                              weight,
                              writer_lock,
                              eval=True,
                              resumed_from_prev_run=False):

        self._name = name

        self._agent = copy.deepcopy(self._agent)

        self._agent.build(training=False, device=self._env_device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env
        env.eval = eval
        env.launch()

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')


        writer = LogWriter(self._logdir, True, True,
                           env_csv='eval_data.csv')
        writer.set_resumed_from_prev_run(resumed_from_prev_run)
        #
        # weight_folders = os.listdir(self._weightsdir)
        # weight_folders = sorted(map(int, weight_folders))
        #
        # # check if previous evaluations exist
        # env_data_csv_file = os.path.join(self._logdir, 'eval_data.csv')
        # if os.path.exists(env_data_csv_file):
        #     env_dict = pd.read_csv(env_data_csv_file).to_dict()
        #     evaluated_weights = sorted(map(int, list(env_dict['step'].values())))
        #     weight_folders = [w for w in weight_folders if w not in evaluated_weights]
        #     writer.set_resumed_from_prev_run(True)

        logging.info('Evaluating weight %s' % weight)
        weight_path = os.path.join(self._weightsdir, str(weight))
        self._agent.load_weights(weight_path)

        new_transitions = {'train_envs': 0, 'eval_envs': 0}
        total_transitions = {'train_envs': 0, 'eval_envs': 0}
        current_task_id = -1

        for n_eval in range(self._num_eval_runs):
            for ep in range(self._eval_episodes):
                eval_demo_seed = ep + self._eval_from_seed
                logging.info('%s: Starting episode %d, seed %d.' % (name, ep, eval_demo_seed))
                # print("Eval: %s: Starting episode %d, seed %d." % (name, ep, eval_demo_seed))
                episode_rollout = []
                generator = self._rollout_generator.generator(
                    self._step_signal, env, self._agent,
                    self._episode_length, self._timesteps, eval,
                    eval_demo_seed=eval_demo_seed)
                try:
                    for replay_transition in generator:
                        while True:
                            if self._kill_signal.value:
                                env.shutdown()
                                return
                            if (eval or self._target_replay_ratio is None or
                                    self._step_signal.value <= 0 or (
                                            self._current_replay_ratio.value >
                                            self._target_replay_ratio)):
                                break
                            time.sleep(1)
                            logging.debug(
                                'Agent. Waiting for replay_ratio %f to be more than %f' %
                                (self._current_replay_ratio.value, self._target_replay_ratio))

                        with self.write_lock:
                            if len(self.agent_summaries) == 0:
                                # Only store new summaries if the previous ones
                                # have been popped by the main env runner.
                                for s in self._agent.act_summaries():
                                    self.agent_summaries.append(s)
                        episode_rollout.append(replay_transition)
                except StopIteration as e:
                    continue
                except Exception as e:
                    env.shutdown()
                    raise e

                with self.write_lock:
                    for transition in episode_rollout:
                        self.stored_transitions.append((name, transition, eval))

                        new_transitions['eval_envs'] += 1
                        total_transitions['eval_envs'] += 1
                        stats_accumulator.step(transition, eval)
                        current_task_id = transition.info['active_task_id']

                self._num_eval_episodes_signal.value += 1

            # report summaries
            summaries = []
            summaries.extend(stats_accumulator.pop())
            for key, value in new_transitions.items():
                summaries.append(ScalarSummary('%s/new_transitions' % key, value))
            for key, value in total_transitions.items():
                summaries.append(ScalarSummary('%s/total_transitions' % key, value))
            summaries.extend(self.agent_summaries)

            if hasattr(self._eval_env, '_task_class'):
                eval_task_name = change_case(self._eval_env._task_class.__name__)
                multi_task = False
            elif hasattr(self._eval_env, '_task_classes'):
                if current_task_id != -1:
                    task_id = (current_task_id) % len(self._eval_env._task_classes)
                    eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
                else:
                    eval_task_name = ''
                multi_task = True
            else:
                raise Exception('Neither task_class nor task_classes found in eval env')

            if eval_task_name and multi_task:
                for s in summaries:
                    if 'eval' in s.name:
                        s.name = '%s/%s' % (s.name, eval_task_name)
            print("Finished %s." % eval_task_name)

            with writer_lock:
                writer.add_summaries(weight, summaries)

            self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
            self.agent_summaries[:] = []
            self.stored_transitions[:] = []

        with writer_lock:
            writer.end_iteration()

        logging.info('Finished evaluation.')
        env.shutdown()

    def _unevaluated_weights(self):
        weight_folders = []

        while not len(weight_folders) > 1:
            weight_folders = os.listdir(self._weightsdir)
            weight_folders = sorted(map(int, weight_folders))

            logging.info('Waiting for first checkpoint.')
            time.sleep(10)

        if self._previous_loaded_weight_folder == weight_folders[-1] and int(weight_folders[-1]) == self._training_iterations:
            return False

        return True

    def kill(self):
        self._kill_signal.value = True
