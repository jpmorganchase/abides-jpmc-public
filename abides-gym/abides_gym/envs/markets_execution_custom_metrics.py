from collections import defaultdict
from typing import Dict

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class MyCallbacks(DefaultCallbacks):
    """
    Class that defines callbacks for the execution environment
    """

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        """Callback run on the rollout worker before each episode starts.

        Args:
                worker (RolloutWorker): Reference to the current rollout worker.
                base_env (BaseEnv): BaseEnv running the episode. The underlying
                        env object can be gotten by calling base_env.get_unwrapped().
                policies (dict): Mapping of policy id to policy objects. In single
                        agent mode there will only be a single "default" policy.
                episode (MultiAgentEpisode): Episode object which contains episode
                        state. You can use the `episode.user_data` dict to store
                        temporary data, and `episode.custom_metrics` to store custom
                        metrics for the episode.
                env_index (EnvID): Obsoleted: The ID of the environment, which the
                        episode belongs to.
                kwargs: Forward compatibility placeholder.
        """
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )

        episode.user_data = defaultdict(default_factory=list)

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        """Runs on each episode step.

        Args:
                worker (RolloutWorker): Reference to the current rollout worker.
                base_env (BaseEnv): BaseEnv running the episode. The underlying
                        env object can be gotten by calling base_env.get_unwrapped().
                policies (Optional[Dict[PolicyID, Policy]]): Mapping of policy id
                        to policy objects. In single agent mode there will only be a
                        single "default_policy".
                episode (MultiAgentEpisode): Episode object which contains episode
                        state. You can use the `episode.user_data` dict to store
                        temporary data, and `episode.custom_metrics` to store custom
                        metrics for the episode.
                env_index (EnvID): Obsoleted: The ID of the environment, which the
                        episode belongs to.
                kwargs: Forward compatibility placeholder.
        """
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        agent0_info = episode._agent_to_last_info["agent0"]

        for k, v in agent0_info.items():
            episode.user_data[k].append(v)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        """Runs when an episode is done.

        Args:
                worker (RolloutWorker): Reference to the current rollout worker.
                base_env (BaseEnv): BaseEnv running the episode. The underlying
                        env object can be gotten by calling base_env.get_unwrapped().
                policies (Dict[PolicyID, Policy]): Mapping of policy id to policy
                        objects. In single agent mode there will only be a single
                        "default_policy".
                episode (MultiAgentEpisode): Episode object which contains episode
                        state. You can use the `episode.user_data` dict to store
                        temporary data, and `episode.custom_metrics` to store custom
                        metrics for the episode.
                env_index (EnvID): Obsoleted: The ID of the environment, which the
                        episode belongs to.
                kwargs: Forward compatibility placeholder.
        """

        # this corresponds to feature we are interested  by the last value - whole episode
        for metrics in [
            "slippage_reward",
            "late_penalty_reward",
            "executed_quantity",
            "remaining_quantity",
        ]:
            episode.custom_metrics[metrics] = np.sum(episode.user_data[metrics])

        # last value
        i = None
        milestone_index = -1
        action_counter = episode.user_data["action_counter"][milestone_index]
        tot_actions = 0
        for key, val in action_counter.items():
            tot_actions += val
        for key, val in action_counter.items():
            episode.custom_metrics[f"pct_action_counter_{key}_{i}"] = val / tot_actions

        metrics = [
            "holdings_pct",
            "time_pct",
            "diff_pct",
            "imbalance_all",
            "imbalance_5",
            "price_impact",
            "spread",
            "direction_feature",
        ]

        for metric in metrics:
            episode.custom_metrics[f"{metric}_{i}"] = episode.user_data[metric][
                milestone_index
            ]

        # milestone steps
        num_max_steps_per_episode = episode.user_data["num_max_steps_per_episode"][-1]
        num_milestone = 4
        len_milestone = num_max_steps_per_episode / num_milestone
        for i in range(num_milestone + 1):
            milestone_index = int(i * len_milestone)
            if milestone_index >= len(episode.user_data["action_counter"]):
                break

            action_counter = episode.user_data["action_counter"][milestone_index]
            tot_actions = 0
            for key, val in action_counter.items():
                tot_actions += val
            for key, val in action_counter.items():
                episode.custom_metrics[f"pct_action_counter_{key}_{i}"] = (
                    val / tot_actions
                )

            for metric in metrics:
                episode.custom_metrics[f"{metric}_{i}"] = episode.user_data[metric][
                    milestone_index
                ]

        # TODO: add the episode.hist_data

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        """Called at the end of RolloutWorker.sample().

        Args:
                worker (RolloutWorker): Reference to the current rollout worker.
                samples (SampleBatch): Batch to be returned. You can mutate this
                        object to modify the samples generated.
                kwargs: Forward compatibility placeholder.
        """
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().

        Args:
                trainer (Trainer): Current trainer instance.
                result (dict): Dict of results returned from trainer.train() call.
                        You can mutate this object to add additional metrics.
                kwargs: Forward compatibility placeholder.
        """
        pass

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        """Called at the beginning of Policy.learn_on_batch().

        Note: This is called before 0-padding via
        `pad_batch_to_sequences_of_same_size`.

        Args:
                policy (Policy): Reference to the current Policy object.
                train_batch (SampleBatch): SampleBatch to be trained on. You can
                        mutate this object to modify the samples generated.
                result (dict): A results dict to add custom metrics to.
                kwargs: Forward compatibility placeholder.
        """
        pass

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: MultiAgentEpisode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch],
        **kwargs,
    ):
        """Called immediately after a policy's postprocess_fn is called.

        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Args:
                worker (RolloutWorker): Reference to the current rollout worker.
                episode (MultiAgentEpisode): Episode object.
                agent_id (str): Id of the current agent.
                policy_id (str): Id of the current policy for the agent.
                policies (dict): Mapping of policy id to policy objects. In single
                        agent mode there will only be a single "default_policy".
                postprocessed_batch (SampleBatch): The postprocessed sample batch
                        for this agent. You can mutate this object to apply your own
                        trajectory postprocessing.
                original_batches (dict): Mapping of agents to their unpostprocessed
                        trajectory data. You should not mutate this object.
                kwargs: Forward compatibility placeholder.
        """
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0

        episode.custom_metrics["num_batches"] += 1
