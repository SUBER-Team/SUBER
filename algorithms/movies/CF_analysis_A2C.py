import argparse
import os
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.distributions import CategoricalDistribution

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import A2C
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import TensorDict
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Callable, Tuple, Union
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import gymnasium as gym

# Our
from algorithms.wrappers import StableBaselineWrapperNum
from environment.movies.configs import (
    get_enviroment_from_args,
    get_base_parser,
)
from environment import load_LLM


# Define arguments
def parse_args():
    parser = get_base_parser()
    parser.add_argument("--model-device", type=str, default="cuda:1")
    parser.add_argument("--gamma", type=float, default=0.975)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--path", type=str)
    parser.add_argument("--logs-path", type=str, default="./tmp/render/render.csv")
    args = parser.parse_args()
    return args


# Define model
class Net(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        num_users: int,
        num_items: int,
    ):
        super().__init__()
        embedding_dim = args.embedding_dim
        self.latent_dim_pi = embedding_dim * 2
        self.latent_dim_vf = embedding_dim * 2

        ## Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(
                self.user_embedding.embedding_dim,
                num_items,
            )
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(
                self.user_embedding.embedding_dim + num_items, self.latent_dim_vf * 2
            ),
            nn.ReLU(),
            nn.Linear(self.latent_dim_vf * 2, self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        user_id = features["user_id"].squeeze(1)
        film_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, film_seen], dim=1)
        user_bias = self.user_bias(user_id)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        logits[mask] = -torch.inf
        return logits, self.value_net(user_embedding_value)

    def forward_actor(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        user_embedding = self.user_embedding(user_id)
        user_bias = self.user_bias(user_id)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        logits[mask] = -torch.inf
        return logits

    def forward_critic(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        film_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, film_seen], dim=1)
        return self.value_net(user_embedding_value)


class DistributionUseLogitsDirectly(CategoricalDistribution):
    def __init__(self, action_dim: int):
        super().__init__(action_dim)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity(latent_dim)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        self.action_dist = DistributionUseLogitsDirectly(action_space.n)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Net(
            self.observation_space,
            train_env.num_users,
            train_env.num_items,
        )


class ExtractPass(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations["user_id"] = observations["user_id"].int()
        return observations


if __name__ == "__main__":
    args = parse_args()
    llm = load_LLM(args.llm_model)

    train_env = get_enviroment_from_args(llm, args)

    os.makedirs(os.path.dirname(args.logs_path), exist_ok=True)
    test_env = get_enviroment_from_args(
        llm,
        args,
        seed=args.seed + 600,
        render_mode="csv",
        render_path=args.logs_path,
        eval_mode=True,
    )

    # Create the custom actor-critic policy
    policy_kwargs = dict(
        features_extractor_class=ExtractPass,
    )

    train_env = StableBaselineWrapperNum(train_env)
    test_env = Monitor(StableBaselineWrapperNum(test_env))

    eval_callback = EvalCallback(
        test_env,
        eval_freq=10000,
        n_eval_episodes=100,
        deterministic=True,
        render="human",
    )

    model = A2C(
        CustomActorCriticPolicy,
        train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=args.model_device,
        gamma=args.gamma,
    )

    model.policy.load_state_dict(torch.load(args.path, map_location=args.model_device))

    evaluate_policy(model, test_env, render=True, n_eval_episodes=600)
