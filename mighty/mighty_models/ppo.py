import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mighty.mighty_models.networks import ACTIVATIONS, make_feature_extractor


class PPOModel(nn.Module):
    """PPO Model with policy and value networks."""

    def __init__(
        self,
        obs_shape: int,
        action_size: int,
        continuous_action: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        tanh_squash: bool = False,  # NEW: Toggle between tanh squashing and standard PPO
        policy_dist: str = "gaussian",  # "gaussian" or "beta"
        **kwargs,
    ):
        """Initialize the PPO model."""
        super().__init__()

        assert policy_dist in ("gaussian", "beta"), (
            f"policy_dist must be 'gaussian' or 'beta', got '{policy_dist}'"
        )
        assert not (policy_dist == "beta" and tanh_squash), (
            "policy_dist='beta' and tanh_squash=True are mutually exclusive: "
            "Beta distributions already have bounded [0, 1] support, so tanh "
            "squashing is neither needed nor supported in beta mode."
        )

        self.obs_size = int(obs_shape)
        self.action_size = int(action_size)
        self.continuous_action = continuous_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash
        self.policy_dist = policy_dist

        # output_style is an instance attribute (not class-level) so it can vary
        # per constructor arg; used by exploration policies / PPOUpdate as the
        # dispatch discriminator (checked before tuple-length, since the beta
        # and standard-PPO branches both return 3-tuples).
        if self.continuous_action and self.tanh_squash:
            self.output_style = "squashed_gaussian"
        elif self.continuous_action and self.policy_dist == "beta":
            self.output_style = "beta"
        elif self.continuous_action:
            self.output_style = "standard_ppo"
        else:
            self.output_style = "discrete"

        # Extract configuration from kwargs or use defaults
        head_kwargs = kwargs.get(
            "head_kwargs",
            {"hidden_sizes": [64], "layer_norm": True, "activation": "tanh"},
        )
        feature_extractor_kwargs = kwargs.get(
            "feature_extractor_kwargs",
            {
                "obs_shape": self.obs_size,
                "activation": "tanh",
                "hidden_sizes": [64, 64],
                "n_layers": 2,
            },
        )

        # Allow direct specification of hidden_sizes and activation at top level
        if "hidden_sizes" in kwargs:
            feature_extractor_kwargs["hidden_sizes"] = kwargs["hidden_sizes"]
        if "activation" in kwargs:
            feature_extractor_kwargs["activation"] = kwargs["activation"]
            head_kwargs["activation"] = kwargs["activation"]

        # Make feature extractors
        self.feature_extractor_policy, feat_dim = make_feature_extractor(
            **feature_extractor_kwargs
        )
        self.feature_extractor_value, _ = make_feature_extractor(
            **feature_extractor_kwargs
        )

        if self.continuous_action:
            if self.tanh_squash:
                # Tanh squashing mode: output mean + log_std from network
                final_out_dim = action_size * 2
                # No learnable parameter needed
                self.log_std = None
            elif self.policy_dist == "beta":
                # Beta mode: output alpha + beta from network, no learnable
                # log_std parameter (Beta has no separate scale parameter).
                final_out_dim = action_size * 2
                self.log_std = None
            else:
                # Standard PPO mode: output only mean, use learnable log_std parameter
                final_out_dim = action_size
                self.log_std = nn.Parameter(torch.zeros(action_size))
        else:
            # For discrete actions, output logits of size = action_size
            final_out_dim = action_size

        # (Architecture based on
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py)

        # Policy network
        self.policy_head = make_ppo_head(feat_dim, final_out_dim, **head_kwargs)

        # Value network
        self.value_head = make_ppo_head(feat_dim, 1, **head_kwargs)

        # Orthogonal initialization
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                out_dim = m.out_features
                if self.continuous_action and out_dim == final_out_dim:
                    # This is the final policy‐output layer (mean & log_std):
                    gain = 0.01
                elif (not self.continuous_action) and out_dim == action_size:
                    # Final policy‐output layer (discrete‐logits):
                    gain = 0.01
                elif out_dim == 1:
                    # Final value‐output layer:
                    gain = 1.0
                else:
                    # Any intermediate hidden layer:
                    gain = math.sqrt(2)
                nn.init.orthogonal_(m.weight, gain)
                nn.init.constant_(m.bias, 0.0)

        self.apply(_init_weights)

        # Create a value function wrapper that can be called like a module
        class ValueFunctionWrapper(nn.Module):
            def __init__(self, parent_model):
                super().__init__()
                self.parent_model = parent_model

            def forward(self, x):
                return self.parent_model.forward_value(x)

        self.value_function_module = ValueFunctionWrapper(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.

        Returns:
        - If discrete: logits tensor
        - If continuous + tanh_squash: (action, z, mean, log_std)
        - If continuous + policy_dist == "beta": (action, alpha, beta)
        - If continuous + not tanh_squash + policy_dist == "gaussian": (action, mean, log_std)
        """

        if self.continuous_action:
            if self.tanh_squash:
                # TANH SQUASHING MODE (4-tuple return)
                feats = self.feature_extractor_policy(x)
                raw = self.policy_head(feats)  # [batch, 2 * action_size]
                mean, log_std = raw.chunk(2, dim=-1)  # each [batch, action_size]
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                std = torch.exp(log_std)  # [batch, action_size]

                # Sample a raw Gaussian z for reparameterization
                eps = torch.randn_like(mean)
                z = mean + std * eps  # [batch, action_size]
                action = torch.tanh(z)  # squash to [−1, +1]

                return action, z, mean, log_std

            elif self.policy_dist == "beta":
                # BETA MODE (3-tuple return): action already in [0, 1], no
                # tanh/rescale needed since Beta's support matches the bounds.
                feats = self.feature_extractor_policy(x)
                raw = self.policy_head(feats)  # [batch, 2 * action_size]
                raw_alpha, raw_beta = raw.chunk(2, dim=-1)  # each [batch, action_size]
                # softplus(.) + 1.0 keeps alpha, beta >= 1: unimodal/concave Beta,
                # near-uniform at init (matches orthogonal gain=0.01 init).
                alpha = F.softplus(raw_alpha) + 1.0
                beta = F.softplus(raw_beta) + 1.0
                dist = torch.distributions.Beta(alpha, beta)
                action = dist.rsample()  # already in [0, 1]

                return action, alpha, beta

            else:
                # STANDARD PPO MODE (3-tuple return)
                feats = self.feature_extractor_policy(x)
                mean = self.policy_head(feats)  # [batch, action_size]

                # Use the learnable log_std parameter
                log_std = self.log_std.expand_as(mean)  # [batch, action_size]
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                std = torch.exp(log_std)  # [batch, action_size]

                # Sample directly from Normal distribution (NO TANH)
                eps = torch.randn_like(mean)
                action = mean + std * eps  # [batch, action_size] - direct sampling

                return action, mean, log_std

        else:
            # DISCRETE ACTION MODE
            feats = self.feature_extractor_policy(x)
            logits = self.policy_head(feats)  # [batch, action_size]
            return logits

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network."""
        feats = self.feature_extractor_value(x)
        result = self.value_head(feats)  # [batch, 1]
        return result


def make_ppo_head(
    in_size, outsize, hidden_sizes=None, layer_norm=True, activation="relu"
):
    """Make PPO head network."""

    # Make fully connected layers
    if hidden_sizes is None:
        hidden_sizes = []

    layers = []
    last_size = in_size
    if isinstance(last_size, list):
        last_size = last_size[0]

    for size in hidden_sizes:
        layers.append(nn.Linear(last_size, size))
        if layer_norm:
            layers.append(nn.LayerNorm(size))
        layers.append(ACTIVATIONS[activation]())
        last_size = size
    layers.append(nn.Linear(last_size, outsize))

    return nn.Sequential(*layers)
