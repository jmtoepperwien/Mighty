from __future__ import annotations

import pytest
import torch

from mighty.mighty_models.ppo import PPOModel
from mighty.mighty_update import PPOUpdate


class TestPPOBetaPolicy:
    """Tests for the Beta-distribution continuous-action policy head."""

    def test_beta_forward_shapes_and_ranges(self):
        """Beta mode should return a 3-tuple with actions in [0, 1]."""
        ppo = PPOModel(
            obs_shape=4,
            action_size=2,
            continuous_action=True,
            policy_dist="beta",
        )

        dummy_input = torch.rand((16, 4))
        out = ppo(dummy_input)

        assert isinstance(out, tuple) and len(out) == 3, (
            "Beta mode should return a 3-tuple (action, alpha, beta)"
        )
        action, alpha, beta = out

        assert action.shape == (16, 2), "Action should have shape (16, 2)"
        assert alpha.shape == (16, 2), "Alpha should have shape (16, 2)"
        assert beta.shape == (16, 2), "Beta should have shape (16, 2)"

        # No NaNs anywhere
        assert torch.all(torch.isfinite(action)), "Actions should be finite"
        assert torch.all(torch.isfinite(alpha)), "Alpha should be finite"
        assert torch.all(torch.isfinite(beta)), "Beta should be finite"

        # Actions must stay within Beta's [0, 1] support
        assert torch.all(action >= 0.0) and torch.all(action <= 1.0), (
            "Actions should be in [0, 1] range"
        )

        # softplus(.) + 1.0 parameterization keeps alpha, beta >= 1.0
        assert torch.all(alpha >= 1.0), "Alpha should be >= 1.0"
        assert torch.all(beta >= 1.0), "Beta should be >= 1.0"

    def test_beta_output_style_and_policy_dist_attrs(self):
        """Beta mode should set policy_dist and output_style consistently."""
        ppo = PPOModel(
            obs_shape=4,
            action_size=2,
            continuous_action=True,
            policy_dist="beta",
        )

        assert ppo.policy_dist == "beta"
        assert ppo.output_style == "beta"
        assert ppo.log_std is None, "log_std should be unused (None) in beta mode"

    def test_beta_and_tanh_squash_are_mutually_exclusive(self):
        """Constructing with both policy_dist='beta' and tanh_squash=True should fail."""
        with pytest.raises(AssertionError):
            PPOModel(
                obs_shape=4,
                action_size=2,
                continuous_action=True,
                policy_dist="beta",
                tanh_squash=True,
            )

    def test_invalid_policy_dist_raises(self):
        """An unrecognized policy_dist value should fail fast."""
        with pytest.raises(AssertionError):
            PPOModel(
                obs_shape=4,
                action_size=2,
                continuous_action=True,
                policy_dist="not_a_real_distribution",
            )

    def test_beta_gradients_flow(self):
        """Gradients from a Beta log-prob loss should reach the shared trunk."""
        ppo = PPOModel(
            obs_shape=4,
            action_size=2,
            continuous_action=True,
            policy_dist="beta",
        )

        dummy_input = torch.rand((8, 4))
        action, alpha, beta = ppo(dummy_input)

        dist = torch.distributions.Beta(alpha, beta)
        log_prob = dist.log_prob(action).sum(dim=-1)
        loss = -log_prob.mean()
        loss.backward()

        feature_extractor_params = list(ppo.feature_extractor_policy.parameters())
        policy_head_params = list(ppo.policy_head.parameters())

        assert any(p.grad is not None for p in feature_extractor_params), (
            "feature_extractor_policy should have received gradients"
        )
        assert any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in feature_extractor_params
        ), "feature_extractor_policy gradients should not be all-zero"

        assert any(p.grad is not None for p in policy_head_params), (
            "policy_head should have received gradients"
        )
        assert any(
            p.grad is not None and torch.any(p.grad != 0) for p in policy_head_params
        ), "policy_head gradients should not be all-zero"

    def test_gaussian_default_unchanged(self):
        """Default policy_dist='gaussian' should be fully backward compatible."""
        ppo = PPOModel(obs_shape=4, action_size=2, continuous_action=True)

        assert ppo.policy_dist == "gaussian"
        assert ppo.output_style == "standard_ppo"

        dummy_input = torch.rand((5, 4))
        action, mean, log_std = ppo(dummy_input)

        assert action.shape == (5, 2)
        assert mean.shape == (5, 2)
        assert log_std.shape == (5, 2)

    def test_ppo_update_end_to_end_with_beta_model(self):
        """A real PPOUpdate.update() call should exercise both the main
        policy-loss beta branch and the post-update KL-diagnostic beta
        branch in ppo_update.py, producing finite metrics.

        Both branches run unconditionally for every minibatch/epoch when
        model.output_style == "beta" and model.continuous_action is True,
        so a single update() call already covers both code paths.
        """
        obs_dim = 4
        action_dim = 2
        batch_size = 8

        model = PPOModel(
            obs_shape=obs_dim,
            action_size=action_dim,
            continuous_action=True,
            policy_dist="beta",
        )

        update_fn = PPOUpdate(
            model=model,
            policy_lr=3e-4,
            value_lr=3e-4,
            epsilon=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=1,
            minibatch_size=batch_size,
            kl_target=0.01,
            adaptive_lr=False,
        )

        class _MiniBatch:
            def __init__(self):
                self.observations = torch.rand(batch_size, obs_dim)
                # Beta's support is [0, 1]; actions must stay inside it.
                self.actions = torch.rand(batch_size, action_dim)
                self.latents = None  # unused outside tanh_squash mode
                self.log_probs = torch.randn(batch_size)
                self.returns = torch.randn(batch_size)
                self.advantages = torch.randn(batch_size)

        class _MaxiBatch:
            def __init__(self):
                self.minibatches = [_MiniBatch(), _MiniBatch()]
                self.advantages = torch.cat(
                    [mb.advantages for mb in self.minibatches]
                )

        batch = _MaxiBatch()

        metrics = update_fn.update(batch)

        required_metrics = [
            "Update/policy_loss",
            "Update/value_loss",
            "Update/entropy",
            "Update/approx_kl",
        ]
        for metric_name in required_metrics:
            assert metric_name in metrics, f"Missing metric: {metric_name}"
            value = metrics[metric_name]
            assert isinstance(value, (int, float)), (
                f"Metric {metric_name} should be scalar"
            )
            assert torch.isfinite(torch.tensor(float(value))), (
                f"Metric {metric_name} should be finite, got {value}"
            )
