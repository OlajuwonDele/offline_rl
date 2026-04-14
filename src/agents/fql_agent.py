from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        z = torch.randn(observation.size(0), self.action_dim, device=observation.device)
        action = self.onestep_actor(observation, z)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        dt = 1.0 / self.flow_steps
        action = noise
        for i in range(self.flow_steps):
            t = i * dt
            t_tensor = torch.full(
                (observation.shape[0], 1), t, 
                device=observation.device
            )
            action = action + self.bc_actor(observation, action, t_tensor) * dt
        action = torch.clamp(action, -1, 1)
        return action

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        q = self.critic(observations, actions)
        z = torch.randn(actions.size(0), self.action_dim, device=actions.device)

        with torch.no_grad():
            one_step_action = self.onestep_actor(next_observations, z)
            one_step_action = torch.clamp(one_step_action, -1, 1)

            next_q = self.target_critic(next_observations, one_step_action).mean(dim=0).squeeze(-1)
            target = rewards + self.discount * (1 - dones) * next_q

        loss = nn.functional.mse_loss(q, target)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        z = torch.randn(actions.size(0), self.action_dim, device=actions.device)
        t = torch.rand(actions.size(0), 1, device=actions.device)
        a_tilda = ((1 - t) * z) + (t * actions)
        
        v_pred = self.bc_actor(observations, a_tilda, t)
        target = actions - z
    
        loss = nn.functional.mse_loss(v_pred, target).mean()
        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss

        z = torch.randn(actions.size(0), self.action_dim, device=actions.device)
        one_step_action = self.onestep_actor(observations, z) 
        bc_action = self.get_bc_action(observations, z)
        distill_loss = self.alpha * ((one_step_action - bc_action.detach()) ** 2).mean(dim=-1).mean()

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        one_step_action = torch.clamp(one_step_action, -1, 1)
        q_loss = -self.critic(observations, one_step_action).mean(dim=0).mean()

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = ((one_step_action - actions) ** 2).mean(dim=-1).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                (1 - self.target_update_rate) * target_param.data 
                + self.target_update_rate * param.data
            )
