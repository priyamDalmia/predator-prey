import numpy as np
from gymnasium.spaces import Discrete
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
from ray.rllib.models.modelv2 import ModelV2
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.models.torch.misc import (
    normc_initializer as torch_normc_initializer,
    SlimFC,
)

import torch
import torch.nn as nn

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

class ComplexInputNetwork(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).

    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.

    The data flow is as follows:

    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and value heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        num_outputs = 1
        if log_once("complex_input_net_deprecation_torch"):
            deprecation_warning(
                old="ray.rllib.models.torch.complex_input_net.ComplexInputNetwork",
            )
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )

        self.processed_obs_space = (
            self.original_space
            if model_config.get("_disable_preprocessor_api")
            else obs_space
        )

        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, self.original_space, action_space, num_outputs, model_config, name
        )

        self.flattened_input_space = flatten_space(self.original_space)

        # Atari type CNNs or IMPALA type CNNs (with residual layers)?
        # self.cnn_type = self.model_config["custom_model_config"].get(
        #     "conv_type", "atari")

        # Build the CNN(s) given obs_space's image components.
        self.cnns = nn.ModuleDict()
        self.one_hot = nn.ModuleDict()
        self.flatten_dims = {}
        self.flatten = nn.ModuleDict()
        concat_size = 0
        for i, component in enumerate(self.flattened_input_space):
            i = str(i)
            # Image space.
            if len(component.shape) == 3 and isinstance(component, Box):
                config = {
                    "conv_filters": [[16, [3, 3], 2]],
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                }
                # if self.cnn_type == "atari":
                self.cnns[i] = ModelCatalog.get_model_v2(
                    component,
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="torch",
                    name="cnn_{}".format(i),
                )
                # TODO (sven): add IMPALA-style option.
                # else:
                #    cnn = TorchImpalaVisionNet(
                #        component,
                #        action_space,
                #        num_outputs=None,
                #        model_config=config,
                #        name="cnn_{}".format(i))

                concat_size += self.cnns[i].num_outputs
                self.add_module("cnn_{}".format(i), self.cnns[i])
            # Discrete|MultiDiscrete inputs -> One-hot encode.
            elif isinstance(component, (Discrete, MultiDiscrete)):
                if isinstance(component, Discrete):
                    size = component.n
                else:
                    size = np.sum(component.nvec)
                config = {
                    "fcnet_hiddens": model_config["fcnet_hiddens"],
                    "fcnet_activation": model_config.get("fcnet_activation"),
                    "post_fcnet_hiddens": [],
                }
                self.one_hot[i] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32),
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="torch",
                    name="one_hot_{}".format(i),
                )
                concat_size += self.one_hot[i].num_outputs
                self.add_module("one_hot_{}".format(i), self.one_hot[i])
            # Everything else (1D Box).
            else:
                size = int(np.product(component.shape))
                config = {
                    "fcnet_hiddens": model_config["fcnet_hiddens"],
                    "fcnet_activation": model_config.get("fcnet_activation"),
                    "post_fcnet_hiddens": [],
                }
                self.flatten[i] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32),
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="torch",
                    name="flatten_{}".format(i),
                )
                self.flatten_dims[i] = size
                concat_size += self.flatten[i].num_outputs
                self.add_module("flatten_{}".format(i), self.flatten[i])

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
            "fcnet_activation": model_config.get("post_fcnet_activation", "relu"),
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"), float("inf"), shape=(concat_size,), dtype=np.float32),
            Discrete(1),
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack",
        )

        self.value_layer = SlimFC(
            in_size=self.post_fc_stack.num_outputs,
            out_size=1,
            initializer=torch_normc_initializer(0.01),
            activation_fn=None,
        )
        self._value_out = None
    
    @override(ModelV2)
    def forward(self, own_obs, opponent_obs):
        # Push through CNN(s).
        cnn_out_0, _= self.cnns['0'](SampleBatch({SampleBatch.OBS: own_obs}))
        cnn_out_1, _ = self.cnns['1'](SampleBatch({SampleBatch.OBS: opponent_obs}))
        outs = [cnn_out_0, cnn_out_1]
        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out = self.value_layer(out)
        self._value_out = torch.reshape(out, [-1])
        return out

    @override(ModelV2)
    def value_function(self):
        return self._value_out

class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        return CCPPOTorchPolicy
    
class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""
    def __init__(self):
        self.compute_central_vf = self.model.central_value_function

class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )

class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Base of the model
        self.model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

        # vf_obs_space = Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(obs_space.shape[0], obs_space.shape[0],  obs_space.shape[-1] * 2),
        #     dtype=np.float32,
        
        # )
        vf_obs_space = Dict(
            {
                0: obs_space,
                1: obs_space,
            }
        )
        self.central_vf = ComplexInputNetwork(vf_obs_space, action_space, num_outputs, model_config, name)
 
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        values = self.central_vf.forward(obs, opponent_obs)
        return torch.reshape(values, [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used

# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    pytorch = True
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
        not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None
        if policy.config["enable_connectors"]:
            [(_, _, opponent_batch)] = list(other_agent_batches.values())
        else:
            [(_, opponent_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.CUR_OBS], policy.device
                ),
                convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                convert_to_torch_tensor(
                    sample_batch[OPPONENT_ACTION], policy.device
                ),
            )
            .cpu()
            .detach()
            .numpy()
        )
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch

# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss

