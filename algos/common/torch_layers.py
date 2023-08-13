from typing import Dict, List, Tuple, Type, Union
import torch as th
from torch import nn
from stable_baselines3.common.utils import get_device

class TriExtractor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        #hazard_net: List[nn.Module] = []
        alpha_net: List[nn.Module] = []
        beta_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim
        #last_layer_dim_hf = feature_dim
        last_layer_dim_al = feature_dim
        last_layer_dim_be = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
            #hf_layers_dims = net_arch.get("hf", [])  # Layer sizes of the hazard network
            al_layers_dims = net_arch.get("al", [])  # Layer sizes of the alpha network
            be_layers_dims = net_arch.get("be", [])  # Layer sizes of the beta network

        else:
            pi_layers_dims = vf_layers_dims = al_layers_dims = be_layers_dims  = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        for curr_layer_dim in al_layers_dims:
            alpha_net.append(nn.Linear(last_layer_dim_al, curr_layer_dim))
            alpha_net.append(activation_fn())
            last_layer_dim_al = curr_layer_dim

        for curr_layer_dim in be_layers_dims:
            beta_net.append(nn.Linear(last_layer_dim_be, curr_layer_dim))
            beta_net.append(activation_fn())
            last_layer_dim_be = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        #self.latent_dim_hf = last_layer_dim_hf
        self.latent_dim_al = last_layer_dim_al
        self.latent_dim_be = last_layer_dim_be

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        #self.hazard_net = nn.Sequential(*hazard_net).to(device)
        self.alpha_net = nn.Sequential(*alpha_net).to(device)
        self.beta_net = nn.Sequential(*beta_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features), self.forward_alpha(features), self.forward_beta(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward_alpha(self, features: th.Tensor) -> th.Tensor:
        return self.alpha_net(features)

    def forward_beta(self, features: th.Tensor) -> th.Tensor:
        return self.beta_net(features)