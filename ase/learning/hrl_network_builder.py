from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from learning import ase_network_builder

class HRLBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            critic_shape = kwargs.get('critic_input_shape')
            self._privileged_critic = critic_shape is not None
            self.critic_input_mean_std = None
            if self._privileged_critic:
                if not self.separate:
                    raise ValueError('Privileged HLC critic requires separate networks')
                self.critic_mlp = self._build_mlp(input_size=critic_shape[-1], units=self.units,
                                                   activation=self.activation, dense_func=nn.Linear)
                initializer = self.init_factory.create(**self.initializer)
                for layer in self.critic_mlp.modules():
                    if isinstance(layer, nn.Linear):
                        initializer(layer.weight)
                        nn.init.zeros_(layer.bias)
                if kwargs.get('normalize_critic_input', False):
                    self.critic_input_mean_std = RunningMeanStd(critic_shape)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)

            return
        
        def forward(self, obs_dict):
            if self._privileged_critic:
                obs = obs_dict['obs']
                hidden = self.actor_mlp(self.actor_cnn(obs).reshape(obs.shape[0], -1))
                mu = torch.tanh(self.mu_act(self.mu(hidden)))
                sigma = (mu*0 + self.sigma_act(self.sigma) if self.space_config['fixed_sigma']
                         else self.sigma_act(self.sigma(hidden)))
                critic = obs_dict.get('critic_obs')
                if critic is None and obs_dict.get('is_train', True):
                    raise ValueError('Training requires privileged critic_obs')
                value = self.eval_critic(critic) if critic is not None else obs.new_zeros((len(obs), self.value_size))
                return mu, sigma, value, obs_dict.get('rnn_states')
            mu, sigma, value, states = super().forward(obs_dict)
            norm_mu = torch.tanh(mu)
            return norm_mu, sigma, value, states

        def eval_critic(self, obs):
            if self.critic_input_mean_std is not None:
                obs = self.critic_input_mean_std(obs)
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = HRLBuilder.Network(self.params, **kwargs)
        return net
