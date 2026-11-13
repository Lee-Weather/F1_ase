import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim

import learning.common_agent as common_agent 
import learning.ase_agent as ase_agent
import learning.ase_models as ase_models
import learning.ase_network_builder as ase_network_builder
from utils.f1_hlc import FrozenF1LLC, control_interval

from tensorboardX import SummaryWriter

class HRLAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']
            self._latent_dim = llc_config_params['config']['latent_dim']
        
        super().__init__(base_name, config)

        self._task_size = self.vec_env.env.task.get_task_obs_size()
        
        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != "")
        self._build_llc(llc_config_params, llc_checkpoint)

        return

    def env_step(self, actions):
        if self.config.get('frozen_f1_llc', False):
            task = self.vec_env.env.task
            obs, rewards, dones, info = control_interval(
                task, self._llc_agent, self.preprocess_actions(actions), self._llc_steps,
                step=self.vec_env.step)
            return self.obs_to_tensors(obs), rewards.unsqueeze(-1), dones, info
        actions = self.preprocess_actions(actions)
        obs = self.obs['obs']

        rewards = 0.0
        disc_rewards = 0.0
        done_count = 0.0
        terminate_count = 0.0
        for t in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs, actions)
            obs, curr_rewards, curr_dones, infos = self.vec_env.step(llc_actions)
            
            rewards += curr_rewards
            done_count += curr_dones
            terminate_count += infos['terminate']
            
            amp_obs = infos['amp_obs']
            curr_disc_reward = self._calc_disc_reward(amp_obs)
            disc_rewards += curr_disc_reward

        rewards /= self._llc_steps
        disc_rewards /= self._llc_steps

        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos['terminate'] = terminate
        infos['disc_rewards'] = disc_rewards

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def get_action_values(self, obs_dict):
        if not self._use_privileged_critic:
            return super().get_action_values(obs_dict)
        self.model.eval()
        with torch.no_grad():
            result = self.model({'is_train': False, 'prev_actions': None,
                                 'obs': self._preproc_obs(obs_dict['obs']),
                                 'critic_obs': obs_dict['states'], 'rnn_states': self.rnn_states})
            if self.normalize_value:
                result['values'] = self.value_mean_std(result['values'], True)
        return result

    def preprocess_actions(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions

    def play_steps(self):
        self.set_eval()
        
        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            if self._use_privileged_critic:
                self.experience_buffer.update_data('critic_obs', n, self.obs['states'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            
            self.experience_buffer.update_data('disc_rewards', n, infos['disc_rewards'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += infos['llc_substeps'] if self.config.get('frozen_f1_llc', False) else 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_disc_rewards = self.experience_buffer.tensor_dict['disc_rewards']
        raw_task_rewards = mb_rewards.clone()
        mb_rewards = self._combine_rewards(mb_rewards, mb_disc_rewards)
        self._mean_training_reward = float(mb_rewards.mean())

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['task_rewards'] = a2c_common.swap_and_flatten01(raw_task_rewards)

        return batch_dict
    
    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        return

    def _get_mean_rewards(self):
        rewards = super()._get_mean_rewards()
        if self.config.get('frozen_f1_llc', False):
            return rewards
        rewards *= self._llc_steps
        return rewards

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim
        return

    def init_tensors(self):
        super().init_tensors()

        del self.experience_buffer.tensor_dict['actions']
        del self.experience_buffer.tensor_dict['mus']
        del self.experience_buffer.tensor_dict['sigmas']

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['actions'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['mus'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['sigmas'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        
        self.experience_buffer.tensor_dict['disc_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.tensor_list += ['disc_rewards']

        return

    def _build_llc(self, config_params, checkpoint_file):
        if self.config.get('frozen_f1_llc', False):
            self._llc_agent = FrozenF1LLC(checkpoint_file, self.ppo_device)
            task = self.vec_env.env.task
            task.validate_control_config(self._llc_agent.control_config)
            if task.hlc['llcSteps'] != self._llc_steps:
                raise ValueError('HLC llc_steps differs from task configuration')
            print('F1_HLC_LLC ' + str(self._llc_agent.metadata), flush=True)
            return
        network_params = config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_agent = ase_agent.ASEAgent('llc', llc_agent_config)
        self._llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        self._llc_agent.set_eval()
        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])

        config = config_params['config']
        config['network'] = network
        config['num_actors'] = self.num_actors
        config['features'] = {'observer' : self.algo_observer}
        config['env_info'] = llc_env_info

        return config

    def _compute_llc_action(self, obs, actions):
        if self.config.get('frozen_f1_llc', False):
            from utils.f1_hlc import unit_latent
            return self._llc_agent(obs[..., :93], unit_latent(actions))
        llc_obs = self._extract_llc_obs(obs)
        processed_obs = self._llc_agent._preproc_obs(llc_obs)

        z = torch.nn.functional.normalize(actions, dim=-1)
        mu, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs, ase_latents=z)
        llc_action = mu
        llc_action = self._llc_agent.preprocess_actions(llc_action)

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs

    def _calc_disc_reward(self, amp_obs):
        if self.config.get('frozen_f1_llc', False):
            return self._llc_agent.disc_reward(amp_obs)
        disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
        return disc_reward

    def _combine_rewards(self, task_rewards, disc_rewards): 
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_rewards
        
        #combined_rewards = task_rewards * disc_rewards
        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        train_info['task_rewards'] = batch_dict['task_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)
        return

    def get_full_state_weights(self):
        weights = super().get_full_state_weights()
        if self.config.get('frozen_f1_llc', False):
            weights['llc_dependency'] = self._llc_agent.metadata.copy()
            weights['hlc_config'] = dict(llc_steps=self._llc_steps,
                                        task_reward_w=self._task_reward_w, disc_reward_w=self._disc_reward_w)
        return weights

    def set_full_state_weights(self, weights):
        if self.config.get('frozen_f1_llc', False):
            if weights.get('llc_dependency') != self._llc_agent.metadata:
                raise ValueError('HLC checkpoint requires a different LLC')
        return super().set_full_state_weights(weights)
