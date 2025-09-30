
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ReachActorCritic
from rsl_rl.storage import ReachRolloutStorage
from rsl_rl.storage import DaggerRolloutStorage

class ReachDAgger:
    actor_critic: ReachActorCritic
    def __init__(self,
                 actor_critic,
                 use_flip = False,
                 num_learning_epochs=5,
                 num_mini_batches=8,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 device='cpu',
                 symmetry_scale=1e-3,
                 ):

        self.device = device
        self.use_flip = use_flip
        self.learning_rate = learning_rate

        # BC components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = ReachRolloutStorage.Transition()
        self.transition_sym = ReachRolloutStorage.Transition()
        self.symmetry_scale = symmetry_scale
        
        # BC parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm

        # DAgger dataset
        self.dagger_dataset = None # initialized later


    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = ReachRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def init_dagger_dataset(self):
        self.dagger_dataset = DaggerRolloutStorage()
    
    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Get policy action
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # Store observations
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        # Handle symmetry observations if enabled
        if self.use_flip:
            obs_sym = self.flip_g1_actor_obs(obs)
            critic_obs_sym = self.flip_g1_critic_obs(critic_obs)
            self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
            self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
            self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
            self.transition_sym.observations = obs_sym
            self.transition_sym.critic_observations = critic_obs_sym
            
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        if self.use_flip:
            next_critic_obs_sym = self.flip_g1_critic_obs(next_critic_obs)
            self.transition_sym.next_critic_observations = next_critic_obs_sym.clone()
            self.transition_sym.rewards = rewards.clone()
            self.transition_sym.dones = dones
            
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        if self.use_flip:
            self.transition_sym.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, gamma=0.99, lam=0.95) #dummy

    def add_dagger_data(self):
        dagger_data= self.storage.get_dagger_dataset() # return observations, gt_actions
        self.dagger_dataset.add_data(dagger_data)

    def update_DAgger(self):
        """Update policy using DAgger loss"""
        mean_DAgger_loss = 0

        # DAgger dataset
        dagger_generator = self.dagger_dataset.get_dagger_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, gt_actions_batch in dagger_generator:
            # Forward pass to get predicted actions
            pred_actions = self.actor_critic.act_inference(obs_batch)
            
            # BC loss (MSE between predicted and expert actions)
            DAgger_loss = torch.mean(torch.sum(torch.square(pred_actions - gt_actions_batch), dim=-1))
            
            # Gradient step
            self.optimizer.zero_grad()
            DAgger_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_DAgger_loss += DAgger_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_DAgger_loss /= num_updates

        return mean_DAgger_loss



    
    def update(self, expert_obs=None, expert_actions=None):
        """Update policy using behavioral cloning loss"""
        mean_bc_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_actor_sym_loss = 0
        mean_critic_sym_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:

            actions_batch =critic_obs_batch[:,-4:]

            # Forward pass to get predicted actions
            pred_actions = self.actor_critic.act_inference(obs_batch)
            
            # BC loss (MSE between predicted and expert actions)
            bc_loss = torch.mean(torch.sum(torch.square(pred_actions - actions_batch), dim=-1))
            # hand_error_pos_loss = torch.mean(torch.sum(torch.square(pred_actions[:, 0:3] - actions_batch[:, 0:3]), dim=-1))
            print(f"pred_actions: {pred_actions[2]}")
            print(f"actions_batch: {actions_batch[2]}")
            
            # print(f"bc_loss: {bc_loss.item()}")
            
            # Estimator update
            if self.use_flip and next_critic_obs_batch is not None:
                flipped_obs_batch = self.flip_g1_actor_obs(obs_batch)
                flipped_next_critic_obs_batch = self.flip_g1_critic_obs(next_critic_obs_batch)
                estimator_update_obs_batch = torch.cat((obs_batch, flipped_obs_batch), dim=0)
                estimator_update_next_critic_obs_batch = torch.cat((next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0)
                estimation_loss, swap_loss = self.actor_critic.update_estimator(estimator_update_obs_batch, estimator_update_next_critic_obs_batch, lr=self.learning_rate)
            elif next_critic_obs_batch is not None:
                estimation_loss, swap_loss = self.actor_critic.update_estimator(obs_batch, next_critic_obs_batch, lr=self.learning_rate)
            else:
                estimation_loss, swap_loss = 0, 0
                
            # Symmetry losses if enabled
            if self.use_flip:
                flipped_obs_batch = self.flip_g1_actor_obs(obs_batch)
                actor_sym_loss = self.symmetry_scale * torch.mean(torch.sum(torch.square(
                    self.actor_critic.act_inference(flipped_obs_batch) - 
                    self.flip_g1_actions(self.actor_critic.act_inference(obs_batch))), dim=-1))
                
                if 'critic_obs_batch' in locals() and critic_obs_batch is not None:
                    flipped_critic_obs_batch = self.flip_g1_critic_obs(critic_obs_batch)
                    critic_sym_loss = self.symmetry_scale * torch.mean(torch.square(
                        self.actor_critic.evaluate(flipped_critic_obs_batch) - 
                        self.actor_critic.evaluate(critic_obs_batch).detach()))
                else:
                    critic_sym_loss = torch.tensor(0.0, device=self.device)
                    
                loss = bc_loss + actor_sym_loss + critic_sym_loss
            else:
                loss = bc_loss
                actor_sym_loss = torch.tensor(0.0, device=self.device)
                critic_sym_loss = torch.tensor(0.0, device=self.device)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_bc_loss += bc_loss.item()
            if isinstance(estimation_loss, (int, float)):
                mean_estimation_loss += estimation_loss
            else:
                mean_estimation_loss += estimation_loss.item()
                
            if isinstance(swap_loss, (int, float)):
                mean_swap_loss += swap_loss
            else:
                mean_swap_loss += swap_loss.item()
                
            if self.use_flip:
                mean_actor_sym_loss += actor_sym_loss.item()
                mean_critic_sym_loss += critic_sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_bc_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        if self.use_flip:
            mean_actor_sym_loss /= num_updates
            mean_critic_sym_loss /= num_updates
        
        self.storage.clear()

        if self.use_flip:
            return mean_bc_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss
        else:
            return mean_bc_loss, mean_estimation_loss, mean_swap_loss, 0, 0

   
    def flip_g1_actor_obs(self, obs):
        proprioceptive_obs = torch.clone(obs[:, :(self.actor_critic.num_one_step_obs) * self.actor_critic.actor_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.actor_history_length, self.actor_critic.num_one_step_obs)
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        # print(f"proprioceptive_obs.shape: {proprioceptive_obs.shape}")
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7] # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8] # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9] # projected gravity z
        
        # joint pos
        flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 16] # lower
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 19]
        flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 20]
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 21]
        flipped_proprioceptive_obs[:, :, 16] =  proprioceptive_obs[:, :, 10]
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped_proprioceptive_obs[:, :, 19] =  proprioceptive_obs[:, :, 13]
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 14]
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 15]
        
        flipped_proprioceptive_obs[:, :, 22] =  -proprioceptive_obs[:, :, 22] # waist
        
        flipped_proprioceptive_obs[:, :, 23] =  proprioceptive_obs[:, :, 30] # left shoulder
        flipped_proprioceptive_obs[:, :, 24] = -proprioceptive_obs[:, :, 31]
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 32]
        flipped_proprioceptive_obs[:, :, 26] =  proprioceptive_obs[:, :, 33] # elbow
        flipped_proprioceptive_obs[:, :, 27] = -proprioceptive_obs[:, :, 34] # wrist
        flipped_proprioceptive_obs[:, :, 28] =  proprioceptive_obs[:, :, 35]
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 36]

        
        flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 23] # right shoulder
        flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 24]
        flipped_proprioceptive_obs[:, :, 32] = -proprioceptive_obs[:, :, 25]
        flipped_proprioceptive_obs[:, :, 33] =  proprioceptive_obs[:, :, 26] # elbow
        flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 27] # wrist
        flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 28]
        flipped_proprioceptive_obs[:, :, 36] = -proprioceptive_obs[:, :, 29]
        
        # joint vel
        flipped_proprioceptive_obs[:, :, 10+27] =  proprioceptive_obs[:, :, 16+27] # lower
        flipped_proprioceptive_obs[:, :, 11+27] = -proprioceptive_obs[:, :, 17+27]
        flipped_proprioceptive_obs[:, :, 12+27] = -proprioceptive_obs[:, :, 18+27]
        flipped_proprioceptive_obs[:, :, 13+27] =  proprioceptive_obs[:, :, 19+27]
        flipped_proprioceptive_obs[:, :, 14+27] =  proprioceptive_obs[:, :, 20+27]
        flipped_proprioceptive_obs[:, :, 15+27] = -proprioceptive_obs[:, :, 21+27]
        flipped_proprioceptive_obs[:, :, 16+27] =  proprioceptive_obs[:, :, 10+27]
        flipped_proprioceptive_obs[:, :, 17+27] = -proprioceptive_obs[:, :, 11+27]
        flipped_proprioceptive_obs[:, :, 18+27] = -proprioceptive_obs[:, :, 12+27]
        flipped_proprioceptive_obs[:, :, 19+27] =  proprioceptive_obs[:, :, 13+27]
        flipped_proprioceptive_obs[:, :, 20+27] =  proprioceptive_obs[:, :, 14+27]
        flipped_proprioceptive_obs[:, :, 21+27] = -proprioceptive_obs[:, :, 15+27]
        
        flipped_proprioceptive_obs[:, :, 22+27] =  -proprioceptive_obs[:, :, 22+27] # waist
        
        flipped_proprioceptive_obs[:, :, 23+27] =  proprioceptive_obs[:, :, 30+27] # left shoulder
        flipped_proprioceptive_obs[:, :, 24+27] = -proprioceptive_obs[:, :, 31+27]
        flipped_proprioceptive_obs[:, :, 25+27] = -proprioceptive_obs[:, :, 32+27]
        flipped_proprioceptive_obs[:, :, 26+27] =  proprioceptive_obs[:, :, 33+27] # elbow
        flipped_proprioceptive_obs[:, :, 27+27] = -proprioceptive_obs[:, :, 34+27] # wrist
        flipped_proprioceptive_obs[:, :, 28+27] =  proprioceptive_obs[:, :, 35+27]
        flipped_proprioceptive_obs[:, :, 29+27] = -proprioceptive_obs[:, :, 36+27]

        
        flipped_proprioceptive_obs[:, :, 30+27] =  proprioceptive_obs[:, :, 23+27] # right shoulder
        flipped_proprioceptive_obs[:, :, 31+27] = -proprioceptive_obs[:, :, 24+27]
        flipped_proprioceptive_obs[:, :, 32+27] = -proprioceptive_obs[:, :, 25+27]
        flipped_proprioceptive_obs[:, :, 33+27] =  proprioceptive_obs[:, :, 26+27] # elbow
        flipped_proprioceptive_obs[:, :, 34+27] = -proprioceptive_obs[:, :, 27+27] # wrist
        flipped_proprioceptive_obs[:, :, 35+27] =  proprioceptive_obs[:, :, 28+27]
        flipped_proprioceptive_obs[:, :, 36+27] = -proprioceptive_obs[:, :, 29+27]
        
        # joint target
        flipped_proprioceptive_obs[:, :, 10+54] =  proprioceptive_obs[:, :, 16+54] # lower
        flipped_proprioceptive_obs[:, :, 11+54] = -proprioceptive_obs[:, :, 17+54]
        flipped_proprioceptive_obs[:, :, 12+54] = -proprioceptive_obs[:, :, 18+54]
        flipped_proprioceptive_obs[:, :, 13+54] =  proprioceptive_obs[:, :, 19+54]
        flipped_proprioceptive_obs[:, :, 14+54] =  proprioceptive_obs[:, :, 20+54]
        flipped_proprioceptive_obs[:, :, 15+54] = -proprioceptive_obs[:, :, 21+54]
        flipped_proprioceptive_obs[:, :, 16+54] =  proprioceptive_obs[:, :, 10+54]
        flipped_proprioceptive_obs[:, :, 17+54] = -proprioceptive_obs[:, :, 11+54]
        flipped_proprioceptive_obs[:, :, 18+54] = -proprioceptive_obs[:, :, 12+54]
        flipped_proprioceptive_obs[:, :, 19+54] =  proprioceptive_obs[:, :, 13+54]
        flipped_proprioceptive_obs[:, :, 20+54] =  proprioceptive_obs[:, :, 14+54]
        flipped_proprioceptive_obs[:, :, 21+54] = -proprioceptive_obs[:, :, 15+54]

        flipped_proprioceptive_obs[:, :, 22+54] =  proprioceptive_obs[:, :, 22+54] # waist

        flipped_proprioceptive_obs[:, :, 23+54] =  proprioceptive_obs[:, :, 30+54] # left shoulder
        flipped_proprioceptive_obs[:, :, 24+54] =  proprioceptive_obs[:, :, 31+54]
        flipped_proprioceptive_obs[:, :, 25+54] =  proprioceptive_obs[:, :, 32+54]
        flipped_proprioceptive_obs[:, :, 26+54] =  proprioceptive_obs[:, :, 33+54] 
        flipped_proprioceptive_obs[:, :, 27+54] =  proprioceptive_obs[:, :, 34+54] 
        flipped_proprioceptive_obs[:, :, 28+54] =  proprioceptive_obs[:, :, 35+54]
        flipped_proprioceptive_obs[:, :, 29+54] =  proprioceptive_obs[:, :, 36+54]
        flipped_proprioceptive_obs[:, :, 30+54] =  proprioceptive_obs[:, :, 23+54] # right shoulder
        flipped_proprioceptive_obs[:, :, 31+54] =  proprioceptive_obs[:, :, 24+54]
        flipped_proprioceptive_obs[:, :, 32+54] =  proprioceptive_obs[:, :, 25+54]
        flipped_proprioceptive_obs[:, :, 33+54] =  proprioceptive_obs[:, :, 26+54]
        flipped_proprioceptive_obs[:, :, 34+54] =  proprioceptive_obs[:, :, 27+54]
        flipped_proprioceptive_obs[:, :, 35+54] =  proprioceptive_obs[:, :, 28+54]
        flipped_proprioceptive_obs[:, :, 36+54] =  proprioceptive_obs[:, :, 29+54]


        return flipped_proprioceptive_obs.view(-1, (self.actor_critic.num_one_step_obs) * self.actor_critic.actor_history_length).detach()                                                                                                                                                                                                                                             
    
    def flip_g1_critic_obs(self, critic_obs):
        proprioceptive_obs = torch.clone(critic_obs[:, :self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.critic_history_length, self.actor_critic.num_one_step_critic_obs)
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        
        # print(f"proprioceptive_obs.shape: {proprioceptive_obs.shape}")
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7] # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8] # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9] # projected gravity z
        
        # joint pos
        flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 16] # lower
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 19]
        flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 20]
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 21]
        flipped_proprioceptive_obs[:, :, 16] =  proprioceptive_obs[:, :, 10]
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped_proprioceptive_obs[:, :, 19] =  proprioceptive_obs[:, :, 13]
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 14]
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 15]
        
        flipped_proprioceptive_obs[:, :, 22] =  -proprioceptive_obs[:, :, 22] # waist
        
        flipped_proprioceptive_obs[:, :, 23] =  proprioceptive_obs[:, :, 30] # left shoulder
        flipped_proprioceptive_obs[:, :, 24] = -proprioceptive_obs[:, :, 31]
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 32]
        flipped_proprioceptive_obs[:, :, 26] =  proprioceptive_obs[:, :, 33] # elbow
        flipped_proprioceptive_obs[:, :, 27] = -proprioceptive_obs[:, :, 34] # wrist
        flipped_proprioceptive_obs[:, :, 28] =  proprioceptive_obs[:, :, 35]
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 36]

        
        flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 23] # right shoulder
        flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 24]
        flipped_proprioceptive_obs[:, :, 32] = -proprioceptive_obs[:, :, 25]
        flipped_proprioceptive_obs[:, :, 33] =  proprioceptive_obs[:, :, 26] # elbow
        flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 27] # wrist
        flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 28]
        flipped_proprioceptive_obs[:, :, 36] = -proprioceptive_obs[:, :, 29]
        
        # joint vel
        flipped_proprioceptive_obs[:, :, 10+27] =  proprioceptive_obs[:, :, 16+27] # lower
        flipped_proprioceptive_obs[:, :, 11+27] = -proprioceptive_obs[:, :, 17+27]
        flipped_proprioceptive_obs[:, :, 12+27] = -proprioceptive_obs[:, :, 18+27]
        flipped_proprioceptive_obs[:, :, 13+27] =  proprioceptive_obs[:, :, 19+27]
        flipped_proprioceptive_obs[:, :, 14+27] =  proprioceptive_obs[:, :, 20+27]
        flipped_proprioceptive_obs[:, :, 15+27] = -proprioceptive_obs[:, :, 21+27]
        flipped_proprioceptive_obs[:, :, 16+27] =  proprioceptive_obs[:, :, 10+27]
        flipped_proprioceptive_obs[:, :, 17+27] = -proprioceptive_obs[:, :, 11+27]
        flipped_proprioceptive_obs[:, :, 18+27] = -proprioceptive_obs[:, :, 12+27]
        flipped_proprioceptive_obs[:, :, 19+27] =  proprioceptive_obs[:, :, 13+27]
        flipped_proprioceptive_obs[:, :, 20+27] =  proprioceptive_obs[:, :, 14+27]
        flipped_proprioceptive_obs[:, :, 21+27] = -proprioceptive_obs[:, :, 15+27]
        
        flipped_proprioceptive_obs[:, :, 22+27] =  -proprioceptive_obs[:, :, 22+27] # waist
        
        flipped_proprioceptive_obs[:, :, 23+27] =  proprioceptive_obs[:, :, 30+27] # left shoulder
        flipped_proprioceptive_obs[:, :, 24+27] = -proprioceptive_obs[:, :, 31+27]
        flipped_proprioceptive_obs[:, :, 25+27] = -proprioceptive_obs[:, :, 32+27]
        flipped_proprioceptive_obs[:, :, 26+27] =  proprioceptive_obs[:, :, 33+27] # elbow
        flipped_proprioceptive_obs[:, :, 27+27] = -proprioceptive_obs[:, :, 34+27] # wrist
        flipped_proprioceptive_obs[:, :, 28+27] =  proprioceptive_obs[:, :, 35+27]
        flipped_proprioceptive_obs[:, :, 29+27] = -proprioceptive_obs[:, :, 36+27]

        
        flipped_proprioceptive_obs[:, :, 30+27] =  proprioceptive_obs[:, :, 23+27] # right shoulder
        flipped_proprioceptive_obs[:, :, 31+27] = -proprioceptive_obs[:, :, 24+27]
        flipped_proprioceptive_obs[:, :, 32+27] = -proprioceptive_obs[:, :, 25+27]
        flipped_proprioceptive_obs[:, :, 33+27] =  proprioceptive_obs[:, :, 26+27] # elbow
        flipped_proprioceptive_obs[:, :, 34+27] = -proprioceptive_obs[:, :, 27+27] # wrist
        flipped_proprioceptive_obs[:, :, 35+27] =  proprioceptive_obs[:, :, 28+27]
        flipped_proprioceptive_obs[:, :, 36+27] = -proprioceptive_obs[:, :, 29+27]
        
        # joint target
        flipped_proprioceptive_obs[:, :, 10+54] =  proprioceptive_obs[:, :, 16+54] # lower
        flipped_proprioceptive_obs[:, :, 11+54] = -proprioceptive_obs[:, :, 17+54]
        flipped_proprioceptive_obs[:, :, 12+54] = -proprioceptive_obs[:, :, 18+54]
        flipped_proprioceptive_obs[:, :, 13+54] =  proprioceptive_obs[:, :, 19+54]
        flipped_proprioceptive_obs[:, :, 14+54] =  proprioceptive_obs[:, :, 20+54]
        flipped_proprioceptive_obs[:, :, 15+54] = -proprioceptive_obs[:, :, 21+54]
        flipped_proprioceptive_obs[:, :, 16+54] =  proprioceptive_obs[:, :, 10+54]
        flipped_proprioceptive_obs[:, :, 17+54] = -proprioceptive_obs[:, :, 11+54]
        flipped_proprioceptive_obs[:, :, 18+54] = -proprioceptive_obs[:, :, 12+54]
        flipped_proprioceptive_obs[:, :, 19+54] =  proprioceptive_obs[:, :, 13+54]
        flipped_proprioceptive_obs[:, :, 20+54] =  proprioceptive_obs[:, :, 14+54]
        flipped_proprioceptive_obs[:, :, 21+54] = -proprioceptive_obs[:, :, 15+54]

        flipped_proprioceptive_obs[:, :, 22+54] =  proprioceptive_obs[:, :, 22+54] # waist

        flipped_proprioceptive_obs[:, :, 23+54] =  proprioceptive_obs[:, :, 30+54] # left shoulder
        flipped_proprioceptive_obs[:, :, 24+54] =  proprioceptive_obs[:, :, 31+54]
        flipped_proprioceptive_obs[:, :, 25+54] =  proprioceptive_obs[:, :, 32+54]
        flipped_proprioceptive_obs[:, :, 26+54] =  proprioceptive_obs[:, :, 33+54]
        flipped_proprioceptive_obs[:, :, 27+54] =  proprioceptive_obs[:, :, 34+54]
        flipped_proprioceptive_obs[:, :, 28+54] =  proprioceptive_obs[:, :, 35+54]
        flipped_proprioceptive_obs[:, :, 29+54] =  proprioceptive_obs[:, :, 36+54]
        flipped_proprioceptive_obs[:, :, 30+54] =  proprioceptive_obs[:, :, 23+54] # right shoulder
        flipped_proprioceptive_obs[:, :, 31+54] =  proprioceptive_obs[:, :, 24+54]
        flipped_proprioceptive_obs[:, :, 32+54] =  proprioceptive_obs[:, :, 25+54]
        flipped_proprioceptive_obs[:, :, 33+54] =  proprioceptive_obs[:, :, 26+54]
        flipped_proprioceptive_obs[:, :, 34+54] =  proprioceptive_obs[:, :, 27+54]
        flipped_proprioceptive_obs[:, :, 35+54] =  proprioceptive_obs[:, :, 28+54]
        flipped_proprioceptive_obs[:, :, 36+54] =  proprioceptive_obs[:, :, 29+54]

        
        flipped_proprioceptive_obs[:, :, -3] =  proprioceptive_obs[:, :, -3] # base lin vel x
        flipped_proprioceptive_obs[:, :, -2] = -proprioceptive_obs[:, :, -2] # base lin vel y
        flipped_proprioceptive_obs[:, :, -1] =  proprioceptive_obs[:, :, -1] # base lin vel z

        return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length).detach()
    
    def flip_g1_actions(self, actions):
        flipped_actions = torch.zeros_like(actions)

        flipped_actions[:,  0] =  actions[:, 7]     # left shoulder PITCH
        flipped_actions[:,  1] = -actions[:, 8]     # left shoulder ROLL
        flipped_actions[:,  2] = -actions[:, 9]     # left shoulder YAW
        flipped_actions[:,  3] =  actions[:, 10]    # elbow
        flipped_actions[:,  4] = -actions[:, 11]    # wrist ROLL
        flipped_actions[:,  5] =  actions[:, 12]    # wrist PITCH
        flipped_actions[:,  6] = -actions[:, 13]    # wrist YAW
        flipped_actions[:,  7] =  actions[:, 0]    # right shoulder PITCH
        flipped_actions[:,  8] = -actions[:, 1]    # right shoulder ROLL
        flipped_actions[:,  9] = -actions[:, 2]    # right shoulder YAW
        flipped_actions[:,  10] =  actions[:, 3]    # elbow
        flipped_actions[:,  11] = -actions[:, 4]    # wrist ROLL
        flipped_actions[:,  12] =  actions[:, 5]     # wrist PITCH
        flipped_actions[:,  13] = -actions[:, 6]    # wrist YAW
        flipped_actions[:,  14] =  actions[:, 14]    # vel x
        flipped_actions[:,  15] = -actions[:, 15]     # vel y
        return flipped_actions.detach()
    