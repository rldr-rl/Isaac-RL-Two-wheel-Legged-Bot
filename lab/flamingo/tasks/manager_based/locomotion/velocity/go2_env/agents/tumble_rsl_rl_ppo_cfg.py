from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg


@configclass
class UnitreeGo2TumblePPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """PPO runner tuned for the staged go2 tumble task.

    This stays intentionally conservative and only overrides fields that are
    usually present on the inherited flat-go2 PPO config.  The hasattr checks
    make the class safer across small IsaacLab version differences.
    """

    def __post_init__(self):
        super().__post_init__()

        # basic run bookkeeping
        self.experiment_name = "unitree_go2_tumble"
        self.run_name = "jump_to_flip_curriculum"
        self.max_iterations = 5000
        self.save_interval = 50

        # rollout size: tumbling is sparse and benefits from longer training.
        if hasattr(self, "num_steps_per_env"):
            self.num_steps_per_env = 24

        # policy network: slightly larger than flat locomotion, but still stable.
        if hasattr(self, "policy"):
            if hasattr(self.policy, "actor_hidden_dims"):
                self.policy.actor_hidden_dims = [512, 256, 128]
            if hasattr(self.policy, "critic_hidden_dims"):
                self.policy.critic_hidden_dims = [512, 256, 128]
            if hasattr(self.policy, "activation"):
                self.policy.activation = "elu"
            if hasattr(self.policy, "init_noise_std"):
                self.policy.init_noise_std = 0.8

        # PPO algorithm knobs.
        if hasattr(self, "algorithm"):
            if hasattr(self.algorithm, "entropy_coef"):
                self.algorithm.entropy_coef = 0.01
            if hasattr(self.algorithm, "learning_rate"):
                self.algorithm.learning_rate = 1.0e-4
            if hasattr(self.algorithm, "num_learning_epochs"):
                self.algorithm.num_learning_epochs = 5
            if hasattr(self.algorithm, "num_mini_batches"):
                self.algorithm.num_mini_batches = 4
            if hasattr(self.algorithm, "clip_param"):
                self.algorithm.clip_param = 0.2
            if hasattr(self.algorithm, "gamma"):
                self.algorithm.gamma = 0.99
            if hasattr(self.algorithm, "lam"):
                self.algorithm.lam = 0.95
            if hasattr(self.algorithm, "desired_kl"):
                self.algorithm.desired_kl = 0.01
            if hasattr(self.algorithm, "max_grad_norm"):
                self.algorithm.max_grad_norm = 1.0
            if hasattr(self.algorithm, "value_loss_coef"):
                self.algorithm.value_loss_coef = 1.0
            if hasattr(self.algorithm, "use_clipped_value_loss"):
                self.algorithm.use_clipped_value_loss = True

        # logging options vary a bit across IsaacLab revisions.
        if hasattr(self, "logger"):
            self.logger = "tensorboard"
        if hasattr(self, "empirical_normalization"):
            self.empirical_normalization = False
