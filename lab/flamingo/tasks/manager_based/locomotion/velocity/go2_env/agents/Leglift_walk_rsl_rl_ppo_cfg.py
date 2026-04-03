from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UnitreeGo2LegliftWalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 50
    experiment_name = "unitree_go2_leglift_walk"
    run_name = "rear_left_dynamic_walk_v2"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.5,
    )

    def __post_init__(self):
        base_post_init = getattr(super(), "__post_init__", None)
        if callable(base_post_init):
            base_post_init()

        self.num_steps_per_env = 32
        self.max_iterations = 10000
        self.save_interval = 50
        self.experiment_name = "unitree_go2_leglift_walk"
        self.run_name = "rear_left_dynamic_walk_v2"

        if hasattr(self, "policy") and self.policy is not None:
            if hasattr(self.policy, "init_noise_std"):
                self.policy.init_noise_std = 0.8
            if hasattr(self.policy, "actor_obs_normalization"):
                self.policy.actor_obs_normalization = False
            if hasattr(self.policy, "critic_obs_normalization"):
                self.policy.critic_obs_normalization = False
            if hasattr(self.policy, "actor_hidden_dims"):
                self.policy.actor_hidden_dims = [128, 128, 128]
            if hasattr(self.policy, "critic_hidden_dims"):
                self.policy.critic_hidden_dims = [128, 128, 128]
            if hasattr(self.policy, "activation"):
                self.policy.activation = "elu"

        if hasattr(self, "algorithm") and self.algorithm is not None:
            if hasattr(self.algorithm, "value_loss_coef"):
                self.algorithm.value_loss_coef = 1.0
            if hasattr(self.algorithm, "use_clipped_value_loss"):
                self.algorithm.use_clipped_value_loss = True
            if hasattr(self.algorithm, "clip_param"):
                self.algorithm.clip_param = 0.2
            if hasattr(self.algorithm, "entropy_coef"):
                self.algorithm.entropy_coef = 0.003
            if hasattr(self.algorithm, "num_learning_epochs"):
                self.algorithm.num_learning_epochs = 5
            if hasattr(self.algorithm, "num_mini_batches"):
                self.algorithm.num_mini_batches = 4
            if hasattr(self.algorithm, "learning_rate"):
                self.algorithm.learning_rate = 7.5e-5
            if hasattr(self.algorithm, "schedule"):
                self.algorithm.schedule = "adaptive"
            if hasattr(self.algorithm, "gamma"):
                self.algorithm.gamma = 0.99
            if hasattr(self.algorithm, "lam"):
                self.algorithm.lam = 0.95
            if hasattr(self.algorithm, "desired_kl"):
                self.algorithm.desired_kl = 0.006
            if hasattr(self.algorithm, "max_grad_norm"):
                self.algorithm.max_grad_norm = 0.5
