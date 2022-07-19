import os  
import os.path as osp 

import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():

    config = suite.load_controller_config(default_controller="JOINT_VELOCITY")

    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_contact_obs=True,
        use_pressure_obs=True,
    )
    env = GymWrapper(env)
    return env

def train_policy(env, timesteps=4000000):
    """
    Train the expert policy in RL
    """
    from stable_baselines3 import SAC

    model_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../expert_models'))
    os.makedirs(model_path, exist_ok=True)

    model = SAC("MlpPolicy", env, gradient_steps=1, verbose=1)
    
    from stable_baselines3.common.logger import configure
    data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../logs'))
    policy_name = "SAC_Lift_Panda_JOINT_VELOCITY"
    tmp_path = data_path + f'/{policy_name}'
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=timesteps, log_interval=4)

    policy_name = model_path + f'/{policy_name}'
    model.save(policy_name)
    return model


def main():

    env = make_vec_env(make_env, vec_env_cls=SubprocVecEnv, n_envs=8)

    model = train_policy(env)

if __name__ == '__main__':
    main()