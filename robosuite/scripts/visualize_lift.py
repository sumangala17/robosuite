import os  
import os.path as osp 
import numpy as np 

import robosuite as suite
from robosuite.wrappers import GymWrapper, VisualizationWrapper

from stable_baselines3 import SAC

def make_env():

    config = suite.load_controller_config(default_controller="JOINT_VELOCITY")
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_contact_obs=True,
        use_pressure_obs=True,
    )
    env = GymWrapper(env)
    env = VisualizationWrapper(env)
    return env

def main():

    env = make_env()
    obs = env.reset()

    policy = SAC.load("./expert_models/SAC_Lift_Panda_JOINT_VELOCITY")

    touch_sensor_names = ['gripper0_touch1', 'gripper0_touch2']

    # while not done:
    for i in range(1000):
        action, _ = policy.predict(obs) # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display

        # ee_force = np.linalg.norm(env.robots[0].ee_force)
        # total_force_ee = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))

        # touch_sensor_values = [env.robots[0].get_sensor_measurement(name)[0] for name in touch_sensor_names]

        # print(i, obs[-1], f"{ee_force:.3f}", f"{total_force_ee:.3f}")
        print(f"Step {i}, contact {obs[-3]}, pressure {obs[-2:]}")


if __name__ == '__main__':
    main()