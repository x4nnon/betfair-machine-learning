from enum import Enum
import os
import gymnasium as gym
from gymnasium import Discrete, Box, Env, Space
import joblib
import numpy as np
from RL.rl_simulation import FlumineRLSimulation
from strategies.rl_strategy import RLStrategy
from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO2
from flumine import clients


class Actions(Enum):
    BACK = 1
    LAY = 2


class FlumineEnv(gym.Env):
    def __init__(self, strategy):
        super(FlumineEnv, self).__init__()
        self.on_init(FlumineRLSimulation(clients.SimulatedClient()), strategy)
        self.action_space = Discrete(3)
        self.state_space = Box(low=-np.inf, high=np.inf, dtype=np.float64)

    def on_init(self, flumine_simulator, strategy):
        self.flumine = flumine_simulator
        self.flumine.add_strategy(strategy)

    def __next_observation(self):
        return

    def step(self, action):
        """_summary_

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Order has been made and matched - the action should be to send_order

        # Perform the action in the environment

        # send order
        observation = self.__execute_action(action)

        # Make a decision using Flumine's strategy
        reward, done = self.strategy.decide(flumine_observation)

        # Convert the decision to the format expected by the environment
        gym_action = self._decision_to_action(flumine_decision)

        return observation, reward, done, info

    def reset(self):
        # Reset the environment
        observation = self.reset()

        # Convert the observation to the format expected by Flumine
        flumine_observation = self._observation_to_flumine(observation)

        # Reset Flumine's strategy
        self.strategy.reset(flumine_observation)

        return observation

    def _observation_to_flumine(self, observation):
        # Convert the observation to the format expected by Flumine
        # TODO: Implement this method based on your specific environment
        pass

    def _decision_to_action(self, decision):
        # Convert the decision to the format expected by the environment
        # TODO: Implement this method based on your specific environment
        pass

    def __execute_action(self, action):
        obseravtion = self.strategy.send_order(Actions(action).name)
        return obseravtion


if __name__ == "__main__":
    # multiprocess environment
    regressor = (
        joblib.load(f"models/BayesianRidge.pkl")
        if os.path.exists(f"models/BayesianRidge.pkl")
        else None
    )
    strategy = RLStrategy(informer=regressor)

    env = DummyVecEnv([lambda: FlumineEnv(RLStrategy())])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo2_pre_horse_race")

    del model  # remove to demonstrate saving and loading

    model = PPO2.load("ppo2_pre_horse_race")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
