# Source: https://github.com/qxcv/dreamerv3
#
# From_gym.py converted to work with gym==0.26.2, which should be identical to Gymnasium.
# Differences:
# - Deals with .step() returning a tuple of (obs, reward, terminated, truncated,
#   info) rather than (obs, reward, done, info).
# - Also deals with .reset() returning a tuple of (obs, info) rather than just
#   obs.
# - Passes render_mode='rgb_array' to gym.make() rather than .render().
# - A bunch of minor/irrelevant type checking changes that stopped pyright from
#   complaining (these have no functional purpose, I'm just a completionist who
#   doesn't like red squiggles).

import functools
from typing import Any, Generic, TypeVar, Union, cast, Dict

import csv
import embodied
import gym
import gym.spaces
import numpy as np
import os

U = TypeVar('U')
V = TypeVar('V')


class FromGymAAI(embodied.Env, Generic[U, V]):
  def __init__(self, env: Union[str, gym.Env[U, V]], step_csv_path = None, multi_obs = True, obs_key='image', act_key='action', **kwargs): #multi_obs is new
    if isinstance(env, str):
      self._env: gym.Env[U, V] = gym.make(env, render_mode="rgb_array", **kwargs)
    else:
      assert not kwargs, kwargs
      assert env.render_mode == "rgb_array", f"render_mode must be rgb_array, got {env.render_mode}"
      self._env = env

    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    self.multi_obs = multi_obs #multi_obs is new
    self.action = None #new
    self.step_csv_path = step_csv_path
    self._reward = 0

  @property
  def info(self):
    return self._info

  @functools.cached_property # type: ignore
  def obs_space(self):
    if self._obs_dict:
      # cast is here to stop type checkers from complaining (we already check
      # that .spaces attr exists in __init__ as a proxy for the type check)
      obs_space = cast(gym.spaces.Dict, self._env.observation_space)
      spaces = obs_space.spaces
    else:
      spaces = {self._obs_key: self._env.observation_space}
    if self.multi_obs: # multi_obs is new
      spaces = self.tuple_to_dict(spaces) #new method to convert tuple to named dict
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    print(spaces)
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @functools.cached_property # type: ignore
  def act_space(self):
    if self._act_dict:
      act_space = cast(gym.spaces.Dict, self._env.action_space)
      spaces = act_space.spaces
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      # we don't bother setting ._info here because it gets set below, once we
      # take the next .step()
      obs, _ = self._env.reset()
      self._reward = 0

      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      gym_action = cast(V, self._unflatten(action))
    else:
      gym_action = cast(V, action[self._act_key])
    obs, reward, terminated, truncated, self._info = self._env.step(gym_action)
    self._done = terminated or truncated
    #print(f"Gym action taken: {gym_action}")
    self.action = gym_action
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    
    if self.multi_obs: #convert output to image and step_details dictionary to be passed on elsewhere
      obs = {
        'image' : obs[0], #image is the image returned from the environment
        'step_details' : obs[1], #other contains an array of length 7 containing health, x velocity, y velocity, z velocity, x position, y position, z position (in that order)
      }

      if self.step_csv_path is not None:
        file_exists = os.path.isfile(self.step_csv_path)
        with open(self.step_csv_path, 'a' if file_exists else 'w', newline='') as csv_file:
          csv_write = csv.writer(csv_file)
          if not file_exists:
            csv_write.writerow(['actiontaken', 'stepreward', 'xvelocity', 'yvelocity', 'zvelocity', 'xpos', 'ypos', 'zpos'])
          self._reward += reward
          csv_write.writerow([str(self.action), str(self._reward), str(obs['step_details'][1]), str(obs['step_details'][2]), str(obs['step_details'][3]), str(obs['step_details'][4]), str(obs['step_details'][5]), str(obs['step_details'][6])])
  
    if not self._obs_dict:
      obs = {self._obs_key: obs}

    obs = self._flatten(obs)
    np_obs: Dict[str, Any] = {k: np.asarray(v) for k, v in obs.items()}
    np_obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return np_obs

  def render(self):
    image = self._env.render()
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)


  def tuple_to_dict(self, spaces): #new method to convert tuple of spaces to dictionary of spaces
    """Converts a tuple observation to a dictionary.

    For use when multi_obs = True

    Args:
      observation: A tuple observation.

    Returns:
      A dictionary containing the tuple observation values, keyed as camera, step_details.
    """

    observation_dict = {
      'image' : spaces[0],
      'step_details' : spaces[1]
    }

    return observation_dict
