import gym

from .buffer_wrapper import BufferWrapper
from .fire_reset import FireResetEnv
from .image_to_pytorch import ImageToPyTorch
from .max_and_skip import MaxAndSkipEnv
from .processframes84 import ProcessFrame84
from .scaled_float_frame import ScaledFloatFrame


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env)

    return ScaledFloatFrame(env)