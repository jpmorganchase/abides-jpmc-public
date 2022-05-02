import gym
from tqdm import tqdm

# Import to register environments
import abides_gym


def test_gym_runner_markets_execution():

    env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
    )

    env.seed(0)
    state = env.reset()
    for i in range(5):
        state, reward, done, info = env.step(0)
    env.step(1)
    env.step(2)
    env.seed()
    env.reset()
    env.close()


def test_gym_runner_markets_daily_investor():

    env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
    )

    env.seed(0)
    state = env.reset()
    for i in range(5):
        state, reward, done, info = env.step(0)
    env.step(1)
    env.step(2)
    env.seed()
    env.reset()
    env.close()
