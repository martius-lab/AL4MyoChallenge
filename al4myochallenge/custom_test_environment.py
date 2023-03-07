import numpy as np
from tonic import logger


def test_die_reorient(env, agent, steps, params=None, test_episodes=20):
    """
    Tests the agent on the test environment.
    Specifically for the die reorientation task.
    """
    # Start the environment.
    if not hasattr(env, "test_observations"):
        env.test_observations = env.start()
        assert len(env.test_observations) == 1

    # Test loop.
    for _ in range(test_episodes):
        metrics = {
            "test/episode_score": 0,
            "test/episode_length": 0,
            "test/effort": 0,
            "test/solved_rate": 0,
            "test/real_solved_rate": 0,
            "test/dropped_rate": 0,
            "test/success_rate": 0,
        }
        while True:
            # Select an action.
            actions = agent.test_step(env.test_observations, steps)
            assert not np.isnan(actions.sum())
            logger.store("test/action", actions, stats=True)

            # Take a step in the environment.
            env.test_observations, info = env.step(actions)

            # Update metrics
            metrics["test/episode_score"] += info["rewards"][0]
            metrics["test/episode_length"] += 1
            metrics["test/effort"] += np.linalg.norm(actions)
            metrics["test/solved_rate"] += int(
                env.environments[0].unwrapped.rwd_dict["solved"]
            )
            metrics["test/dropped_rate"] += int(
                env.environments[0].unwrapped.rwd_dict["done"]
            )

            if info["resets"][0]:
                if info["terminations"]:
                    metrics["test/success_rate"] += 1
                break
        # Log the data.
        metrics["test/solved_rate"] /= (
            metrics["test/episode_length"] * test_episodes
        )
        metrics["test/real_solved_rate"] /= test_episodes
        metrics["test/dropped_rate"] /= (
            metrics["test/episode_length"] * test_episodes
        )
        metrics["test/effort"] /= (
            metrics["test/episode_length"] * test_episodes
        )
        for k, v in metrics.items():
            logger.store(k, v, stats=True)
    return metrics


def test_baoding(env, agent, steps, params=None, test_episodes=50):
    """
    Tests the agent on the test environment.
    Specifically for the die reorientation task.
    """
    # Start the environment.
    if not hasattr(env, "test_observations"):
        env.test_observations = env.start()
        assert len(env.test_observations) == 1

    # Test loop.
    for _ in range(test_episodes):
        metrics = {
            "test/episode_score": 0,
            "test/episode_length": 0,
            "test/effort": 0,
            "test/solved_rate": 0,
            "test/dropped_rate": 0,
        }
        while True:
            # Select an action.
            actions = agent.test_step(env.test_observations, steps)
            assert not np.isnan(actions.sum())
            logger.store("test/action", actions, stats=True)

            # Take a step in the environment.
            env.test_observations, info = env.step(actions)

            # Update metrics
            metrics["test/episode_score"] += info["rewards"][0]
            metrics["test/episode_length"] += 1
            metrics["test/effort"] += np.linalg.norm(actions)
            metrics["test/solved_rate"] += int(
                env.environments[0].unwrapped.rwd_dict["solved"]
            )
            metrics["test/dropped_rate"] += int(
                env.environments[0].unwrapped.rwd_dict["done"]
            )

            if info["resets"][0]:
                break
        # Log the data.
        # dont divide by test episodes here, we log every episode. the logger
        # averages automatically
        metrics["test/solved_rate"] /= 200
        metrics["test/effort"] /= metrics["test/episode_length"]
        for k, v in metrics.items():
            logger.store(k, v, stats=True)
    return metrics
