from al4myochallenge.env_wrappers.wrappers import ExceptionWrapper, GymWrapper


def apply_wrapper(env):
    return ExceptionWrapper(env)
