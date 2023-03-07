import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import tonic  # needed for eval(build)

from al4myochallenge.env_wrappers import apply_wrapper


def env_tonic_compat(env, preid=5, parallel=1, sequential=1):
    """
    Applies wrapper for tonic and passes random seed.
    """
    if "ostrich" in env:
        return lambda identifier=0: apply_wrapper(eval(env))

    elif "biped" in env:

        def build_env(identifier=0):
            id_eff = preid * (parallel * sequential) + identifier
            build = env[:-1]
            build = build + f",identifier={id_eff})"
            return apply_wrapper(eval(build))

    else:
        return lambda identifier=0: apply_wrapper(eval(env))
    return build_env


def prepare_files(orig_params):
    params = get_params(orig_params)
    os.makedirs(params.working_dir, exist_ok=True)
    return params


def get_params(orig_params):
    params = orig_params.copy()
    for key, val in params.items():
        if type(params[key]) == dict:
            params[key] = SimpleNamespace(**val)
    params = SimpleNamespace(**params)
    return params


def prepare_params():
    f = open(sys.argv[-1], "r")
    orig_params = json.load(f)
    params = prepare_files(orig_params)
    return orig_params, params
