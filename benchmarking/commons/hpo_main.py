# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Optional, List
import logging
from argparse import ArgumentParser
import copy

try:
    from coolname import generate_slug
except ImportError:
    print("coolname is not installed, will not be used")


def parse_args(methods: dict, extra_args: Optional[List[dict]] = None):
    try:
        default_experiment_tag = generate_slug(2)
    except Exception:
        default_experiment_tag = "syne_tune_experiment"
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=default_experiment_tag,
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="Number of seeds to run",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="First seed to run",
    )
    parser.add_argument("--method", type=str, required=False, help="HPO method to run")
    parser.add_argument(
        "--save_tuner",
        type=int,
        default=0,
        help="Serialize Tuner object at the end of tuning?",
    )
    if extra_args is not None:
        extra_args = copy.deepcopy(extra_args)
        for kwargs in extra_args:
            name = kwargs.pop("name")
            parser.add_argument("--" + name, **kwargs)
    args, _ = parser.parse_known_args()
    args.save_tuner = bool(args.save_tuner)
    seeds = list(range(args.start_seed, args.num_seeds))
    method_names = [args.method] if args.method is not None else list(methods.keys())
    return args, method_names, seeds


def set_logging_level(args):
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
        logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
        logging.getLogger(
            "syne_tune.backend.simulator_backend.simulator_backend"
        ).setLevel(logging.WARNING)


def get_metadata(
    seed, method, experiment_tag, benchmark_name, benchmark=None, extra_args=None
):
    metadata = {
        "seed": seed,
        "algorithm": method,
        "tag": experiment_tag,
        "benchmark": benchmark_name,
    }
    if benchmark is not None:
        metadata.update(
            {
                "n_workers": benchmark.n_workers,
                "max_wallclock_time": benchmark.max_wallclock_time,
            }
        )
    if extra_args is not None:
        metadata.update(extra_args)
    return metadata
