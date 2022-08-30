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
"""
Example for running ASHA with 4 workers with the simulator back-end based on Yahpo surrogate benchmark.
"""
import logging

import matplotlib.pyplot as plt

from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion


def plot_yahpo_learning_curves(trial_backend, time_col: str, metric_col: str):
    bb = trial_backend.blackbox
    plt.figure()
    plt.title("10 learning curves from Yahpo benchmark")
    for i in range(1000):
        config = {k: v.sample() for k, v in bb.configuration_space.items()}
        evals = bb(config)
        time_index = next(
            i for i, name in enumerate(bb.objectives_names) if name == time_col
        )
        accuracy_index = next(
            i for i, name in enumerate(bb.objectives_names) if name == metric_col
        )
        import numpy as np

        if np.diff(evals[:, time_index]).min() < 0:
            print("negative time between two different steps...")
        plt.plot(evals[:, time_index], evals[:, accuracy_index])
    plt.xlabel(time_col)
    plt.ylabel(metric_col)
    plt.show()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    n_workers = 4
    elapsed_time_attr = "time"
    metric = "val_accuracy"

    trial_backend = BlackboxRepositoryBackend(
        blackbox_name="yahpo-lcbench",
        elapsed_time_attr=elapsed_time_attr,
        dataset="3945",
    )

    plot_yahpo_learning_curves(
        trial_backend, time_col=elapsed_time_attr, metric_col=metric
    )

    scheduler = ASHA(
        config_space=trial_backend.blackbox.configuration_space,
        max_t=52,
        resource_attr="epoch",
        mode="max",
        metric=metric,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=600)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        tuner_name="ASHA-Yahpo",
    )
    tuner.run()

    tuning_experiment = load_experiment(tuner.name)
    tuning_experiment.plot()
