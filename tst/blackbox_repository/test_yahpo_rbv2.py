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
import pytest
import numpy as np

from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.conversion_scripts.scripts.yahpo_import import (
    BlackBoxYAHPO,
)


yahpo_rbv2_small_f1_instances = {
    "rpart": [
        4135,
        40927,
        40923,
        41163,
        4538,
        375,
        40496,
        40966,
        1457,
        458,
        469,
        11,
        40975,
        23,
        1468,
        40668,
        4541,
        40670,
        188,
        41164,
        1475,
        1476,
        1478,
        41169,
        41212,
        300,
        41168,
        41027,
        6,
        12,
        14,
        16,
        18,
        40979,
        22,
        1515,
        1493,
        28,
        32,
        182,
        40984,
        1501,
        40685,
        46,
        40982,
        377,
        40499,
        54,
        41166,
        307,
        1497,
        60,
    ],
    "glmnet": [
        41163,
        4538,
        375,
        40496,
        40966,
        1457,
        458,
        469,
        11,
        40975,
        23,
        1468,
        40668,
        4541,
        40670,
        188,
        41164,
        1475,
        1476,
        1478,
        41169,
        41212,
        300,
        41168,
        41027,
        6,
        12,
        14,
        16,
        18,
        40979,
        22,
        1515,
        41278,
        1493,
        28,
        32,
        182,
        40984,
        1501,
        40685,
        42,
        46,
        40982,
        377,
        40499,
        54,
        41216,
        41166,
        307,
        1497,
        60,
        40498,
        181,
        554,
    ],
    "ranger": [
        4538,
        375,
        40496,
        40966,
        1457,
        458,
        469,
        40975,
        23,
        1468,
        40670,
        188,
        41164,
        1475,
        1476,
        1478,
        41212,
        41027,
        6,
        12,
        14,
        16,
        18,
        40979,
        22,
        1515,
        28,
        32,
        182,
        40984,
        1501,
        40685,
        42,
        46,
        40982,
        377,
        40499,
        54,
        41216,
        307,
        1497,
        60,
        40498,
        181,
        41163,
        300,
        41165,
        41166,
        40927,
        41168,
        1493,
        41169,
    ],
    "xgboost": [
        41169,
    ],
    "svm": [
        12,
        16,
        14,
        22,
        1493,
        28,
        377,
        60,
        1501,
        300,
    ],
    "aknn": [
        40927,
        41163,
        40996,
        4538,
        375,
        40496,
        40966,
        1457,
        458,
        469,
        11,
        40975,
        23,
        1468,
        40670,
        41164,
        1475,
        1478,
        41169,
        300,
        41168,
        12,
        14,
        18,
        40979,
        22,
        1515,
        554,
        41278,
        1493,
        41165,
        40984,
        1501,
        42,
        46,
        40982,
        377,
        40499,
        54,
        41216,
        41166,
        307,
        1497,
    ],
}


@pytest.mark.skip("Needs blackbox data files locally or on S3.")
@pytest.mark.parametrize("method", yahpo_rbv2_small_f1_instances.keys())
def test_yahpo_rbv2_small_f1(method):
    random_seed = 31415927
    np.random.seed(random_seed)
    num_configs = 50
    small_f1_instances = [str(x) for x in sorted(yahpo_rbv2_small_f1_instances[method])]

    blackbox_name = f"yahpo-rbv2_{method}"
    yahpo_kwargs = dict(check=True)
    blackboxes = load_blackbox(blackbox_name, yahpo_kwargs=yahpo_kwargs)

    for instance in small_f1_instances:
        blackbox = blackboxes[instance]
        assert isinstance(blackbox, BlackBoxYAHPO)
        # Sample `num_configs` configs at random
        configs = [
            dict(xs.get_dictionary(), trainsize=1.0, repl=10)
            for xs in blackbox.benchmark.get_opt_space(
                drop_fidelity_params=True
            ).sample_configuration(num_configs)
        ]
        # f1 metric values
        f1_values = [
            result["f1"] for result in blackbox.benchmark.objective_function(configs)
        ]
        max_f1 = np.max(f1_values)
        print(f"rbv2_{method}_f1[{instance:5s}]: {max_f1:.3e}")
    assert False, "UUPS"


@pytest.mark.skip("Needs blackbox data files locally or on S3.")
@pytest.mark.parametrize("method", yahpo_rbv2_small_f1_instances.keys())
def test_yahpo_rbv2_filter_by_f1(method):
    random_seed = 31415927
    np.random.seed(random_seed)
    num_configs = 100

    blackbox_name = f"yahpo-rbv2_{method}"
    yahpo_kwargs = dict(check=True)
    blackboxes = load_blackbox(blackbox_name, yahpo_kwargs=yahpo_kwargs)
    blackbox = next(iter(blackboxes.values()))
    benchmark = blackbox.benchmark
    instances_ok = []
    f1vals_notok = []
    for instance in benchmark.instances:
        # Sample `num_configs` configs at random
        benchmark.set_instance(instance)
        configs = [
            dict(xs.get_dictionary(), trainsize=1.0, repl=10)
            for xs in benchmark.get_opt_space(
                drop_fidelity_params=True
            ).sample_configuration(num_configs)
        ]
        # f1 metric values
        f1_values = [result["f1"] for result in benchmark.objective_function(configs)]
        max_f1 = np.max(f1_values)
        if max_f1 > 0.2:
            instances_ok.append(instance)
        else:
            f1vals_notok.append(max_f1)
    parts = (
        [f'    "{method}": {{', f'        "f1": [']
        + [f'            "{x}",' for x in instances_ok]
        + ["        ],", "    },"]
    )
    print("\n".join(parts))
    print(f"\n Not OK: max={max(f1vals_notok)}, avg={np.mean(f1vals_notok)}")
    assert False, "UUPS"


yahpo_rbv2_auc_one_instances = {
    "xgboost": [
        16,
        40923,
        41143,
        470,
        1487,
        40499,
        40966,
        41164,
        1497,
        40975,
        1461,
        41278,
        11,
        54,
        300,
        40984,
        31,
        1067,
        1590,
        40983,
        41163,
        41165,
        182,
        1220,
        41159,
        41169,
        42,
        188,
        1457,
        1480,
        6332,
        181,
        1479,
        40670,
        40536,
        41138,
        41166,
        6,
        14,
        29,
        458,
        1056,
        1462,
        1494,
        40701,
        12,
        1493,
        44,
        307,
        334,
        40982,
        41142,
        38,
        1050,
        469,
        23381,
        41157,
        15,
        4541,
        23,
        4134,
        40927,
        40981,
        41156,
        3,
        1049,
        40900,
        1063,
        23512,
        40979,
        1040,
        1068,
        41161,
        22,
        1489,
        41027,
        24,
        4135,
        23517,
        1053,
        1468,
        312,
        377,
        1515,
        18,
        1476,
        1510,
        41162,
        28,
        375,
        1464,
        40685,
        40996,
        41146,
        41216,
        40668,
        41212,
        32,
        60,
        4538,
        40496,
        41150,
        37,
        46,
        554,
        1475,
        1485,
        1501,
        1111,
        4534,
        41168,
        151,
        4154,
        40978,
        40994,
        50,
        1478,
        1486,
        40498,
    ],
    "svm": [
        40981,
        4134,
        1220,
        40978,
        40966,
        40536,
        41156,
        458,
        41157,
        40975,
        40994,
        1468,
        6332,
        40670,
        151,
        1475,
        1476,
        1478,
        1479,
        41212,
        1480,
        1053,
        1067,
        1056,
        12,
        1487,
        1068,
        32,
        470,
        312,
        38,
        40982,
        50,
        41216,
        307,
        40498,
        181,
        1464,
        41164,
        16,
        1461,
        41162,
        6,
        14,
        1494,
        54,
        375,
        1590,
        23,
        41163,
        1111,
        41027,
        40668,
        41138,
        4135,
        4538,
        40496,
        4534,
        40900,
        1457,
        11,
        1462,
        41142,
        40701,
        29,
        37,
        23381,
        188,
        41143,
        1063,
        18,
        40979,
        22,
        1515,
        334,
        24,
        1493,
        28,
        1050,
        1049,
        40984,
        40685,
        42,
        44,
        46,
        1040,
        41146,
        377,
        40499,
        1497,
        60,
        40983,
        4154,
        469,
        31,
        41278,
        1489,
        1501,
        15,
        300,
        1485,
        1486,
        1510,
        182,
        41169,
    ],
}


@pytest.mark.skip("Needs blackbox data files locally or on S3.")
@pytest.mark.parametrize("method", yahpo_rbv2_auc_one_instances.keys())
def test_yahpo_rbv2_auc_one(method):
    random_seed = 31415927
    np.random.seed(random_seed)
    num_configs = 1000
    auc_one_instances = [str(x) for x in sorted(yahpo_rbv2_auc_one_instances[method])]

    blackbox_name = f"yahpo-rbv2_{method}"
    yahpo_kwargs = dict(check=True)
    blackboxes = load_blackbox(blackbox_name, yahpo_kwargs=yahpo_kwargs)

    for instance in auc_one_instances:
        blackbox = blackboxes[instance]
        assert isinstance(blackbox, BlackBoxYAHPO)
        # Sample `num_configs` configs at random
        configs = [
            dict(xs.get_dictionary(), trainsize=1.0, repl=10)
            for xs in blackbox.benchmark.get_opt_space(
                drop_fidelity_params=True
            ).sample_configuration(num_configs)
        ]
        # auc metric values
        auc_values = np.array(
            [result["auc"] for result in blackbox.benchmark.objective_function(configs)]
        )
        num_equal_one = np.sum(auc_values == 1)
        print(f"rbv2_{method}_auc[{instance:5s}]: {num_equal_one} of {num_configs}")
    assert False, "UUPS"


@pytest.mark.skip("Needs blackbox data files locally or on S3.")
@pytest.mark.parametrize("method", yahpo_rbv2_small_f1_instances.keys())
def test_yahpo_rbv2_filter_by_auc(method):
    random_seed = 31415927
    np.random.seed(random_seed)
    num_configs = 1000

    blackbox_name = f"yahpo-rbv2_{method}"
    yahpo_kwargs = dict(check=True)
    blackboxes = load_blackbox(blackbox_name, yahpo_kwargs=yahpo_kwargs)
    blackbox = next(iter(blackboxes.values()))
    benchmark = blackbox.benchmark
    instances_ok = []
    aucvals_notok = []
    for instance in benchmark.instances:
        # Sample `num_configs` configs at random
        benchmark.set_instance(instance)
        configs = [
            dict(xs.get_dictionary(), trainsize=1.0, repl=10)
            for xs in benchmark.get_opt_space(
                drop_fidelity_params=True
            ).sample_configuration(num_configs)
        ]
        # auc metric values
        auc_values = [result["auc"] for result in benchmark.objective_function(configs)]
        max_auc = np.max(auc_values)
        if max_auc < 0.999:
            instances_ok.append(instance)
        else:
            aucvals_notok.append(max_auc)
    parts = (
        [f'    "{method}": {{', f'        "auc": [']
        + [f'            "{x}",' for x in instances_ok]
        + ["        ],", "    },"]
    )
    print("\n".join(parts))
    print(f"\n Not OK: min = {min(aucvals_notok)}, avg = {np.mean(aucvals_notok)}")
    assert False, "UUPS"
