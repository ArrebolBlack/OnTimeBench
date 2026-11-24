"""Sample a training scenario and verify it in the simulator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ontimebench.scenarios._feasibility import _validate_physical_scenario
from ontimebench.scenarios._recipes import _materialize_recipe
from ontimebench.scenarios.spec import ScenarioSpec
from ontimebench.tasks.spec import TaskSpec


def generate_scenario(task: TaskSpec, recipe: Mapping[str, object], seed: int) -> ScenarioSpec:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    return _validate_physical_scenario(task, _materialize_recipe(task, recipe, rng))
