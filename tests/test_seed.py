from aerial_geoloc.seed import seed_everything
import random

import numpy as np
import torch


def test_seed_repeatable_rand():
    seed_everything(123, deterministic=True)
    a = random.random()
    b = np.random.rand()
    c = torch.rand(3)
    seed_everything(123, deterministic=True)
    assert a == random.random()
    assert b == np.random.rand()
    assert torch.equal(c, torch.rand(3))
