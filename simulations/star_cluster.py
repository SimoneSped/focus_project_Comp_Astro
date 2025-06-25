import numpy as np
from typing import Tuple

def sample_plummer_sphere(N: int, prng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample N positions from a Plummer sphere using a custom PRNG."""
    x, y, z = [], [], []

    for _ in range(N):
        # Generate radius using inverse transform sampling
        while True:
            u = prng() / prng.max_value
            if u > 0:
                break
        r = 1 / np.sqrt(u**(-2/3) - 1)

        # Generate uniform direction on sphere
        while True:
            u1 = 2 * (prng() / prng.max_value) - 1
            u2 = 2 * (prng() / prng.max_value) - 1
            s = u1**2 + u2**2
            if s < 1:
                break
        x_val = 2 * r * u1 * np.sqrt(1 - s)
        y_val = 2 * r * u2 * np.sqrt(1 - s)
        z_val = r * (1 - 2 * s)

        x.append(x_val)
        y.append(y_val)
        z.append(z_val)

    return np.array(x), np.array(y), np.array(z)
