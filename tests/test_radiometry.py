"""Toy radiometric check: L**gamma / t recovers relative radiance below clipping;
clipped samples cannot be recovered by inverting gamma and exposure."""
import numpy as np

GAMMA = 2.2


def simulate(radiance, t):
    return np.clip(radiance * t, 0, 1) ** (1 / GAMMA)


def test_recovers_below_clip_and_fails_above():
    rad = np.linspace(0.001, 0.5, 500)  # relative radiance ramp
    for t in (1.0, 4.0, 16.0):
        l = simulate(rad, t)
        rec = l ** GAMMA / t
        below = rad * t < 1
        assert np.allclose(rec[below], rad[below], rtol=1e-6)
        if (~below).any():
            # every clipped sample maps to the same value 1/t, whatever the true radiance
            assert np.allclose(rec[~below], 1 / t)
            assert np.max(np.abs(rec[~below] - rad[~below])) > 0.1 * rad[~below].max()
    # the three exposures agree below clipping
    l1, l3 = simulate(rad, 1.0), simulate(rad, 16.0)
    ok = rad * 16 < 1
    assert np.allclose(l1[ok] ** GAMMA / 1.0, l3[ok] ** GAMMA / 16.0, rtol=1e-6)
