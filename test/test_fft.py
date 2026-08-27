import pytest


def test_plan_cache_entry_limit_is_raised():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no GPU available")
    except Exception:
        pytest.skip("no usable CUDA/HIP runtime")

    from abtem.core import config
    from abtem.core import fft as abtem_fft

    cache = cp.fft.config.get_plan_cache()

    # CuPy's own default of 16 thrashes for varying batch shapes.
    abtem_fft._CUFFT_CACHE_STATE = None
    abtem_fft._configure_cufft_cache()
    assert cache.get_size() == 64

    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-entries": 128}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_size() == 128

    # Disabling the cache still wins over the entry count.
    abtem_fft._CUFFT_CACHE_STATE = None
    with config.set({"cupy.fft-cache-size": "0 MB"}):
        abtem_fft._configure_cufft_cache()
        assert cache.get_size() == 0

    abtem_fft._CUFFT_CACHE_STATE = None
    abtem_fft._configure_cufft_cache()
