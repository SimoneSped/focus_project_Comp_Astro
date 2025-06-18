# Period Length

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def general_tests(prng_func, num_samples=100000, plot=False):
    """
    General PRNG test suite.
    prng_func: function with no arguments returning a random float in [0,1).
    """
    
    print("Running PRNG test suite...")
    samples = np.array([prng_func() for _ in range(num_samples)])
    
    # Test 1: Range check
    assert np.all(samples >= 0) and np.all(samples < 1), "Samples out of [0, 1) range"
    print("✅ Range test passed")
    
    # Test 2: Mean and Variance (expected mean = 0.5, variance = 1/12)
    mean = np.mean(samples)
    var = np.var(samples)
    print(f"Mean: {mean:.5f}, Variance: {var:.5f}")
    assert abs(mean - 0.5) < 0.01, "Mean significantly off"
    assert abs(var - 1/12) < 0.01, "Variance significantly off"
    print("✅ Moments test passed")
    
    # Test 3: Chi-squared uniformity test
    bins = 10
    observed, _ = np.histogram(samples, bins=bins, range=(0, 1))
    expected = np.full(bins, num_samples / bins)
    chi2, p = stats.chisquare(observed, expected)
    print(f"Chi-squared test p-value: {p:.5f}")
    assert p > 0.05, "Uniformity test failed"
    print("✅ Uniformity test passed")
    
    # Test 4: Autocorrelation (lag-1)
    autocorr = np.corrcoef(samples[:-1], samples[1:])[0,1]
    print(f"Lag-1 autocorrelation: {autocorr:.5f}")
    assert abs(autocorr) < 0.02, "Autocorrelation too strong"
    print("✅ Autocorrelation test passed")

    # Optional plot
    if plot:
        plt.hist(samples, bins=50, density=True)
        plt.title("Sample distribution")
        plt.show()
    
    print("All tests passed!")

