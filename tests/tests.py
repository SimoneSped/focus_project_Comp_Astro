import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from collections import Counter

def general_tests(prng_func, num_samples=100000, plot=False):
    """
    General PRNG test suite.
    prng_func: function with no arguments returning a random float in [0,1).
    """
    samples = np.array([prng_func() for _ in range(num_samples)])
    
    # Test 1: Mean and Variance (expected mean = 0.5, variance = 1/12)
    mean = np.mean(samples)
    var = np.var(samples)
    print(f"Mean: {mean:.5f}, Variance: {var:.5f}")
    assert abs(mean - 0.5) < 0.01, "Mean significantly off"
    assert abs(var - 1/12) < 0.01, "Variance significantly off"
    print("Moments test passed")
    
    # Test 2: Chi-squared uniformity test
    bins = 10
    observed, _ = np.histogram(samples, bins=bins, range=(0, 1))
    expected = np.full(bins, num_samples / bins)
    chi2, p = stats.chisquare(observed, expected)
    print(f"Chi-squared test p-value: {p:.5f}")
    assert p > 0.05, "Uniformity test failed"
    print("Uniformity test passed")
    
    # Test 3: Autocorrelation (lag-1)
    autocorr = np.corrcoef(samples[:-1], samples[1:])[0,1]
    print(f"Lag-1 autocorrelation: {autocorr:.5f}")
    assert abs(autocorr) < 0.02, "Autocorrelation too strong"
    print("Autocorrelation test passed")

    # Optional plot
    if plot:
        plt.hist(samples, bins=50, density=True)
        plt.title("Sample distribution")
        plt.show()
    
    print("All tests passed!")

def estimate_period(prng_func, max_iter=10**6):
    """
    Estimate the period of a PRNG by checking for cycles.
    prng_func: function with no arguments returning a random float in [0,1).
    max_iter: maximum number of iterations to check for cycles
    """
    seen = set()
    # simply checks for repeated values
    for i in range(max_iter):
        x = prng_func()
        if x in seen:
            print(f"Cycle detected after {i} iterations")
            return i
        seen.add(x)
    print("No cycle detected within max iterations")
    return None

def birthday_spacings_test(prng_func, N=10000, bins=100):
    """
    Birthday spacings test.
    prng_func: function that returns one random float in [0, 1)
    N: number of samples
    bins: how many quantization bins to use for spacings (coarser binning makes repeated spacings easier to observe)
    """
    print("Running Birthday Spacings Test...")
    
    samples = np.array([prng_func() for _ in range(N)])
    samples.sort()

    # Compute spacings
    spacings = np.diff(samples)

    # Quantize spacings to discrete bins
    quantized_spacings = np.floor(spacings * bins)
    spacing_counts = Counter(quantized_spacings)
    
    # Count repeated spacings
    collisions = sum(count-1 for count in spacing_counts.values() if count > 1)
    total_unique_spacings = len(spacing_counts)
    
    print(f"Total unique spacings: {total_unique_spacings}")
    print(f"Total collisions: {collisions}")

    if collisions == 0:
        print("No repeated spacings detected (good).")
    elif collisions < 5:
        print("Few repeated spacings (still reasonable for N = {})".format(N))
    else:
        print("High number of collisions, possible non-randomness.")

