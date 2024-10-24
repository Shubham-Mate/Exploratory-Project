from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def generate_random_covariance_matrix(n: int, low: int, high: int):
    '''
    Produces a symmetric and positive semi-definite matrix which can be used as a covariance matrix
    '''
    a = np.random.uniform(low=low, high=high, size=(n, n))
    return (a @ a.T) / high

def kl_divergence(samples, target_distribution):
    """
    Compute the KL divergence between a true Gaussian distribution P and an empirical distribution Q using samples.
    
    Parameters:
    - samples: Samples from the empirical distribution Q.
    - target_distribution: The true distribution P
    
    Returns:
    - KL divergence estimate.
    """
    
    # Fit KDE to the samples to estimate Q
    kde = gaussian_kde(samples, bw_method='scott')
    
    # Log probabilities for the true distribution P
    p_vals = target_distribution.pdf(samples)
    
    # Log probabilities for the empirical distribution Q (estimated via KDE)
    q_vals = kde(samples)
    q_vals = np.maximum(q_vals, 1e-10)
    
    # Avoid division by zero in log computation
    log_term = (1/2)*np.square(np.log(p_vals / q_vals))
    
    # Monte Carlo approximation of KL divergence
    kl_divergence = np.mean(log_term)

    return kl_divergence

def plot_sampled_distribution(ax, target, samples, mean, std_dev, title='PDF Estimations'):
    kde = gaussian_kde(samples)
    x = np.linspace(mean-1.5*std_dev, mean+1.5*std_dev, 100)

    ax.plot(x, target.pdf(x), label='Original Distribution')
    ax.plot(x, kde(x), linestyle='--', label='Approximation by Sampling')

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('PDF')
    ax.legend()
    