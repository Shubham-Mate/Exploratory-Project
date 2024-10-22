from sklearn.neighbors import KernelDensity
import numpy as np

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
    N = samples.shape[0]
    d = samples.shape[1]
    

    # Fit KDE to the samples to estimate Q
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(samples)
    
    # Compute the KL divergence using Monte Carlo approximation
    kl_div = 0.0
    
    for sample in samples:
        # Log probabilities for the true distribution P
        log_p_theta = target_distribution.logpdf(sample)
        
        # Log probabilities for the empirical distribution Q (estimated via KDE)
        log_q_theta = kde.score_samples(sample.reshape(1, -1))
        
        kl_div += (log_p_theta - log_q_theta)
    
    kl_div /= N
    
    return kl_div