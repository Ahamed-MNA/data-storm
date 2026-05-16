import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
import logging

logger = logging.getLogger(__name__)

def sfa_log_likelihood(params, X, y):
    """
    Log-likelihood for a Stochastic Frontier Model with half-normal inefficiency.
    ε = y - Xβ
    σ² = σ_u² + σ_v²
    λ = σ_u / σ_v
    """
    k = X.shape[1]
    beta = params[:k]
    sigma_u = np.exp(params[k]) # Ensure positive
    sigma_v = np.exp(params[k+1]) # Ensure positive
    
    sigma_sq = sigma_u**2 + sigma_v**2
    sigma = np.sqrt(sigma_sq)
    lam = sigma_u / sigma_v
    
    epsilon = y - np.dot(X, beta)
    
    # Log-likelihood formula:
    # lnL = Σ [ln(2) - ln(σ) + ln(ϕ(ε/σ)) + ln(Φ(-ε*λ/σ))]
    # - Vectorized and extremely fast analytical logpdf: -0.5 * ln(2*pi) - 0.5 * z**2
    # - Vectorized and compiled compiled logcdf: log_ndtr
    term1 = np.log(2/sigma)
    z = epsilon / sigma
    term2 = -0.9189385332046727 - 0.5 * (z**2)
    term3 = log_ndtr(-z * lam)
    
    ll = np.sum(term1 + term2 + term3)
    return -ll # Minimize negative log-likelihood

class SFAModel:
    def __init__(self):
        self.beta = None
        self.sigma_u = None
        self.sigma_v = None
        self.feature_names = None

    def fit(self, X, y, verbose=True):
        self.feature_names = X.columns.tolist()
        X_val = X.values
        y_val = y.values
        
        k = X_val.shape[1]
        # Initial guess: OLS for beta, small values for sigmas
        initial_beta = np.linalg.lstsq(X_val, y_val, rcond=None)[0]
        initial_params = np.concatenate([initial_beta, [0.1, 0.1]]) # log(0.1) ~ -2.3
        
        logger.info(f"Starting SFA optimization for {k} features...")
        
        iteration = 0
        def callback(xk):
            nonlocal iteration
            iteration += 1
            if verbose:
                ll_val = sfa_log_likelihood(xk, X_val, y_val)
                logger.info(f"Iteration {iteration:03d}: Neg Log-Likelihood = {ll_val:.4f}")
        
        res = minimize(sfa_log_likelihood, initial_params, args=(X_val, y_val), method='L-BFGS-B', callback=callback)
        
        if res.success:
            self.beta = res.x[:k]
            self.sigma_u = np.exp(res.x[k])
            self.sigma_v = np.exp(res.x[k+1])
            logger.info("SFA Model fitted successfully.")
        else:
            logger.error(f"SFA Optimization failed: {res.message}")
            raise Exception("Model fitting failed.")

    def predict_potential(self, X):
        """
        Predicts potential (frontier) volume.
        Y_potential = exp(Xβ) * exp(0.5 * σ_v²)
        """
        X_val = X.values
        ln_y_potential = np.dot(X_val, self.beta)
        # Bias adjustment for log-normal distribution
        y_potential = np.exp(ln_y_potential) * np.exp(0.5 * self.sigma_v**2)
        return y_potential

    def get_efficiency(self, X, y_actual):
        """
        Estimates technical efficiency TE = exp(-u)
        Jondrow et al. (1982) estimator for E[u|ε] can be complex, 
        but TE = actual / potential is a simpler proxy.
        """
        potential = self.predict_potential(X)
        efficiency = y_actual / potential
        return efficiency

    def save(self, filepath):
        """
        Saves the SFA model to a file using pickle.
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Loads the SFA model from a file using pickle.
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {filepath}")
        return model

