import numpy as np
from hmmlearn.hmm import GaussianHMM
import pickle

def train_hmm(features, n_components=3):
    hmm = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100000000, verbose=False, init_params='mc')
    hmm.transmat_ = np.array([[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]])
    hmm.startprob_ = np.array([1/3, 1/3, 1/3])
    hmm.fit(features)
    return hmm

def predict_hmm_states(hmm, features):
    return hmm.predict(features)

def save_hmm_model(hmm, filepath):
    """Save the trained HMM model to a file using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(hmm, f)
    print(f"HMM model saved to {filepath}")

def load_hmm_model(filepath):
    """Load a trained HMM model from a file using pickle."""
    with open(filepath, 'rb') as f:
        hmm = pickle.load(f)
    print(f"HMM model loaded from {filepath}")
    return hmm