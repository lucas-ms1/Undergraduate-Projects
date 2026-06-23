"""Quick test for utils.markov – run with: python test_markov.py"""
from utils.markov import simulate_shock_path
import numpy as np

P = np.array([[0.9, 0.1],
              [0.1, 0.9]])

path = simulate_shock_path(P, T=5, initial_idx=0, seed=42)
print("Shock path:", path)
