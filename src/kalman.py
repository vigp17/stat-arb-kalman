# src/kalman.py
import numpy as np
import pandas as pd

class KalmanFilterReg:
    def __init__(self, delta=1e-5, R=1e-3):
        """
        delta: The 'Process Noise'. Controls how flexible the line is.
               Small delta = Stiff line. Large delta = Wiggly line.
        R:     The 'Measurement Noise'. How much noise is in the daily price.
        """
        self.delta = delta
        self.R = R

    def run_filter(self, y_series, x_series):
        """
        Estimates the dynamic relationship: y = alpha + beta * x
        
        y: The Target Asset (e.g., MCD)
        x: The Reference Asset (e.g., YUM)
        """
        # Convert to numpy arrays for speed
        x = x_series.values
        y = y_series.values
        n = len(y)
        
        # --- INITIALIZATION ---
        # State Vector [alpha, beta]. We start assuming 0.
        state_mean = np.zeros(2) 
        
        # Covariance Matrix. Start with high uncertainty (ones).
        state_cov = np.ones((2, 2)) 
        
        # Identity matrix for calculations
        I = np.eye(2)
        
        # Storage for results
        results_alpha = []
        results_beta = []
        
        # --- THE LOOP (Update everyday) ---
        for t in range(n):
            # 1. PREDICTION STEP
            # Assume state (alpha, beta) is same as yesterday...
            pred_state_mean = state_mean
            
            # ...but uncertainty grows slightly (Q matrix adds 'process noise')
            Q = (self.delta / (1 - self.delta)) * I
            pred_state_cov = state_cov + Q

            # 2. OBSERVATION STEP
            # Create Observation matrix H = [1, x_t]
            H = np.array([[1.0, x[t]]])
            
            # What price do we expect MCD to be?
            y_pred = H @ pred_state_mean
            
            # 3. CORRECTION STEP (Innovation)
            # How wrong was our prediction?
            error = y[t] - y_pred
            
            # Kalman Gain (K): How much should we trust this new data?
            S = H @ pred_state_cov @ H.T + self.R
            K = pred_state_cov @ H.T @ np.linalg.inv(S)
            
            # Update State (alpha, beta) based on the error
            state_mean = pred_state_mean + K.flatten() * error
            
            # Update Covariance (reduce uncertainty)
            state_cov = (I - K @ H) @ pred_state_cov
            
            # Store estimates
            results_alpha.append(state_mean[0])
            results_beta.append(state_mean[1])
            
        # Return results as a DataFrame
        return pd.DataFrame({
            'alpha': results_alpha,
            'beta': results_beta
        }, index=x_series.index)