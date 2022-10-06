import numpy as np
import scipy.stats
import copy
import os
import pickle
from smoothers import *
from ensemble_smoothers import *
from transport_smoothers import *

# Find the current path
root_directory = os.path.dirname(os.path.realpath(__file__))

#%%

# =============================================================================
# Define auto-regressive model, prior, and likelihood
# =============================================================================

# Set a fixed random seed
np.random.seed(0)

# Problem setup ---------------------------------------------------------------

# Define the problem's dimensions
D       = 1     # Number of state dimensions
O       = 1     # Number of observation dimensions

# Ensemble sizes we want to try
Ns      = [100,1000]

# Number of repeat simulations
repeats = 1000

# Get the number of total time steps
T       = 30

# Define map order
order   = 1

# Dynamical model -------------------------------------------------------------

# Forecast operator
A       = np.identity(D)*0.9

# Forecast error covariance matrix
sigma_x = 1.0
Q       = np.identity(D)*sigma_x

# Get the observation operator
H       = np.zeros((O,D))
np.fill_diagonal(H,1)   # We observe all states

# Get the observation error
R       = np.zeros((O,O))
sigma_y = 1.0
np.fill_diagonal(R,sigma_y)

# set the initial prior variance based on model
a              = (sigma_y**2*(1 - A**2) + sigma_x**2)
post_c         = (-a + np.sqrt(a**2 + 4*A**2*sigma_y**2*sigma_x**2))/(2*A**2)
C0             = A**2*post_c + sigma_x**2

# Get the prior mean and cov
mu_a    = np.zeros((D,1))        # Initial true mean
cov_a   = np.identity(D)*C0      # Initial true covariance

# assign model
model = {}
model['A'] = A
model['H'] = H
model['Q'] = Q
model['R'] = R
model['T'] = T
model['mu_a'] = mu_a
model['cov_a'] = cov_a

# Run algorithms  -----------------------------------------------------------------

# Initialize the output dictionary
dct         = {} 

# Repeat this procedure 
for rep in range(repeats):
    
    # Print an update
    print('random seed '+str(rep))
        
    # Create a new key for this ensemble size
    dct[rep]    = {}
    
    # =========================================================================
    # Reference solution
    # =========================================================================
    
    # Get the true dynamics
    x_true          = np.zeros((int(T*D),1))
    x_true[:D,0]    = scipy.stats.norm.rvs(loc=mu_a,scale=np.sqrt(cov_a),size=1)
    for t in np.arange(1,T,1):
        # update state
        x_true[int(t*D):int((t+1)*D),0] = np.dot(
            A,
            x_true[int((t-1)*D):int((t)*D),:]) 
        # sample noise
        noise = scipy.stats.multivariate_normal.rvs(
                mean    = np.zeros(D),
                cov     = Q,
                size    = 1)
        # Add the noise
        x_true[int(t*D):int((t+1)*D),0] += noise

    # Create observations by perturbing the true state
    x_obs       = copy.copy(x_true) 
    x_obs[:,0]  += scipy.stats.norm.rvs(loc=0,scale=np.sqrt(R[0,0]),size=T)
    model['x_obs'] = x_obs    
    model['x_true'] = x_true
    dct[rep]['truth'] = {'x': x_obs, 'y': x_true}

    # =========================================================================
    # Kalman Smoother
    # =========================================================================
   
    (muT_KS, covT_KS) = KalmanSmoother(model, T)
 
    # Store the analytic reference solution
    dct[rep]['KS']      = {
        'mean'  : muT_KS,
        'cov'   : covT_KS}
    
    # =============================================================================
    # Rauch-Tung-Striebel Smoother
    # =============================================================================
   
    (muT_RTS, covT_RTS) = RTSSmoother(model, T)
                    
    # Store the analytic reference solution
    dct[rep]['RTSS']    = {
        'mean'  : muT_RTS,
        'cov'   : covT_RTS}
    
    # =============================================================================
    # Ensemble Smoothers
    # =============================================================================
    
    for N in Ns:
    
        # Create a new key for this seed
        dct[rep][N] = {}
        
        # Draw prior samples
        X   = scipy.stats.multivariate_normal.rvs(
            mean    = mu_a,
            cov     = cov_a,
            size    = N)
        
        # ====================================================================
        # EnKS (empirical)
        # =====================================================================
        
        log_EnKS, gain_EnKS, signal_EnKS = EnsembleKS(model, T, X) 
        
        # Store the results for the EnKS
        dct[rep][N]['EnKS']     = {
            'mean'  : np.mean(log_EnKS,axis=0),
            'cov'   : np.cov(log_EnKS.T),
            'map'   : gain_EnKS,
            'signal': signal_EnKS}
        
        # =============================================================================
        # single-pass EnRTSS (empirical)
        # =============================================================================
        
        log_EnRTSS, gain_EnRTSS, signal_EnRTSS  = SinglePassEnsembleRTS(model, T, X)
        
        # Store the results for the EnKS
        dct[rep][N]['EnRTSS (single-pass)']     = {
            'mean'  : np.mean(log_EnRTSS,axis=0),
            'cov'   : np.cov(log_EnRTSS.T),
            'map'   : gain_EnRTSS,
            'signal': signal_EnRTSS}

        # =============================================================================
        # multi-pass EnRTSS (empirical)
        # =============================================================================
        
        log_EnRTSS, gain_EnRTSS, signal_EnRTSS  = MultiPassEnsembleRTS(model, T, X)
    
        # Store the results for the EnKS
        dct[rep][N]['EnRTSS (multi-pass)']     = {
            'mean'  : np.mean(log_EnRTSS,axis=0),
            'cov'   : np.cov(log_EnRTSS.T),
            'map'   : gain_EnRTSS,
            'signal': signal_EnRTSS}
                
        # =============================================================================
        # multi-pass FIT (empirical)
        # =============================================================================
        
        log_FIT, gainK_FIT, gainC_FIT, signalK_FIT, signalC_FIT = MultiPassEnsembleFIT(model, T, X)

        # Store the results for the EnKS
        dct[rep][N]['EnFIT (multi-pass)']     = {
            'mean'  : np.mean(log_FIT,axis=0),
            'cov'   : np.cov(log_FIT.T),
            'map'   : np.stack([gainK_FIT, gainC_FIT],axis=3),
            'signal': np.stack([signalK_FIT, signalC_FIT],axis=3)}
                
        # ====================================================================
        # EnTS (empirical)
        # =====================================================================
        
        log_EnTS = EnsembleTransportSmoother(model, T, X, order)
        
        # Store the results for the EnKS
        dct[rep][N]['EnTS (joint-analysis)']     = {
            'mean'  : np.mean(log_EnTS,axis=0),
            'cov'   : np.cov(log_EnTS.T)}
        
        # =============================================================================
        # single-pass BIT EnTS (empirical)
        # =============================================================================
        
        log_EnTS, map_EnTS, signal_EnTS = SinglePassEnsembleTransportBIT(model, T, X, order)
        
        # Store the results for the EnKS
        dct[rep][N]['EnTS (backward, single-pass)']     = {
            'mean'  : np.mean(log_EnTS,axis=0),
            'cov'   : np.cov(log_EnTS.T),
            'map'   : map_EnTS,
            'signal': signal_EnTS}
    
        # =============================================================================
        # multi-pass BIT EnTS (empirical)
        # =============================================================================
        
        log_EnTS, map_EnTS, signal_EnTS = MultiPassEnsembleTransportBIT(model, T, X, order)
    
        # Store the results for the EnKS
        dct[rep][N]['EnTS (backward, multi-pass)']     = {
            'mean'  : np.mean(log_EnTS,axis=0),
            'cov'   : np.cov(log_EnTS.T),
            'map'   : map_EnTS,
            'signal': signal_EnTS}
                
        # =============================================================================
        # multi-pass FIT (empirical)
        # =============================================================================
        
        log_FIT    = MultiPassEnsembleTransportFIT(model, T, X, order)
        
        # Store the results for the EnKS
        dct[rep][N]['EnTS (forward, multi-pass)'] = {
            'mean'  : np.mean(log_FIT,axis=0),
            'cov'   : np.cov(log_FIT.T)}
#%%

# Store the results
pickle.dump(dct,open('autoregressive_model_results.p','wb'))
        
