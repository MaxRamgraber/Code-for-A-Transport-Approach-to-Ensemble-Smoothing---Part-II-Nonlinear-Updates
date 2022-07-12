import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats
import copy
import os
from transport_map_138 import *
import pickle


def stochastic_EnKF(X,y,R,H):
    
    """
    This function implements a stochastic EnKF update. It requires the follow-
    ing variables:
        
        X       - an N-by-D array of samples, where N is the ensemble size, and
                  D is the dimensionality of state space
        y       - a vector of length O containing the observations made
        R       - the O-by-O observation error covariance matrix
        H       - an O-by-D observation operator
    """
    
    # Get the number of particles
    N       = X.shape[0]
    
    # Get the state covariance matrix
    C       = np.cov(X.T)   # We need the transpose of X
    if C.shape != (X.shape[1],X.shape[1]):
        C   = np.asarray([C]).reshape((X.shape[1],X.shape[1]))
        
    
    # Calculate the Kalman gain
    K       = np.linalg.multi_dot((
        C,
        H.T,
        np.linalg.inv(
            np.linalg.multi_dot((
                H,
                C,
                H.T)) + R)))
    
    # print(K.shape)
    
    # Draw observation error realizations
    v       = scipy.stats.multivariate_normal.rvs(
        mean    = np.zeros(R.shape[0]),
        cov     = R,
        size    = N)
    
    # print(v.shape)
    
    # Perturb the observations
    obs     = y[np.newaxis,:] + v   
    obs     = obs.T
    
    # Apply the stochastic Kalman update
    for n in range(N):
        
        X[n,:]  += np.dot(
            K,
            obs[n,:][:,np.newaxis] - np.dot(H,X[n,:][:,np.newaxis] ) )[:,0]
        
    return X

# Find the current path
root_directory = os.path.dirname(os.path.realpath(__file__))

# # Clear previous figures
# root_directory = os.path.dirname(os.path.realpath(__file__))
# results = []
# results += [each for each in os.listdir(root_directory) if each.endswith('.png') or each.endswith('.p')]
# for fl in range(len(results)):
#     os.remove(root_directory+'\\'+results[fl])

plt.close('all')

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

# Map orders (1 is linear, 3 is cubic)
orders  = [1,3]
order   = 1

# Dynamical model -------------------------------------------------------------

# Forecast operator
A       = np.identity(D)*0.9

# Forecast error covariance matrix
Q       = np.identity(D)

# Get the observation operator
H       = np.zeros((O,D))
np.fill_diagonal(H,1)   # We observe all states

# Get the observation error
R       = np.zeros((O,O))
np.fill_diagonal(R,4)

# Get the true dynamics
x_true          = np.zeros((int(T*D),1))
x_true[:D,0]    = 10
for t in np.arange(1,T,1):
    x_true[int(t*D):int((t+1)*D),0] = np.dot(
        A,
        x_true[int((t-1)*D):int((t)*D),:]) 
    
# Synthetic reference solution setup ------------------------------------------

# Get the prior mean and cov
mu_a    = np.ones((D,1))*10     # Initial true mean
cov_a   = np.identity(D)*4      # Initial true covariance

# Prior setup -----------------------------------------------------------------

# Set the prior to the true initial moments
mu_pri  = copy.copy(mu_a)
cov_pri = copy.copy(cov_a)

# Set up muT and covT
# Pre-allocate mean vector and covariance matrix for the analytical reference
muT     = np.zeros((int(T*D),1))
covT    = np.zeros((int(T*D),int(T*D)))

# Fill in the prior
muT[:D,0]   = copy.copy(mu_a)
covT[:D,:D] = copy.copy(cov_a)

 
#%%

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
    
    # Create observations by perturbing the true state
    x_obs       = copy.copy(x_true) 
    x_obs[:,0]  += scipy.stats.norm.rvs(loc=0,scale=np.sqrt(R[0,0]),size=T)
    
    # =========================================================================
    # Kalman Smoother
    # =========================================================================
    
    muT_KS  = copy.copy(muT)
    
    covT_KS = copy.copy(covT)
    
    muT_KS_intermediate     = []
    covT_KS_intermediate    = []
    
    for t in np.arange(0,T,1):
        
        # Find the appropriate indices
        ibgn    = int((t)*D)
        iend    = int((t+1)*D)
        
        # Find the time indices before
        ibgn_bf = int((t-1)*D)
        iend_bf = int((t)*D)
        
        if t > 0:
        
            # -------------------------------------------------------------------------
            # Stochastic forecast
            # -------------------------------------------------------------------------
            
            # Update the mean
            muT_KS[ibgn:iend,:]            = np.dot(
                A,
                copy.copy(muT_KS[ibgn_bf:iend_bf,:]))
                
            # Update the covariance
            covT_KS[ibgn:iend,ibgn:iend]   = np.linalg.multi_dot((
                A,
                copy.copy(covT_KS[ibgn_bf:iend_bf,ibgn_bf:iend_bf]),
                A.T)) + Q
            
            # Go through all off-diagonal entries
            for s in range(t):
                
                # Update cov_hor
                covT_KS[int((t)*D):int((t+1)*D),int((s)*D):int((s+1)*D)] = \
                    np.dot(
                        A,
                        copy.copy(
                            covT_KS[int((t-1)*D):int((t)*D),int((s)*D):int((s+1)*D)]))
                
                # Update cov_vert
                covT_KS[int((s)*D):int((s+1)*D),int((t)*D):int((t+1)*D)] = \
                    np.dot(
                        copy.copy(
                            covT_KS[int((s)*D):int((s+1)*D),int((t-1)*D):int((t)*D)]),
                        A.T)
                    
        # Save intermediate muT before update
        muT_KS_intermediate.append(copy.copy(muT_KS[:iend,:]))
        covT_KS_intermediate.append(copy.copy(covT_KS[:iend,:iend]))
                    
        # -------------------------------------------------------------------------
        # Extend covariance with observation predictions
        # -------------------------------------------------------------------------
        
        # Create an augmented state vector with observations in the last spot
        mu  = np.row_stack((
            copy.copy(muT_KS[:iend]),
            np.dot(H,copy.copy(muT_KS[ibgn:iend,:])) ))
            
        # Fill in the augmented covariance matrix
        cov = np.zeros((int((t+1)*D)+O,int((t+1)*D)+O))
        
        # UPPER DIAGONAL: States
        #   | X X   |
        #   | X X   |
        #   |       |
        cov[:int((t+1)*D),:int((t+1)*D)] = copy.copy(
            covT_KS[:iend,:iend])
        
        # LOWER DIAGONAL: Observations
        #   |       |
        #   |       |
        #   |     X |
        cov[int((t+1)*D):,int((t+1)*D):] = np.linalg.multi_dot((
            H,
            copy.copy(covT_KS[ibgn:iend,ibgn:iend]),
            H.T)) + R
    
        # UPPER TRIANGULAR: Covariances between states and observations
        #   |     X |
        #   |     X |
        #   |       |
        for s in range(t+1):
            cov[int((s)*D):int((s+1)*D),int((t+1)*D):] = np.dot(
                copy.copy(cov[int((s)*D):int((s+1)*D),int((t)*D):int((t+1)*D)]),
                H.T)
    
        # LOWER TRIANGULAR: Covariances between observations and states
        #   |       |
        #   |       |
        #   | X X   |
        for s in range(t+1):
            cov[int((t+1)*D):,int((s)*D):int((s+1)*D)] = np.dot(
                H,
                copy.copy(cov[int((t)*D):int((t+1)*D),int((s)*D):int((s+1)*D)]) )
        
        # -------------------------------------------------------------------------
        # Condition full distribution on observations
        # -------------------------------------------------------------------------
        
        # Calculate the conditioned mean
        mu_cond     = mu[:-O,:] + np.linalg.multi_dot((
            cov[:-O,-O:],
            np.linalg.inv(cov[-O:,-O:]),
            x_obs[t,:] - mu[-O:,:]))
        
        # Calculate the conditioned covariance
        cov_cond    = cov[:-O,:-O] - np.linalg.multi_dot((
            cov[:-O,-O:],
            np.linalg.inv(cov[-O:,-O:]),
            cov[-O:,:-O]))
        
        # Plug them into the mean and covariance arrays
        muT_KS[:iend,:]         = copy.copy(mu_cond)
        covT_KS[:iend,:iend]    = copy.copy(cov_cond)
    
    # Store the analytic reference solution
    dct[rep]['KS']      = {
        'mean'  : muT_KS,
        'cov'   : covT_KS}
    
    # Delete the variables
    del muT_KS, covT_KS, mu_cond, cov_cond, mu, cov, muT_KS_intermediate, covT_KS_intermediate
    
    #%%
    
    # =============================================================================
    # Rauch-Tung-Striebel Smoother
    # =============================================================================
    
    muT_RTS     = [muT[0,:].reshape((D,1))]
    covT_RTS    = [covT[:D,:D].reshape((D,1))]
    
    muT_RTS_intermediate    = []
    covT_RTS_intermediate   = []
    
    for t in np.arange(0,T,1):
        
        # Find the appropriate indices
        ibgn    = int((t)*D)
        iend    = int((t+1)*D)
        
        # Find the time indices before
        ibgn_bf = int((t-1)*D)
        iend_bf = int((t)*D)
        
        if t > 0:
        
            # ---------------------------------------------------------------------
            # Stochastic forecast (marginal)
            # ---------------------------------------------------------------------
            
            # Forecast the mean
            muT_RTS.append(
                np.dot(
                    A,
                    copy.copy(muT_RTS[-1])))
            
            # Forecast the covariance
            covT_RTS.append(
                np.linalg.multi_dot((
                    A,
                    copy.copy(covT_RTS[-1]),
                    A.T)) + Q)
                    
        # Save intermediate muT before update, required for RST update
        muT_RTS_intermediate.append(copy.deepcopy(muT_RTS))
        covT_RTS_intermediate.append(copy.deepcopy(covT_RTS))
                    
        # -------------------------------------------------------------------------
        # Kalman Filter update (marginal)
        # -------------------------------------------------------------------------
        
        # Create an augmented state vector with observations in the last spot
        mu  = np.row_stack((
            copy.copy(muT_RTS[-1]),
            np.dot(H,copy.copy(muT_RTS[-1])) ))
            
        # Fill in the augmented covariance matrix
        cov = np.zeros((O+D,O+D))
        
        # UPPER DIAGONAL: States
        #   | X   |
        #   |     |
        cov[:D,:D] = copy.copy(covT_RTS[-1])
        
        # LOWER DIAGONAL: Observations
        #   |     |
        #   |   X |
        cov[D:,D:] = np.linalg.multi_dot((
            H,
            copy.copy(covT_RTS[-1]),
            H.T)) + R
    
        # UPPER TRIANGULAR: Covariances between states and observations
        #   |   X |
        #   |     |
        cov[:D,O:] = np.dot(
            copy.copy(cov[:D,:D]),
            H.T)
    
        # LOWER TRIANGULAR: Covariances between observations and states
        #   |     |
        #   | X   |
        cov[D:,:O] = np.dot(
            H,
            copy.copy(cov[:D,:D]) )
    
        # -------------------------------------------------------------------------
        # Condition full distribution on observations
        # -------------------------------------------------------------------------
        
        # Calculate the conditioned mean
        mu_cond     = mu[:-O,:] + np.linalg.multi_dot((
            cov[:-O,-O:],
            np.linalg.inv(cov[-O:,-O:]),
            x_obs[t,:] - mu[-O:,:]))
        
        # Calculate the conditioned covariance
        cov_cond    = cov[:-O,:-O] - np.linalg.multi_dot((
            cov[:-O,-O:],
            np.linalg.inv(cov[-O:,-O:]),
            cov[-O:,:-O]))
        
        # Plug them into the mean and covariance arrays
        muT_RTS[-1]     = copy.copy(mu_cond)
        covT_RTS[-1]    = copy.copy(cov_cond)
        
        # -------------------------------------------------------------------------
        # At the end, start a backwards filtering pass
        # -------------------------------------------------------------------------
        
        if t == T-1:
            
            # Move backwards through time
            for s in np.arange(t-1,-1,-1):
                
                # Get the update array
                # This is the semi-analytical RTS gain; see Raanes paper
                C   = np.linalg.multi_dot((
                    copy.copy(covT_RTS_intermediate[s+1][s]),
                    A.T,
                    np.linalg.inv(copy.copy(covT_RTS_intermediate[s+1][s+1]))))

                # Update the mean
                muT_RTS[s]  += \
                    np.dot(
                        C,
                        muT_RTS[s+1] - muT_RTS_intermediate[s+1][s+1])
                    
                # Update the covariance
                # This equation is straight from https://en.wikipedia.org/wiki/Kalman_filter#Rauch%E2%80%93Tung%E2%80%93Striebel
                covT_RTS[s] += \
                    np.linalg.multi_dot((
                        C,
                        copy.copy(covT_RTS[s+1]) - copy.copy(covT_RTS_intermediate[s+1][s+1]),
                        C.T))
                    
    # Store the analytic reference solution
    dct[rep]['RTSS']    = {
        'mean'  : muT_RTS,
        'cov'   : covT_RTS}
    
    # Delete the variables
    del muT_RTS, covT_RTS, mu_cond, cov_cond, C, muT_RTS_intermediate, covT_RTS_intermediate
    
    #%%
    
    for N in Ns:
    
        # Define a projector which extracts the anomalies of an ensemble matrix
        projector   = np.identity(N) - np.ones((N,N))/N


        # Create a new key for this seed
        dct[rep][N] = {}
        
        #%%
        
        # ====================================================================
        # EnKF (empirical)
        # =====================================================================
        
        # Reset the random seed
        np.random.seed(rep)
        
        # Draw prior samples
        X   = scipy.stats.multivariate_normal.rvs(
            mean    = mu_pri,
            cov     = cov_pri,
            size    = N)
        
        # Turn it into a 2D arary if one-dimensional
        if D == 1: X = X[:,np.newaxis]
        
        # Create a storage log for the samples
        log_EnKF        = np.zeros((N,int(T*D)))
        log_EnKF[:,:D]  = copy.copy(X)
        
        # Create a log for the forecasts and observation predictions
        log_EnKF_f      = copy.copy(log_EnKF)
        log_EnKF_Y      = copy.copy(log_EnKF)
        
        # Work through all time steps
        for t in np.arange(0,T,1):
            
            # Sample stochastic noise
            noise = \
                scipy.stats.multivariate_normal.rvs(
                    mean    = np.zeros(O),
                    cov     = R,
                    size    = N)
            if O == 1: noise = noise[np.newaxis,:]
             
            # Create observation predictions by perturbing the forecast
            ysim    = np.dot(H,copy.copy(log_EnKF_f[:,int((t)*D):int((t+1)*D)]).T)
            ysim    += noise
            
            # Store the result
            log_EnKF_Y[:,int((t)*D):int((t+1)*D)] = copy.copy(ysim.T)
            
            # Make a sample-based estimate of the prediction covariance
            cov_y       = np.cov(ysim)
            if cov_y.shape != (O,O):
                cov_y   = cov_y.reshape((O,O))
            
            # Get its precision matrix
            cov_y_inv   = np.linalg.inv(cov_y)
            
            # Calculate the cross-covariance
            cov_xy  = np.dot(
                (log_EnKF_f[:,int((t)*D):int((t+1)*D)] - np.mean(log_EnKF_f[:,int((t)*D):int((t+1)*D)],axis=0)[np.newaxis,:]).T,
                (ysim - np.mean(ysim,axis=-1)[np.newaxis,:]).T
                ) / (N - 1)
            
            # Apply the Kalman update
            log_EnKF[:,int((t)*D):int((t+1)*D)] -= \
                np.linalg.multi_dot((
                    cov_xy,
                    cov_y_inv,
                    (ysim - x_obs[t,...][np.newaxis,:]) )).T
            
            # Make a forecast to the next time step
            if t < T-1:
                
                # Deterministic forecast
                log_EnKF[:,int((t+1)*D):int((t+2)*D)] = np.dot(
                    A,
                    log_EnKF[:,int((t)*D):int((t+1)*D)].T).T
        
                # Sample stochastic noise
                noise = \
                    scipy.stats.multivariate_normal.rvs(
                        mean    = np.zeros(O),
                        cov     = Q,
                        size    = N)
                if D == 1: noise = noise[:,np.newaxis]
        
                # Add the noise
                log_EnKF[:,int((t+1)*D):int((t+2)*D)] += noise
                
                log_EnKF_f[:,int((t+1)*D):int((t+2)*D)] = copy.copy(
                    log_EnKF[:,int((t+1)*D):int((t+2)*D)])
                
        # Store the results for the EnKF
        dct[rep][N]['EnKF']     = {
            'mean'  : np.mean(log_EnKF,axis=0),
            'cov'   : np.cov(log_EnKF.T)}
        
        #%%
        
        # =====================================================================
        # EnKS (empirical)
        # =====================================================================
        
        # Reset the random seed
        np.random.seed(rep)
        
        # Copy the EnKF array
        log_EnKS        =  copy.copy(log_EnKF)

        # Move through all time steps
        for t in np.arange(0,T,1):
            
            # Store the map and signal for the last update pass
            if t == T-1:
                
                # Store the map
                map_EnKS    = np.zeros((T-1,1))*np.nan
                
                # Store the signal
                signal_EnKS = np.zeros((T-1,N))
            
            # The EnKS is based on a multi-pass procedure; go through all time
            # steps so far
            for s in range(t):
            
                # Recover the observation predictions
                ysim    = log_EnKF_Y[:,int((t)*D):int((t+1)*D)].T
        
                # Make a sample-based estimate of the prediction covariance
                cov_y       = np.cov(ysim)
                if cov_y.shape != (O,O):
                    cov_y   = cov_y.reshape((O,O))
                
                # Get its precision matrix
                cov_y_inv   = np.linalg.inv(cov_y)

                # Calculate the cross-covariance
                cov_xy  = np.dot(
                    (log_EnKS[:,int((s)*D):int((s+1)*D)] - np.mean(log_EnKS[:,int((s)*D):int((s+1)*D)],axis=0)[np.newaxis,:]).T,
                    (ysim - np.mean(ysim,axis=-1)[np.newaxis,:]).T
                    ) / (N - 1)
                
                # Apply the EnKS update
                log_EnKS[:,int((s)*D):int((s+1)*D)] -= \
                    np.linalg.multi_dot((
                        cov_xy,
                        cov_y_inv,
                        (ysim - x_obs[t,...][np.newaxis,:]) )).T
                
        # Store the results for the EnKS
        dct[rep][N]['EnKS']     = {
            'mean'  : np.mean(log_EnKS,axis=0),
            'cov'   : np.cov(log_EnKS.T),
            'map'   : map_EnKS,
            'signal': signal_EnKS}
                
        # Delete the logs
        del log_EnKS, map_EnKS, signal_EnKS
        
        #%%
        
        # =============================================================================
        # single-pass EnRTSS (empirical)
        # =============================================================================
        
        # Reset the random seed
        np.random.seed(rep)
        
        # Copy the EnKF array
        log_EnRTSS      =  copy.copy(log_EnKF)
        
        # Store the map and signal for the last update pass
        map_EnRTSS      = np.zeros((T-1,1))*np.nan
        signal_EnRTSS   = np.zeros((T-1,N))
        
        # Go backwards through the time series
        for s in np.arange(T-2,-1,-1):
            
            # Calculate the Jt operator (see Raanes paper)
            Jt  = np.dot(
                np.dot(log_EnKF[:,int((s+0)*D):int((s+1)*D)].T,projector),
                np.linalg.pinv(np.dot(log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T,projector)) )
            
            # Determine the cross-covariance between analysis and forecast
            crosscov    = np.dot(
                np.dot(log_EnKF[:,int((s+0)*D):int((s+1)*D)].T,
                    projector),
                np.dot(log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T,
                    projector).T )
            
            # Determine the forecast precision matrix
            invcov      = np.linalg.inv(
                np.dot(
                    np.dot(log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T,
                        projector),
                    np.dot(log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T,
                        projector).T ) )
            
            # Store the map and signal for the last update pass
            map_EnRTSS[s,:]     = np.dot(
                crosscov,
                invcov)
            signal_EnRTSS[s,:]  = copy.copy(log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T)
            
            # Update the states
            log_EnRTSS[:,int((s+0)*D):int((s+1)*D)]  += np.dot(
                Jt,
                log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T).T
                

        # Store the results for the EnKS
        dct[rep][N]['EnRTSS (single-pass)']     = {
            'mean'  : np.mean(log_EnRTSS,axis=0),
            'cov'   : np.cov(log_EnRTSS.T),
            'map'   : map_EnRTSS,
            'signal': signal_EnRTSS}
                
        # Delete the logs
        del log_EnRTSS, map_EnRTSS, signal_EnRTSS
        
        #%%
    
        # =============================================================================
        # multi-pass EnRTSS (empirical)
        # =============================================================================
        
        # Reset the random seed
        np.random.seed(rep)
        
        # Copy the EnKF array
        log_EnRTSS      =  copy.copy(log_EnKF)
        
        # Start the filtering
        for t in np.arange(1,T,1):
            
            # Store the map and signal for the last update pass
            if t == T-1:
                
                # Store the map
                map_EnRTSS      = np.zeros((T-1,1))*np.nan
                
                # Store the signal
                signal_EnRTSS   = np.zeros((T-1,N))
    
            # Create a local copy of the backwards smoothing samples
            log_EnRTSS_orig    = copy.copy(log_EnRTSS)
            
            # Go backwards through the time series
            for s in np.arange(t-1,-1,-1):
                
                # The first operation is based on the filter, not smoothing samples
                if s == t-1:
                    
                    # Operator from Raanes et al.
                    Jt  = np.dot(
                        np.dot(log_EnRTSS_orig[:,int((s+0)*D):int((s+1)*D)].T,projector),
                        np.linalg.pinv(np.dot(log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T,projector)) )
                    
                    # Store the map and signal for the last update pass
                    if t == T-1:
                        map_EnRTSS[s,:]     = copy.copy(Jt)
                        signal_EnRTSS[s,:]  = copy.copy(log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T)
                    
                    # Update the states
                    log_EnRTSS[:,int((s+0)*D):int((s+1)*D)]  += np.dot(
                        Jt,
                        log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T).T
                    
                # Later operations are based on previous smoothing passes
                else:
            
                    # Operator from Raanes et al.
                    Jt  = np.dot(
                        np.dot(log_EnRTSS_orig[:,int((s+0)*D):int((s+1)*D)].T,projector),
                        np.linalg.pinv(np.dot(log_EnRTSS_orig[:,int((s+1)*D):int((s+2)*D)].T,projector)) )
                    
                    # Store the map and signal for the last update pass
                    if t == T-1:
                        map_EnRTSS[s,:]     = copy.copy(Jt)
                        signal_EnRTSS[s,:]  = copy.copy(log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnRTSS_orig[:,int((s+1)*D):int((s+2)*D)].T)
            
                    # Update the states
                    log_EnRTSS[:,int((s+0)*D):int((s+1)*D)]  += np.dot(
                        Jt,
                        log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnRTSS_orig[:,int((s+1)*D):int((s+2)*D)].T).T
    
        # Store the results for the EnKS
        dct[rep][N]['EnRTSS (multi-pass)']     = {
            'mean'  : np.mean(log_EnRTSS,axis=0),
            'cov'   : np.cov(log_EnRTSS.T),
            'map'   : map_EnRTSS,
            'signal': signal_EnRTSS}
                
        # Delete the logs
        del log_EnRTSS, log_EnRTSS_orig, map_EnRTSS, signal_EnRTSS
    
        #%%
        
        # =====================================================================
        # EnTS (joint-analysis)
        # =====================================================================
        
        # Reset the random seed
        np.random.seed(rep)
        
        # Draw prior samples
        X   = scipy.stats.multivariate_normal.rvs(
            mean    = mu_pri,
            cov     = cov_pri,
            size    = N)
        if D == 1: X = X[:,np.newaxis]
        
        # Create an array for the EnTS samples, copy prior samples in
        log_EnTS        = np.zeros((N,int(T*D)))
        log_EnTS[:,:D]  = copy.copy(X)
    
        # Create arrays for map and signal
        map_EnTS       = np.zeros((T,1))*np.nan
        signal_EnTS    = np.zeros((T,N))*np.nan
        
        # Walk through all time steps
        for t in np.arange(0,T,1):

            # Create the map components
            if order == 1:
        
                # Build increasingly large transport maps
                maps        = [[],[0]]
                nonmonotone = []
                monotone    = []
                for s in range(t+1):
                    
                    nonmonotone .append(copy.deepcopy(maps))
                    
                    monotone    .append([[O+s]])
                    
                    maps        .append([O+s])
                    
            elif order == 2:
                
                # Build increasingly large transport maps
                maps        = [[],[0],[0,0,'HF']]
                nonmonotone = []
                monotone    = []
                for s in range(t+1):
                    
                    nonmonotone .append(copy.deepcopy(maps))
                    
                    monotone    .append([[O+s],'iRBF '+str(O+s)])
                    
                    maps        .append([O+s])
                    maps        .append([O+s,O+s,'HF'])
                    
            elif order == 3:
                
                # Build increasingly large transport maps
                maps        = [[],[0],[0,0,'HF'],[0,0,0,'HF']]
                nonmonotone = []
                monotone    = []
                for s in range(t+1):
                    
                    nonmonotone .append(copy.deepcopy(maps))
                    
                    monotone    .append([[O+s],'iRBF '+str(O+s),'iRBF '+str(O+s)])
                    
                    maps        .append([O+s])
                    maps        .append([O+s,O+s,'HF'])
                    maps        .append([O+s,O+s,O+s,'HF'])

            # Sample observation noise
            noise = \
                scipy.stats.multivariate_normal.rvs(
                    mean    = np.zeros(O),
                    cov     = R,
                    size    = N)
            if O == 1: noise = noise[:,np.newaxis]
            
            # Duplicate the observations
            Ysim        = copy.copy(
                log_EnTS[:,int((t)*D):int((t+1)*D)] + noise)
            
            # Create the map input
            map_input   = np.column_stack((
                Ysim,
                np.flip(log_EnTS[:,:int((t+1)*D)],axis=-1)))
            
            if t == T-1:
                
                # Store map and signal
                crosscov            = np.dot(
                    np.dot(projector,copy.copy(log_EnTS[:,:int((t+1)*D)])).T,
                    np.dot(projector,copy.copy(Ysim)) )/N
                
                invcov              = np.dot(
                    np.dot(projector,copy.copy(Ysim)).T,
                    np.dot(projector,copy.copy(Ysim)) )/N
                invcov              = np.linalg.inv(invcov)
            
                map_EnTS[:,:]      = copy.copy(np.dot(crosscov,invcov))
                
                signal_EnTS[:,:]   = np.repeat(
                    Ysim - \
                    np.repeat(
                        a       = x_obs[t,:].reshape((1,O)),
                        repeats = N, 
                        axis    = 0),
                    repeats = T,
                    axis =-1).T
            
            # Delete and map object which might already exist
            if "tm" in globals():
                del tm
            
            # Parameterize the transport map
            tm     = transport_map(
                monotone                = monotone,
                nonmonotone             = nonmonotone,
                X                       = map_input,       
                polynomial_type         = "probabilist's hermite", 
                monotonicity            = "separable monotonicity",
                standardize_samples     = True)
            
            # Start optimizing the transport map
            tm.optimize()
        
            # Composite map section -------------------------------------------
        
            # Once the map is optimized, use it to convert the samples to samples from
            # the reference distribution X
            norm_samples = tm.map(map_input)
            
            # Create an array with the observations
            X_star = np.repeat(
                a       = x_obs[t,:].reshape((1,O)),
                repeats = N, 
                axis    = 0)
            
            # Apply the inverse map to obtain samples from (Y,Z_a)
            ret = tm.inverse_map(
                X_star      = X_star,
                Z           = norm_samples)
            
            # Composite map section end ---------------------------------------
        
            # Store results
            log_EnTS[:,:int((t+1)*D)] = copy.copy(np.flip(ret,axis=-1))
            
            if t < T-1:
                
                # Deterministic forecast
                log_EnTS[:,int((t+1)*D):int((t+2)*D)] = np.dot(
                    A,
                    log_EnTS[:,int((t)*D):int((t+1)*D)].T).T
        
                # Sample stochastic noise
                noise = \
                    scipy.stats.multivariate_normal.rvs(
                        mean    = np.zeros(O),
                        cov     = Q,
                        size    = N)
                if D == 1: noise = noise[:,np.newaxis]
        
                # Add the noise
                log_EnTS[:,int((t+1)*D):int((t+2)*D)] += noise
            
        # Store the results for the EnKS
        dct[rep][N]['EnTS (joint-analysis)']     = {
            'mean'  : np.mean(log_EnTS,axis=0),
            'cov'   : np.cov(log_EnTS.T),
            'map'   : map_EnTS,
            'signal': signal_EnTS}
                
        # Delete the logs
        del log_EnTS, map_EnTS, signal_EnTS
        
        #%%
        
        # =============================================================================
        # EnTF
        # =============================================================================
        
        # Reset the random seed
        np.random.seed(rep)
        
        # Draw prior samples
        X   = scipy.stats.multivariate_normal.rvs(
            mean    = mu_pri,
            cov     = cov_pri,
            size    = N)
        if D == 1: X = X[:,np.newaxis]
        
        # Create an array for the EnTF samples, copy prior samples in
        log_EnTF        = np.zeros((N,int(T*D)))
        log_EnTF[:,:D]  = copy.copy(X)
        
        # Create a copy for the forecast
        log_EnTF_f      = copy.copy(log_EnTF)
    
        for t in np.arange(0,T,1):
            
            # Build increasingly large transport maps
            if order == 1:
            
                nonmonotone = [[[],[0]]]
                monotone    = [[[1]]]
            
            elif order == 2:
                
                nonmonotone = [[[],[0],[0,0,'HF']]]
                monotone    = [[[1],'iRBF 1']]
                
            elif order == 3:
                
                nonmonotone = [[[],[0],[0,0,'HF'],[0,0,0,'HF']]]
                monotone    = [[[1],'iRBF 1','iRBF 1']]
            
            # Sample observation noise
            noise = \
                scipy.stats.multivariate_normal.rvs(
                    mean    = np.zeros(O),
                    cov     = R,
                    size    = N)
            if O == 1: noise = noise[:,np.newaxis]
            
            # Duplicate the observations
            Ysim        = copy.copy(
                log_EnTF[:,int((t)*D):int((t+1)*D)] + noise)
            
            # Create the map input
            map_input   = np.column_stack((
                Ysim,
                np.flip(log_EnTF[:,int((t)*D):int((t+1)*D)],axis=-1)))
            
            # Delete and map object which might already exist
            if "tm" in globals():
                del tm
            
            # Parameterize the transport map
            tm     = transport_map(
                monotone                = monotone,
                nonmonotone             = nonmonotone,
                X                       = map_input,       
                polynomial_type         = "probabilist's hermite", 
                monotonicity            = "separable monotonicity",
                standardize_samples     = True)
            
            # Start optimizing the transport map
            tm.optimize()
            
            # Composite map section -------------------------------------------
        
            # Once the map is optimized, use it to convert the samples to samples from
            # the reference distribution X
            norm_samples = tm.map(map_input)
            
            # Create an array with the observations
            X_star = np.repeat(
                a       = x_obs[t,:].reshape((1,O)),
                repeats = N, 
                axis    = 0)
            
            # Apply the inverse map to obtain samples from (Y,Z_a)
            ret = tm.inverse_map(
                X_star      = X_star,
                Z           = norm_samples)
            
            # Composite map section end ---------------------------------------
        
            # Store results
            log_EnTF[:,int((t)*D):int((t+1)*D)] = copy.copy(np.flip(ret,axis=-1))
            
            if t < T-1:
                
                # Deterministic forecast
                log_EnTF[:,int((t+1)*D):int((t+2)*D)] = np.dot(
                    A,
                    log_EnTF[:,int((t)*D):int((t+1)*D)].T).T
        
                # Sample stochastic noise
                noise = \
                    scipy.stats.multivariate_normal.rvs(
                        mean    = np.zeros(O),
                        cov     = Q,
                        size    = N)
                if D == 1: noise = noise[:,np.newaxis]
        
                # Add the noise
                log_EnTF[:,int((t+1)*D):int((t+2)*D)] += noise
                
                log_EnTF_f[:,int((t+1)*D):int((t+2)*D)] = copy.copy(
                    log_EnTF[:,int((t+1)*D):int((t+2)*D)])
            
        # Store the results for the EnKS
        dct[rep][N]['EnTF']     = {
            'mean'  : np.mean(log_EnTF,axis=0),
            'cov'   : np.cov(log_EnTF.T)}
        
        #%%
        
        # =============================================================================
        # EnTS (backward, single-pass)
        # =============================================================================
        
        # Reset the random seed
        np.random.seed(rep)
        
        # Create the current and forecast arrays
        log_EnTS    = copy.copy(log_EnTF)
        log_EnTS_f  = copy.copy(log_EnTF_f)
        
        # Once we are done, do a backwards pass
        if order == 1:
        
            nonmonotone = [[[],[0]]]
            monotone    = [[[1]]]
        
        elif order == 2:
            
            nonmonotone = [[[],[0],[0,0,'HF']]]
            monotone    = [[[1],'iRBF 1']]
            
        elif order == 3:
            
            nonmonotone = [[[],[0],[0,0,'HF'],[0,0,0,'HF']]]
            monotone    = [[[1],'iRBF 1','iRBF 1']]
        
        # Create the map input
        map_input   = np.random.uniform(size=(N,D+D))
        
        # Delete and map object which might already exist
        if "tm" in globals():
            del tm
        
        # Parameterize the transport map
        tm     = transport_map(
            monotone                = monotone,
            nonmonotone             = nonmonotone,
            X                       = map_input,       
            polynomial_type         = "probabilist's hermite", 
            monotonicity            = "separable monotonicity",
            standardize_samples     = True)
        
        # Create arrays for map and signal
        map_EnTS       = np.zeros((T,1))*np.nan
        signal_EnTS    = np.zeros((T,N))*np.nan
        
        
        # We can re-use Ysim from the filtering algorithm!
        crosscov            = np.dot(
            np.dot(projector,copy.copy(log_EnTS_f[:,int((T-1)*D):int((T)*D)])).T,
            np.dot(projector,copy.copy(Ysim)) )/N
        
        invcov              = np.dot(
            np.dot(projector,copy.copy(Ysim)).T,
            np.dot(projector,copy.copy(Ysim)) )/N
        invcov              = np.linalg.inv(invcov)
        
        # Save the first signal and map
        map_EnTS[-1:,:]     = copy.copy(np.dot(crosscov,invcov))
        signal_EnTS[-1:,:]  = copy.copy(Ysim)[:,0] - copy.copy(x_obs[T-1,:])
        
        # Go back through the graph
        for t in range(T-2,-1,-1):
            
            # Store map and signal
            crosscov            = np.dot(
                np.dot(projector,copy.copy(log_EnTS[:,int((t+0)*D):int((t+1)*D)])).T,
                np.dot(projector,copy.copy(log_EnTS_f[:,int((t+1)*D):int((t+2)*D)])) )/(N - 1)
            
            invcov              = np.dot(
                np.dot(projector,copy.copy(log_EnTS_f[:,int((t+1)*D):int((t+2)*D)])).T,
                np.dot(projector,copy.copy(log_EnTS_f[:,int((t+1)*D):int((t+2)*D)])) )/(N - 1)
            invcov              = np.linalg.inv(invcov)
            
            map_EnTS[t,:]     = copy.copy(np.dot(crosscov,invcov))
            signal_EnTS[t,:]  = \
                copy.copy(log_EnTS[:,int((t+1)*D):int((t+2)*D)])[:,0] - \
                copy.copy(log_EnTS_f[:,int((t+1)*D):int((t+2)*D)])[:,0]
            
            # Build the map input
            map_input = np.column_stack((
                copy.copy(log_EnTS_f[:,int((t+1)*D):int((t+2)*D)]),
                copy.copy(log_EnTS[:,int((t+0)*D):int((t+1)*D)]) ))
            
            # Reset the map
            tm.reset(map_input)
        
            # Start optimizing the transport map
            tm.optimize()
            
            # Composite map section -------------------------------------------
        
            # Once the map is optimized, use it to convert the samples to samples from
            # the reference distribution X
            norm_samples = tm.map(map_input)
            
            # Create an array with the observations
            X_star = copy.copy(log_EnTS[:,int((t+1)*D):int((t+2)*D)])
            
            # Apply the inverse map to obtain samples from (Y,Z_a)
            ret = tm.inverse_map(
                X_star      = X_star,
                Z           = norm_samples)
            
            # Composite map section end ---------------------------------------
            
            # Store the result
            log_EnTS[:,int((t+0)*D):int((t+1)*D)] = copy.copy(ret)
            
        # Store the results for the EnKS
        dct[rep][N]['EnTS (backward, single-pass)']     = {
            'mean'  : np.mean(log_EnTS,axis=0),
            'cov'   : np.cov(log_EnTS.T),
            'map'   : map_EnTS,
            'signal': signal_EnTS}
                
        # Delete the logs
        del log_EnTS, log_EnTS_f, map_EnTS, signal_EnTS
        
        
        #%%
        
        # =============================================================================
        # EnTS (backward, multi-pass)
        # =============================================================================
        
        # Reset the random seed
        np.random.seed(rep)
        
        # Create the current and forecast arrays
        log_EnTS    = copy.copy(log_EnTF)
        log_EnTS_f  = copy.copy(log_EnTF_f)
        
        if order == 1:
        
            nonmonotone = [[[],[0]]]
            monotone    = [[[1]]]
        
        elif order == 2:
            
            nonmonotone = [[[],[0],[0,0,'HF']]]
            monotone    = [[[1],'iRBF 1']]
            
        elif order == 3:
            
            nonmonotone = [[[],[0],[0,0,'HF'],[0,0,0,'HF']]]
            monotone    = [[[1],'iRBF 1','iRBF 1']]
        
        # Delete and map object which might already exist
        if "tm" in globals():
            del tm
        
        # Parameterize the transport map
        tm     = transport_map(
            monotone                = monotone,
            nonmonotone             = nonmonotone,
            X                       = map_input,       
            polynomial_type         = "probabilist's hermite", 
            monotonicity            = "separable monotonicity",
            standardize_samples     = True)
        
        # Go through all timesteps
        for t in np.arange(0,T,1):
            
            # If we are in the last time step, store the map and the signal
            if t == T-1:
                map_EnTS        = np.zeros((T,1))*np.nan
                signal_EnTS     = np.zeros((T,N))*np.nan
                
                # We can re-use Ysim from the filtering algorithm!
                crosscov            = np.dot(
                    np.dot(projector,copy.copy(log_EnTS_f[:,int((T-1)*D):int((T)*D)])).T,
                    np.dot(projector,copy.copy(Ysim)) )/N
                
                invcov              = np.dot(
                    np.dot(projector,copy.copy(Ysim)).T,
                    np.dot(projector,copy.copy(Ysim)) )/N
                invcov              = np.linalg.inv(invcov)
                
                map_EnTS[-1:,:]   = copy.copy(np.dot(crosscov,invcov))
                signal_EnTS[-1:,:]= copy.copy(Ysim)[:,0] - copy.copy(x_obs[T-1,:])
            
            
            # -------------------------------------------------------------------------
            # And for my next trick: the multi-pass RTS run
            log_EnTS_orig  = copy.copy(log_EnTS)
    
            # Make a single backwards pass
            for s in np.arange(t-1,-1,-1):
            
                if s == t-1:
                    
                    # The first transport map is built from forecast and analysis
                    map_input = np.column_stack((
                        copy.copy(log_EnTS_f[:,int((s+1)*D):int((s+2)*D)]),
                        copy.copy(log_EnTS_orig[:,int((s+0)*D):int((s+1)*D)]) ))
                    
                    # If we are in the last time step, store the map and the signal
                    if t == T-1:
                        #-marked-
                        crosscov            = np.dot(
                            np.dot(projector,copy.copy(log_EnTS_orig[:,int((s+0)*D):int((s+1)*D)])).T,
                            np.dot(projector,copy.copy(log_EnTS_f[:,int((s+1)*D):int((s+2)*D)])) )/N
                        
                        invcov              = np.dot(
                            np.dot(projector,copy.copy(log_EnTS_f[:,int((s+1)*D):int((s+2)*D)])).T,
                            np.dot(projector,copy.copy(log_EnTS_f[:,int((s+1)*D):int((s+2)*D)])) )/N
                        invcov              = np.linalg.inv(invcov)
                        
                        map_EnTS[s,:]       = copy.copy(np.dot(crosscov,invcov))
                        signal_EnTS[s,:]    = \
                            copy.copy(log_EnTS[:,int((s+1)*D):int((s+2)*D)])[:,0] - \
                            copy.copy(log_EnTS_f[:,int((s+1)*D):int((s+2)*D)])[:,0]
                    
                else:
                    
                    # The first transport map is built from forecast and analysis
                    map_input = np.column_stack((
                        copy.copy(log_EnTS_orig[:,int((s+1)*D):int((s+2)*D)]),
                        copy.copy(log_EnTS_orig[:,int((s+0)*D):int((s+1)*D)]) ))
                    
                    
                    
                    # If we are in the last time step, store the map and the signal
                    if t == T-1:
                        
                        #-marked-
                        crosscov            = np.dot(
                            np.dot(projector,copy.copy(log_EnTS_orig[:,int((s+0)*D):int((s+1)*D)])).T,
                            np.dot(projector,copy.copy(log_EnTS_orig[:,int((s+1)*D):int((s+2)*D)])) )/N
                        
                        invcov              = np.dot(
                            np.dot(projector,copy.copy(log_EnTS_orig[:,int((s+1)*D):int((s+2)*D)])).T,
                            np.dot(projector,copy.copy(log_EnTS_orig[:,int((s+1)*D):int((s+2)*D)])) )/N
                        invcov              = np.linalg.inv(invcov)
                        
                        map_EnTS[s,:]       = copy.copy(np.dot(crosscov,invcov))
                        signal_EnTS[s,:]    = \
                            copy.copy(log_EnTS[:,int((s+1)*D):int((s+2)*D)])[:,0] - \
                            copy.copy(log_EnTS_orig[:,int((s+1)*D):int((s+2)*D)])[:,0]

                tm.reset(map_input)
                    
                # Start optimizing the transport map
                tm.optimize()
                
                # Composite map section ---------------------------------------
            
                # Once the map is optimized, use it to convert the samples to samples from
                # the reference distribution X
                norm_samples = tm.map(copy.copy(map_input))
                
                # Create an array with the observations
                X_star = copy.copy(log_EnTS[:,int((s+1)*D):int((s+2)*D)])
                
                # Apply the inverse map to obtain samples from (Y,Z_a)
                ret = tm.inverse_map(
                    X_star      = X_star,
                    Z           = norm_samples)
                
                # Composite map section end -----------------------------------
                
                # Store the result
                log_EnTS[:,int((s+0)*D):int((s+1)*D)] = copy.copy(ret)
            
        # Store the results for the EnKS
        dct[rep][N]['EnTS (backward, multi-pass)']     = {
            'mean'  : np.mean(log_EnTS,axis=0),
            'cov'   : np.cov(log_EnTS.T),
            'map'   : map_EnTS,
            'signal': signal_EnTS}
                
        # Delete the logs
        del log_EnTS, log_EnTS_f, map_EnTS, signal_EnTS
        
#%%

# Store the results
pickle.dump(dct,open('autoregressive_model_results.p','wb'))
        