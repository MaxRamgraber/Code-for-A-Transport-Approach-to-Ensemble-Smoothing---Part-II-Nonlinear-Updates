import numpy as np
import scipy.stats
import copy

def EnsembleKF(model, T, X):
    
    # ====================================================================
    # EnKF (empirical)
    # =====================================================================
 
    # identify the number of samples
    N = X.shape[0]
    
    # extract matrices
    A = model['A']
    H = model['H']   
    R = model['R']
    Q = model['Q']
    mu_a  = model['mu_a']
    cov_a = model['cov_a']
    x_obs = model['x_obs']

    # determine dimensions
    D = A.shape[0]
    O = H.shape[0]
    
    # Turn X into a 2D arary if one-dimensional
    if D == 1: X = X[:,np.newaxis]
    
    # Create a storage log for the samples
    log_EnKF        = np.zeros((N,int(T*D)))
    log_EnKF[:,:D]  = copy.copy(X)
    
    # Create a log for the forecasts and observation predictions
    log_EnKF_f      = copy.copy(log_EnKF)
    log_EnKF_Y      = copy.copy(log_EnKF)
    
    # define arrays to store gain and signal
    gain     = np.zeros((T,D,O))*np.nan
    signal   = np.zeros((T,N,O))*np.nan

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
        
        # extract gain and signal
        gain_t = np.dot(cov_xy, cov_y_inv)
        signal_t = (ysim - x_obs[t,...][np.newaxis,:])

        # Apply the Kalman update
        log_EnKF[:,int((t)*D):int((t+1)*D)] -= np.dot(gain_t, signal_t).T
       
        # save gain/signal
        gain[t,:,:]   = gain_t
        signal[t,:,:] = signal_t.T
 
        # Make a forecast to the next time step
        if t < T-1:
            
            # Deterministic forecast
            log_EnKF[:,int((t+1)*D):int((t+2)*D)] = np.dot(
                A,
                log_EnKF[:,int((t)*D):int((t+1)*D)].T).T
    
            # Sample stochastic noise
            noise = \
                scipy.stats.multivariate_normal.rvs(
                    mean    = np.zeros(D),
                    cov     = Q,
                    size    = N)
            if D == 1: noise = noise[:,np.newaxis]
    
            # Add the noise
            log_EnKF[:,int((t+1)*D):int((t+2)*D)] += noise
            
            log_EnKF_f[:,int((t+1)*D):int((t+2)*D)] = copy.copy(
                log_EnKF[:,int((t+1)*D):int((t+2)*D)])
            
    return log_EnKF, log_EnKF_f, log_EnKF_Y, gain, signal
