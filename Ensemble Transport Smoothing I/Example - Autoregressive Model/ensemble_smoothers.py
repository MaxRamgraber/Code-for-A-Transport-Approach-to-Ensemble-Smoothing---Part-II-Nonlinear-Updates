import numpy as np
import scipy.stats
import copy
from ensemble_filters import EnsembleKF

def EnsembleKS(model, T, X):

    # =====================================================================
    # EnKS (empirical)
    # =====================================================================
    
    # extract matrices
    x_obs = model['x_obs']

    # determine dimensions
    D = model['A'].shape[0]
    O = model['H'].shape[0]
  
    # determine the number of samples 
    N = X.shape[0]

    # run EnKF first
    log_EnKF, _, log_EnKF_Y, gain_EnKF, signal_EnKF = EnsembleKF(model, T, X) 

    # Copy the EnKF array
    log_EnKS = copy.copy(log_EnKF)

    # Assimilate data at time t
    for t in np.arange(0,T,1):
        
        # Define array to store smoothing gain and signal in the last step
        if t==(T-1):
            gain   = np.zeros((T,D,O))*np.nan
            signal = np.zeros((T,N,O))*np.nan
            gain[T-1,:,:]   = gain_EnKF[-1,:,:]
            signal[T-1,:,:] = signal_EnKF[-1,:,:]

        # The EnKS is based on a multi-pass procedure; go through all time steps so far
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

            # compute gain and signal
            K = np.dot(cov_xy, cov_y_inv)
            signal_s = ysim - x_obs[t,...][np.newaxis,:]
            
            # Apply the EnKS update
            log_EnKS[:,int((s)*D):int((s+1)*D)] -= np.dot(K, signal_s).T

            # save gain and signal
            if t==(T-1):
                gain[s,:,:] = K
                signal[s,:,:] = signal_s.T

    return log_EnKS, gain, signal
        
def SinglePassEnsembleRTS(model, T, X):

    # =============================================================================
    # single-pass EnRTSS (empirical)
    # =============================================================================

    # determine dimensions
    D = model['A'].shape[0]
    O = model['H'].shape[0]

    # Define a projector which extracts the anomalies of an ensemble matrix
    N = X.shape[0]
    #projector = np.identity(N) - np.ones((N,N))/N
    
    # run EnKF first
    log_EnKF, log_EnKF_f, _, gain_EnKF, signal_EnKF = EnsembleKF(model, T, X) 

    # Copy the EnKF array
    log_EnRTSS = copy.copy(log_EnKF)

    # Define array to store smoothing gain and signal
    gain   = np.zeros((T,D,O))*np.nan
    signal = np.zeros((T,N,O))*np.nan
    gain[T-1,:,:] = gain_EnKF[-1,:,:]
    signal[T-1,:,:] = signal_EnKF[-1,:,:]

    # Go backwards through the time series
    for s in np.arange(T-2,-1,-1):

         # extract samples
         Xs     = log_EnKF[:,int((s)*D):int((s+1)*D)]
         Xf_sp1 = log_EnKF_f[:,int((s+1)*D):int((s+2)*D)]

         # compute empirical covariances
         Sigma = np.cov(np.hstack((Xs,Xf_sp1)).T)
    
         # compute Kalman gain
         K = np.linalg.solve(Sigma[D:,D:], Sigma[:D,D:]).T
    
         # extract signal at step s
         signal_s = log_EnRTSS[:,int((s+1)*D):int((s+2)*D)] - Xf_sp1

         # update the states
         log_EnRTSS[:,int((s+0)*D):int((s+1)*D)] += np.dot(K, signal_s.T).T

         # store gain and signal
         gain[s,:,:]   = K
         signal[s,:,:] = signal_s

         # Calculate the Jt operator (see Raanes paper)
         #Jt  = np.dot(
         #    np.dot(log_EnKF[:,int((s+0)*D):int((s+1)*D)].T,projector),
         #    np.linalg.pinv(np.dot(log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T,projector)) )
        
         # Update the states
         #log_EnRTSS[:,int((s+0)*D):int((s+1)*D)] += np.dot(
         #    Jt,
         #    log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T).T
                
    return log_EnRTSS, gain, signal

def MultiPassEnsembleRTS(model, T, X):

    # =============================================================================
    # multi-pass EnRTSS (empirical)
    # =============================================================================

    # determine dimensions
    D = model['A'].shape[0]
    O = model['H'].shape[0]

    # Define a projector which extracts the anomalies of an ensemble matrix
    N = X.shape[0]
    #projector  = np.identity(N) - np.ones((N,N))/N

    # run EnKF first
    log_EnKF, log_EnKF_f, _, gain_EnKF, signal_EnKF = EnsembleKF(model, T, X) 

    # Copy the EnKF array
    log_EnRTSS = copy.copy(log_EnKF)
   
    # Start the filtering
    for t in np.arange(0,T,1):

        # Create a local copy of the backwards smoothing samples
        log_EnRTSS_orig    = copy.copy(log_EnRTSS)
        
        # Define array to store smoothing gain and signal in the last step
        if t==(T-1):
            gain   = np.zeros((T,D,O))*np.nan
            signal = np.zeros((T,N,O))*np.nan
            #gain[T-1,:,:] = gain_EnKF[-1,:,:]
            #signal[T-1,:,:] = signal_EnKF[-1,:,:]
        
        # Go backwards through the time series
        for s in np.arange(t-1,-1,-1):
            
            # The first operation is based on the filter, not smoothing samples
            if s == t-1:
                
                # extract samples
                Xs     = log_EnRTSS_orig[:,int((s)*D):int((s+1)*D)]
                Xf_sp1 = log_EnKF_f[:,int((s+1)*D):int((s+2)*D)]

                # compute empirical covariances
                Sigma = np.cov(np.hstack((Xs,Xf_sp1)).T)
    
                # compute Kalman gain
                K = np.linalg.solve(Sigma[D:,D:], Sigma[:D,D:]).T

                # extract signal 
                signal_s = log_EnRTSS[:,int((s+1)*D):int((s+2)*D)] - Xf_sp1

                # Operator from Raanes et al.
                #Jt  = np.dot(
                #    np.dot(log_EnRTSS_orig[:,int((s+0)*D):int((s+1)*D)].T,projector),
                #    np.linalg.pinv(np.dot(log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T,projector)) )
                
                # Update the states
                #log_EnRTSS[:,int((s+0)*D):int((s+1)*D)]  += np.dot(
                #    Jt,
                #    log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnKF_f[:,int((s+1)*D):int((s+2)*D)].T).T
                
            # Later operations are based on previous smoothing passes
            else:
                
                # extract samples
                Xs     = log_EnRTSS_orig[:,int((s)*D):int((s+1)*D)]
                Xsp1   = log_EnRTSS_orig[:,int((s+1)*D):int((s+2)*D)]

                # compute empirical covariances
                Sigma = np.cov(np.hstack((Xs,Xsp1)).T)
    
                # compute Kalman gain
                K = np.linalg.solve(Sigma[D:,D:], Sigma[:D,D:]).T
                
                # extract signal 
                signal_s = log_EnRTSS[:,int((s+1)*D):int((s+2)*D)] - Xsp1
        
                # Operator from Raanes et al.
                #Jt  = np.dot(
                #    np.dot(log_EnRTSS_orig[:,int((s+0)*D):int((s+1)*D)].T,projector),
                #    np.linalg.pinv(np.dot(log_EnRTSS_orig[:,int((s+1)*D):int((s+2)*D)].T,projector)) )
                
                # Update the states
                #log_EnRTSS[:,int((s+0)*D):int((s+1)*D)]  += np.dot(
                #    Jt,
                #    log_EnRTSS[:,int((s+1)*D):int((s+2)*D)].T-log_EnRTSS_orig[:,int((s+1)*D):int((s+2)*D)].T).T

            # Update the states
            log_EnRTSS[:,int((s+0)*D):int((s+1)*D)] += np.dot(K, signal_s.T).T

            # save signal and gain 
            if t==(T-1):
                gain[s,:,:]   = K
                signal[s,:,:] = signal_s

    return log_EnRTSS, gain, signal

def MultiPassEnsembleFIT(model, T, X):

    # =============================================================================
    # multi-pass FIT (empirical)
    # =============================================================================
    
    # extract matrices
    A = model['A']
    H = model['H']
    R = model['R']
    Q = model['Q']
    x_obs = model['x_obs']

    # determine dimensions
    D = model['A'].shape[0]
    O = model['H'].shape[0]
    
    # Turn X into a 2D arary if one-dimensional
    if D == 1: X = X[:,np.newaxis]

    # Define a projector which extracts the anomalies of an ensemble matrix
    N = X.shape[0]
    
    # Create a storage log for the samples
    log_FIT        = np.zeros((N,int(T*D)))
    log_FIT[:,:D]  = copy.copy(X)

    # Create a log for the observation predictions
    log_FIT_Y      = copy.copy(log_FIT)
    
    # Start the filtering
    for t in np.arange(0,T,1):
        
        # Sample stochastic noise
        noise = \
            scipy.stats.multivariate_normal.rvs(
                mean    = np.zeros(O),
                cov     = R,
                size    = N)
        if O == 1: noise = noise[np.newaxis,:]

        # Create observation predictions by perturbing the forecast
        ysim  = np.dot(H,copy.copy(log_FIT[:,int((t)*D):int((t+1)*D)]).T)
        ysim  += noise
        
        # Store the result
        log_FIT_Y[:,int((t)*D):int((t+1)*D)] = copy.copy(ysim.T)

        # Create a local copy of the backwards smoothing samples
        log_FIT_orig = copy.copy(log_FIT)
        
        # Define array to store smoothing gain and signal in the last step
        if t==(T-1):
            gainK   = np.zeros((T,D,O))*np.nan
            gainC   = np.zeros((T,D,D))*np.nan
            signalK = np.zeros((T,N,O))*np.nan
            signalC = np.zeros((T,N,O))*np.nan

        ### Update the first state ###

        # extract x(1) state
        x1 = log_FIT_orig[:,0:D]

        # compute empirical covariance 
        Sigma = np.cov(np.hstack((x1, ysim.T)).T)

        # compute Kalman gain
        cov_xy = Sigma[0:D,D:]
        cov_y  = Sigma[D:,D:]
        if cov_y.shape != (O,O):
            cov_y = cov_y.reshape((O,O))
        # Get its precision matrix
        cov_y_inv = np.linalg.inv(cov_y)
        K = -1. * np.dot(cov_xy, cov_y_inv)        

        # extract signal
        signalK_s = ysim - x_obs[t,...][np.newaxis,:]
        # update states
        log_FIT[:,0:D] = log_FIT_orig[:,0:D] + np.dot(K, signalK_s).T

        # save signal and gain
        if t==(T-1):
            gainK[0,:,:]   = K
            signalK[0,:,:] = signalK_s.T

        # Go forwards through the time series
        for s in np.arange(1,t+1,1):

            # extract states
            xsm1 = log_FIT_orig[:,int((s-1)*D):int((s)*D)]
            xs   = log_FIT_orig[:,int((s)*D):int((s+1)*D)]

            # compute empirical covariances
            Sigma = np.cov(np.hstack((xsm1, xs, ysim.T)).T)

            # compute gain terms
            Sigma_y      = Sigma[(2*D):(2*D+O),(2*D):(2*D+O)]
            Sigma_xsm1   = Sigma[0:D, 0:D]
            Sigma_yxs    = Sigma[(2*D):(2*D+O), D:(2*D)]
            Sigma_yxsm1  = Sigma[(2*D):(2*D+O), 0:D]
            Sigma_xsxsm1 = Sigma[D:(2*D), 0:D]
            As = -1*np.block([[Sigma_y, Sigma_yxsm1], [Sigma_yxsm1.T, Sigma_xsm1]])
            Bs = np.vstack((Sigma_yxs, Sigma_xsxsm1.T))
            KCT = np.linalg.solve(As, Bs)

            # extract gain terms
            K = KCT[:O,:].T 
            C = KCT[O:,:].T

            # extract signal terms
            signalK_s = ysim - x_obs[t,...][np.newaxis,:]
            signalC_s = xsm1 - log_FIT[:,int((s-1)*D):int((s)*D)]

            # update states
            update = np.dot(K, signalK_s).T + np.dot(C, signalC_s.T).T
            log_FIT[:,int((s)*D):int((s+1)*D)] = log_FIT_orig[:,int((s)*D):int((s+1)*D)] + update

            # save signal and gain
            if t==(T-1):
                gainK[s,:,:]   = K
                gainC[s,:,:]   = C
                signalK[s,:,:] = signalK_s.T
                signalC[s,:,:] = signalC_s

        # Make a forecast to the next time step
        if t < T-1:
 
            # Deterministic forecast
            log_FIT[:,int((t+1)*D):int((t+2)*D)] = np.dot(
                A,
                log_FIT[:,int((t)*D):int((t+1)*D)].T).T
    
            # Sample stochastic noise
            noise = \
                scipy.stats.multivariate_normal.rvs(
                    mean    = np.zeros(D),
                    cov     = Q,
                    size    = N)
            if D == 1: noise = noise[:,np.newaxis]
    
            # Add the noise
            log_FIT[:,int((t+1)*D):int((t+2)*D)] += noise

    return log_FIT, gainK, gainC, signalK, signalC

