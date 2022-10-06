import numpy as np
import scipy.stats
import copy

def KalmanSmoother(model,T):

    # =========================================================================
    # Kalman Smoother
    # =========================================================================

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
 
    # Pre-allocate mean vector and covariance matrix for the analytical reference
    muT_KS     = np.zeros((int(T*D),1))
    covT_KS    = np.zeros((int(T*D),int(T*D)))
    
    # Fill in the prior
    muT_KS[:D,0]   = copy.copy(mu_a)
    covT_KS[:D,:D] = copy.copy(cov_a)

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
    
    return muT_KS, covT_KS

def RTSSmoother(model, T):
    
    # =============================================================================
    # Rauch-Tung-Striebel Smoother
    # =============================================================================
    
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
 
    # Pre-allocate mean vector and covariance matrix for the analytical reference
    muT     = np.zeros((int(T*D),1))
    covT    = np.zeros((int(T*D),int(T*D)))
    
    # Fill in the prior
    muT[:D,0]   = copy.copy(mu_a)
    covT[:D,:D] = copy.copy(cov_a)
 
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

 
    return muT_RTS, covT_RTS
