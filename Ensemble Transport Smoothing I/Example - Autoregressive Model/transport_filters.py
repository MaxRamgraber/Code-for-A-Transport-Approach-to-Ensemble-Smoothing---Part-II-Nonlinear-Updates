import numpy as np
import scipy.stats
import copy
import sys
sys.path.append('../Example - Autoregressive Model/')
from transport_map_138 import *

def EnsembleTransportFilter(model, T, X, order):

    # =============================================================================
    # EnTF
    # =============================================================================
    
    # identify the number of samples
    N = X.shape[0]
    
    # extract matrices
    A = model['A']
    H = model['H']  
    R = model['R']
    Q = model['Q']
    x_obs = model['x_obs']

    # determine dimensions
    D = model['A'].shape[0]
    O = model['H'].shape[0]
    
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
        
    return log_EnTF, log_EnTF_f, Ysim

