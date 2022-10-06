import numpy as np
import scipy.stats
import copy
from transport_filters import EnsembleTransportFilter
import sys
sys.path.append('../Example - Autoregressive Model/')
from transport_map_138 import *

def EnsembleTransportSmoother(model, T, X, order):

    # =====================================================================
    # Transport EnKS (empirical)
    # =====================================================================
        
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
    
    # identify the number of samples
    N = X.shape[0]
    # Define a projector which extracts the anomalies of an ensemble matrix
    #projector   = np.identity(N) - np.ones((N,N))/N
    
    # Create an array for the EnTS samples, copy prior samples in
    log_EnTS        = np.zeros((N,int(T*D)))
    log_EnTS[:,:D]  = copy.copy(X)
    
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

    return log_EnTS
        
def SinglePassEnsembleTransportBIT(model, T, X, order): 

    # =============================================================================
    # EnTS (backward, single-pass)
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
    
    # identify the number of samples
    N = X.shape[0]
    # Define a projector which extracts the anomalies of an ensemble matrix
    projector   = np.identity(N) - np.ones((N,N))/N
    
    # run filter
    log_EnTF, log_EnTF_f, Ysim = EnsembleTransportFilter(model, T, X, order) 
    
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
            
    # Delete the logs
    return log_EnTS, map_EnTS, signal_EnTS
        
def MultiPassEnsembleTransportBIT(model, T, X, order): 

    # =============================================================================
    # EnTS (backward, multi-pass)
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

    # identify the number of samples
    N = X.shape[0]
    # Define a projector which extracts the anomalies of an ensemble matrix
    projector   = np.identity(N) - np.ones((N,N))/N
    
    # run filter
    log_EnTF, log_EnTF_f, Ysim = EnsembleTransportFilter(model, T, X, order) 

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
            
    # Delete the logs
    return log_EnTS, map_EnTS, signal_EnTS

def MultiPassEnsembleTransportFIT(model, T, X, order): 

    # =============================================================================
    # EnTS (forward, multi-pass)
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

    # identify the number of samples
    N = X.shape[0]
  
    # Create array to store samples
    log_FIT        = np.zeros((N,int(T*D)))
    log_FIT[:,:D]  = copy.copy(X)
  
    if order == 1:
  
        nonmonotone_s1 = [[[],[0]]]
        monotone_s1    = [[[1]]]
         
        nonmonotone_sp = [[[],[0],[1]]]
        monotone_sp    = [[[2]]]        #diagonal part depends on linearly on last variable
  
    elif order == 2:
      
        nonmonotone_s1 = [[[],[0],[0,0,'HF']]]
        monotone_s1    = [[[1],'iRBF 1']]
      
    elif order == 3:
      
        nonmonotone_s1 = [[[],[0],[0,0,'HF'],[0,0,0,'HF']]]
        monotone_s1    = [[[1],'iRBF 1','iRBF 1']]
  
    # Create the map input
    map_input_s1 = np.random.uniform(size=(N,O+D))
    map_input_sp = np.random.uniform(size=(N,O+D+D))

    # Delete and map object which might already exist
    if "tm" in globals():
        del tm
    if "tm_s1" in globals():
        del tm_s1
    if "tm_sp" in globals():
        del tm_sp

    # Parameterize the transport map
    tm_s1 = transport_map(
        monotone                = monotone_s1,
        nonmonotone             = nonmonotone_s1,
        X                       = map_input_s1,
        polynomial_type         = "probabilist's hermite", 
        monotonicity            = "separable monotonicity",
        standardize_samples     = True)
    tm_sp = transport_map(
        monotone                = monotone_sp,
        nonmonotone             = nonmonotone_sp,
        X                       = map_input_sp,   
        polynomial_type         = "probabilist's hermite", 
        monotonicity            = "separable monotonicity",
        standardize_samples     = True)

    # Go through all timesteps to assimilate each data
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
        #log_FIT_Y[:,int((t)*D):int((t+1)*D)] = copy.copy(ysim.T)
      
        # Create a local copy of the backwards smoothing samples
        log_FIT_orig = copy.copy(log_FIT)

        ### Update the first state ###

        # extract x(1) state and define map inputs
        x1 = log_FIT_orig[:,0:D]
        map_input = np.column_stack((copy.copy(ysim.T), copy.copy(x1)))

        tm_s1.reset(map_input)
          
        # Start optimizing the transport map
        tm_s1.optimize()

        # Composite map section ---------------------------------------

        # Once the map is optimized, use it to convert the samples to samples from
        # the reference distribution X
        norm_samples = tm_s1.map(copy.copy(map_input))
      
        # Create an array with the observations
        Y_star = np.repeat(a=x_obs[t,:].reshape((1,O)), repeats=N, axis=0)
      
        # Apply the inverse map to obtain samples from (Y,Z_a) and store result
        log_FIT[:,0:D] = tm_s1.inverse_map(
            X_star      = Y_star,
            Z           = norm_samples)

        # Composite map section end -----------------------------------
        
        ### Update the remaining states in forwards pass ###
        for s in np.arange(1,t+1,1):
      
            # extract states and define map inputs
            xsm1 = log_FIT_orig[:,int((s-1)*D):int((s)*D)]
            xs   = log_FIT_orig[:,int((s)*D):int((s+1)*D)]
            map_input = np.column_stack((copy.copy(ysim.T), copy.copy(xsm1), copy.copy(xs)))
 
            tm_sp.reset(map_input)
              
            # Start optimizing the transport map
            tm_sp.optimize()
          
            # Composite map section ---------------------------------------
      
            # Once the map is optimized, use it to convert the samples to samples from
            # the reference distribution X
            norm_samples = tm_sp.map(copy.copy(map_input))
          
            # Create an array with the observations
            X_star = copy.copy(log_FIT[:,int((s-1)*D):int((s)*D)])
            YX_star = np.column_stack((Y_star, X_star))          

            # Apply the inverse map to obtain samples from (Y,Z_a)
            log_FIT[:,int((s)*D):int((s+1)*D)] = \
                tm_sp.inverse_map(
                X_star      = YX_star,
                Z           = norm_samples)
          
            # Composite map section end -----------------------------------
          
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
          
    # Delete the logs
    return log_FIT
