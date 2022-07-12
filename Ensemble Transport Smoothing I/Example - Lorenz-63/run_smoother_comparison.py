if __name__ == '__main__':    

    # Load in a number of libraries we will use
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats
    import copy
    import scipy.optimize
    from transport_map_138 import *
    import time
    import pickle
    import os
    
    # Find the current path
    root_directory = os.path.dirname(os.path.realpath(__file__))

    # Define and initialize a random seed
    random_seed     = 0
    np.random.seed(random_seed)
    
    # Lorenz63 dynamics
    def lorenz_dynamics(t, Z, beta=8/3, rho=28, sigma=10):
        
        if len(Z.shape) == 1: # Only one particle
        
            dZ1ds   = - sigma*Z[0] + sigma*Z[1]
            dZ2ds   = - Z[0]*Z[2] + rho*Z[0] - Z[1]
            dZ3ds   = Z[0]*Z[1] - beta*Z[2]
            
            dyn     = np.asarray([dZ1ds, dZ2ds, dZ3ds])
            
        else:
            
            dZ1ds   = - sigma*Z[...,0] + sigma*Z[...,1]
            dZ2ds   = - Z[...,0]*Z[...,2] + rho*Z[...,0] - Z[...,1]
            dZ3ds   = Z[...,0]*Z[...,1] - beta*Z[...,2]
    
            dyn     = np.column_stack((dZ1ds, dZ2ds, dZ3ds))
    
        return dyn
    
    # Finds value of y for a given x using step size h
    # and initial value y0 at x0.
    def rk4(Z,fun,t=0,dt=1,nt=1):#(x0, y0, x, h):
        
        """
        Parameters
            t       : initial time
            Z       : initial states
            fun     : function to be integrated
            dt      : time step length
            nt      : number of time steps
        
        """
        
        # Prepare array for use
        if len(Z.shape) == 1: # We have only one particle, convert it to correct format
            Z       = Z[np.newaxis,:]
            
        # Go through all time steps
        for i in range(nt):
            
            # Calculate the RK4 values
            k1  = fun(t + i*dt,           Z);
            k2  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k1);
            k3  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k2);
            k4  = fun(t + i*dt + dt,      Z + dt*k3);
        
            # Update next value
            Z   += dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        return Z
    
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
        
        # Calculate the Kalman gain
        K       = np.linalg.multi_dot((
            C,
            H.T,
            np.linalg.inv(
                np.linalg.multi_dot((
                    H,
                    C,
                    H.T)) + R)))
        
        # Draw observation error realizations
        v       = scipy.stats.multivariate_normal.rvs(
            mean    = np.zeros(R.shape[0]),
            cov     = R,
            size    = N)
        
        # Perturb the observations
        obs     = y[np.newaxis,:] + v
        
        # Apply the stochastic Kalman update
        for n in range(N):
            X[n,:]  += np.dot(
                K,
                obs[n,:][:,np.newaxis] - np.dot(H,X[n,:][:,np.newaxis] ) )[:,0]
            
        return X
    

    # -------------------------------------------------------------------------
    # Set up exercise
    # -------------------------------------------------------------------------
    
    # Define problem dimensions
    O                   = 3 # Observation space dimensions
    D                   = 3 # State space dimensions
    
    # Define all ensemble sizes
    Ns                  = [50,100,175,250,375,500,750,1000]
    
    # Save full output for these ensemble sizes
    full_output         = [1000,100,50]
    
    # Set up time
    T                   = 1000  # Full time series length
    dt                  = 0.1   # Time step length
    times = np.arange(T)*dt
    
    # Define a maximum lag for the joint-analysis smoothers
    maxlag              = 100
    
    # Observation error
    obs_sd              = 2
    R                   = np.identity(O)*obs_sd**2
    
    # Forecast error
    mod_sd              = 1E-2
    Q                   = np.identity(D)*mod_sd**2
    
    # Repetition across this many random seeds
    repeats             = 100

    #%%
    
    # Create the synthetic reference
    synthetic_truth         = np.zeros((1000+T,1,D))
    
    # Initiate it with standard Gaussian samples
    synthetic_truth[0,0,:]  = scipy.stats.norm.rvs(size=3)
    
    # Go through all timesteps plus 1000 steps spinup
    for t in np.arange(0,1000+T-1,1):
         
        # Make a Lorenz forecast
        synthetic_truth[t+1,:,:] = rk4(
            Z           = copy.copy(synthetic_truth[t,:,:]),
            fun         = lorenz_dynamics,
            t           = 0,
            dt          = dt/2,
            nt          = 2)
        
    # Remove the unnecessary particle index
    synthetic_truth     = synthetic_truth[:,0,:]
    
    # Discard the spinup
    synthetic_truth     = synthetic_truth[1000:,...]
        
    # Create perturbed observations from the synthetic truth
    observations        = copy.copy(synthetic_truth) + scipy.stats.norm.rvs(
        scale   = obs_sd, 
        size    = synthetic_truth.shape)
    
    # =====================================================================
    # Create the filter reference
    # =====================================================================
    
    # Reset the random seed
    np.random.seed(0)
    
    # We base all smoothers on a linear, sparse EnTF. We only have to specify 
    # the lower three map components, because we only want to condition.
    nonmonotone_filter  = [
        [[],[0]],
        [[],[1]],
        [[],[1],[2]]]

    # Linear monotone map component terms
    monotone_filter     = [
        [[1]],
        [[2]],
        [[3]]]

    # -------------------------------------------------------------------------
    # Prepare the stochastic map filtering
    # -------------------------------------------------------------------------
    
    N           = 1000
    
    # Initialize the array for the analysis
    X_a         = np.zeros((T,N,D))
    X_a[0,:,:]  = scipy.stats.norm.rvs(
        loc     = copy.copy(synthetic_truth[0,:]),
        size    = (N,D))
    
    # Initialize the array for the forecast
    X_f         = copy.copy(X_a)
    
    # Initialize the arrays for the simulated observations
    Y_sim       = np.zeros((T,N,O))
    
    # Initialize the list for the RMSE
    RMSE_list           = []
    RMSE_list_reference = []
    
    map_input = np.column_stack((
        X_f[0,:,0][:,np.newaxis],   # First O dimensions: simulated observations
        X_f[0,:,:]))    # Next D dimensions: predicted states
    
    if "tm" in globals():
        del tm

    # Parameterize the transport map
    tm     = transport_map(
        monotone                = monotone_filter,                 # Our monotone map component specification
        nonmonotone             = nonmonotone_filter,              # Our nonmonotone map component specification
        X                       = map_input,                # The dummy map input
        polynomial_type         = "probabilist's hermite",  # Polynomial type used for map component terms; doesn't really matter in the linear case
        monotonicity            = "separable monotonicity", # We enforce monotonicity through constrained optimization, not integration
        standardize_samples     = True,                     # Standardize the samples before training the map - should almost always be True
        verbose                 = False)                    # Do not print optimization updates

    # Start the filtering
    # Now handle the forward filtering pass
    for t in np.arange(0,T,1):
        
        # Copy the forecast into the analysis array
        X_a[t,:,:]  = copy.copy(X_f[t,:,:])
    
        # Start the time
        start   = time.time()
    
        # Assimilate the observations one at a time
        for idx,perm in enumerate([[0,1,2],[1,0,2],[2,1,0]]):
            
            # Simulate observations
            Y_sim[t,:,idx] = copy.copy(X_a[t,:,idx]) + \
                scipy.stats.norm.rvs(
                    loc     = 0,
                    scale   = obs_sd,
                    size    = X_a[t,:,idx].shape)
                
            # Create the uninflated map input
            map_input = copy.copy(np.column_stack((
                Y_sim[t,:,idx][:,np.newaxis],   # First O dimensions: simulated observations
                X_a[t,:,:][:,perm])))    # Next D dimensions: predicted states
                
            # Reset the transport map with the new values
            tm.reset(copy.copy(map_input))
            
            # Start optimizing the transport map
            tm.optimize()

            # Once the map is optimized, use it to convert the samples to samples from
            # the reference distribution X
            norm_samples = tm.map(copy.copy(map_input))
            
            # Create an array with the observations
            X_star = np.repeat(
                a       = observations[t,idx].reshape((1,1)),
                repeats = N, 
                axis    = 0)
            
            # Apply the inverse map to obtain samples from (Y,Z_a)
            ret = tm.inverse_map(
                X_star      = X_star,
                Z           = norm_samples)
            
            # Undo the permutation
            ret = ret[:,perm] #[perm[i] for i in perm]
            
            # Ssave the result in the analysis array
            X_a[t,...]  = copy.copy(ret)
            
        # Stop the clock and print optimization time
        end     = time.time()
        # print('Optimization took '+str(end-start)+' seconds.')
            
        # Calculate RMSE
        RMSE = (np.mean(X_a[t,...],axis=0) - synthetic_truth[t,:])**2
        RMSE = np.mean(RMSE)
        RMSE = np.sqrt(RMSE)
        RMSE_list.append(RMSE)
        
        # After the analysis step, make a forecast to the next timestep
        if t < T-1:
            
            # Make a Lorenz forecast
            X_f[t+1,:,:] = rk4(
                Z           = copy.copy(X_a[t,:,:]),
                fun         = lorenz_dynamics,
                t           = 0,
                dt          = dt/2,
                nt          = 2)
            
            # After the deterministic forecast, add the forecast error
            X_f[t+1,:,:] += scipy.stats.norm.rvs(
                loc     = 0,
                scale   = mod_sd,
                size    = X_f[t+1,:,:].shape)


    # =====================================================================
    # Create the EnRTS
    # =====================================================================
            
    np.random.seed(0)

    # Start the EnRTS
    X_EnRTS     = copy.copy(X_a)
    projector   = np.identity(N) - np.ones((N,N))/N
    RMSE_list   = []
    
    full_start  = time.time()
    
    for t in np.arange(T-2,-1,-1):
        
        # print(t)
        
        # Calculate the anomalies
        A_t_minus   = np.dot(projector,X_a[t,...])
        A_t_plus    = np.dot(projector,X_f[t+1,...])
        
        # Calculate the gain term
        gain    = np.linalg.multi_dot((
            A_t_minus.T,
            A_t_plus,
            np.linalg.inv(
                np.dot(
                    A_t_plus.T,
                    A_t_plus))))
        
        # Calculate the signal term
        signal  = X_f[t+1,...] - X_EnRTS[t+1,...]
        
        # Update the Smoothing term
        X_EnRTS[t,...] -= np.dot(
            gain,
            signal.T).T
        
    full_end    = time.time()
    
    RMSE_list   = []
    
    for t in range(T):
    
        # Calculate RMSE
        RMSE = (np.mean(X_EnRTS[t,...],axis=0) - synthetic_truth[t,:])**2
        RMSE = np.mean(RMSE)
        RMSE = np.sqrt(RMSE)
        RMSE_list.append(RMSE)
        
    output_dictionary_EnRTS                 = {}
    output_dictionary_EnRTS['RMSE_list']    = RMSE_list
    output_dictionary_EnRTS['X_EnRTS']      = X_EnRTS
    output_dictionary_EnRTS['duration']     = full_end-full_start
    output_dictionary_EnRTS['synthetic_truth']     = synthetic_truth
    
    # Write the result dictionary to a file
    pickle.dump(output_dictionary_EnRTS,  open(
        'EnRTS_smoother_N='+str(N).zfill(4)+'.p','wb'))
    
    del output_dictionary_EnRTS
    
    
    # =====================================================================
    # Single-pass Backward Smoothing
    # =====================================================================
    
    np.random.seed(0)

    nonmonotone = [
        [[],[0],[1],[2]],
        [[],[0],[1],[2],[3]],
        [[],[0],[1],[2],[3],[4]]]
    
    monotone        = [
        [[3]],
        [[4]],
        [[5]]]
    
    # -------------------------------------------------------------------------
    # Prepare the backwards smoother
    # -------------------------------------------------------------------------

    # Initialize the array for the analysis
    X_s         = np.zeros((T,N,D))
    X_s         = copy.copy(X_a[:,:,:])
    
    # Initialize the list for the RMSE
    RMSE_list   = []
    
    # Calculate the RMSE
    RMSE = (np.mean(X_s[-1,:,:],axis=0) - synthetic_truth[-1,:])**2
    RMSE = np.mean(RMSE)
    RMSE = np.sqrt(RMSE)
    RMSE_list.append(RMSE)
    
    # Create a copy of the original smoothing data for map construction
    X_s_orig    = copy.copy(X_s)
    
    # Create the transport map input
    map_input = copy.copy(np.column_stack((
        X_f[1,:,:],   # First D dimensions: predicted states
        X_a[0,...])))    # Next D dimensions: filtered states
    
    if "tm" in globals():
        del tm
        
    # Parameterize the transport map
    tm     = transport_map(
        monotone                = monotone,                 # Our monotone map component specification
        nonmonotone             = nonmonotone,              # Our nonmonotone map component specification
        X                       = map_input,                # The dummy map input
        polynomial_type         = "probabilist's hermite",  # Polynomial type used for map component terms; doesn't really matter in the linear case
        monotonicity            = "separable monotonicity", # We enforce monotonicity through constrained optimization, not integration
        standardize_samples     = True,                     # Standardize the samples before training the map - should almost always be True
        verbose                 = False)                    # Do not print optimization updates
    
    full_start  = time.time()
    
    # Within a smoothing period, make a backwards smoothing pass for smoothing_period_length
    for t in np.arange(
            T-2,    # Start from the penultimate time T
            -1,     # Go to time 0)
            -1):    # Backwards
        
        # Interrupt the algorithm, if desired.
        if 'stop.txt' in os.listdir(root_directory):
            raise Exception
        
        # Create the transport map input
        map_input = copy.copy(np.column_stack((
            X_f[t+1,:,:],   # First D dimensions: predicted states
            X_a[t,...])))    # Next D dimensions: filtered states
        
        # Reset the map
        tm.reset(copy.copy(map_input))
        
        # Start optimizing the transport map
        start   = time.time()
        tm.optimize()
        end     = time.time()
        
        # Once the map is optimized, use it to convert the samples to samples from
        # the reference distribution X
        norm_samples = tm.map(map_input)
        
        # Now condition on the previous smoothing distribution
        X_star = copy.copy(X_s[t+1,...])
        
        # Then invert the map
        ret = tm.inverse_map(
            X_star      = X_star,
            Z           = norm_samples) # Only necessary when heuristic is deactivated

        # Copy the results into the smoothing array
        X_s[t,...]    = copy.copy(ret)
        
        # Calculate the RMSE
        RMSE = (np.mean(X_s[t,:,:],axis=0) - synthetic_truth[t,:])**2
        RMSE = np.mean(RMSE)
        RMSE = np.sqrt(RMSE)
        RMSE_list.append(RMSE)

    full_end    = time.time()
    
    RMSE_list   = []
    
    for t in range(T):
    
        # Calculate RMSE
        RMSE = (np.mean(X_s[t,...],axis=0) - synthetic_truth[t,:])**2
        RMSE = np.mean(RMSE)
        RMSE = np.sqrt(RMSE)
        RMSE_list.append(RMSE)

        
    # Store the results in the output dictionary
    output_dictionary_TM_BWS    = {
        'X_s'                   : X_s,
        'RMSE_list'             : RMSE_list,
        'duration'              : full_end-full_start,
        'synthetic_truth'       : synthetic_truth}

    # Write the result dictionary to a file
    pickle.dump(output_dictionary_TM_BWS,  open(
        'TM_BW_smoother_N='+str(N).zfill(4)+'.p','wb') )
        
    del output_dictionary_TM_BWS
    
    #%%
    
    # Go through all random seeds
    for rep in np.arange(0,repeats,1):
        
        # Set the random seed
        np.random.seed(rep)
    
        # Create the synthetic reference array, initiate it with Gaussian samples
        synthetic_truth         = np.zeros((1000+T,1,D))
        synthetic_truth[0,0,:]  = scipy.stats.norm.rvs(size=3)
        
        # Run through all timesteps plus spinup
        for t in np.arange(0,1000+T-1,1):
             
            # Make a Lorenz forecast
            synthetic_truth[t+1,:,:] = rk4(
                Z           = copy.copy(synthetic_truth[t,:,:]),
                fun         = lorenz_dynamics,
                t           = 0,
                dt          = dt/2,
                nt          = 2)
            
        # Remove the unnecessary particle index
        synthetic_truth     = synthetic_truth[:,0,:]
        
        # Discard the spinup
        synthetic_truth     = synthetic_truth[1000:,...]
            
        # Create observations by perturbing the syntehtic truth
        observations        = copy.copy(synthetic_truth) + scipy.stats.norm.rvs(
            scale   = obs_sd, 
            size    = synthetic_truth.shape)
        
        # Run through all ensemble sizes
        for N in Ns:
    
            # =====================================================================
            # Create the filter reference
            # =====================================================================
            
            # Reset the random seed
            np.random.seed(rep)
        
            # Initialize the output dictionary
            output_dictionary   = {
                'N'             : N}
            
            # Nonmonotone terms for the linear sparse EnTF
            nonmonotone_filter  = [
                [[],[0]],
                [[],[1]],
                [[],[1],[2]]]
        
            # Same, but monotone
            monotone_filter     = [
                [[1]],
                [[2]],
                [[3]]]
        
            # -------------------------------------------------------------------------
            # Prepare the stochastic map filtering
            # -------------------------------------------------------------------------
            
            # Initialize the array for the analysis
            X_a         = np.zeros((T,N,D))
            X_a[0,:,:]  = scipy.stats.norm.rvs(
                loc     = copy.copy(synthetic_truth[0,:]),
                size    = (N,D))
            
            # Initialize the array for the forecast
            X_f         = copy.copy(X_a)
            
            # Initialize the arrays for the simulated observations
            Y_sim       = np.zeros((T,N,O))
            
            # Initialize the list for the RMSE and time
            RMSE_list           = []
            time_list           = []
            
            # Create a dummy map input; we will replace it later on
            map_input = np.column_stack((
                X_f[0,:,0][:,np.newaxis],   # First O dimensions: simulated observations
                X_f[0,:,:]))    # Next D dimensions: predicted states
            
            # Remove any existing map objects
            if "tm" in globals():
                del tm
        
            # Parameterize the transport map
            tm     = transport_map(
                monotone                = monotone_filter,          # Our monotone map component specification
                nonmonotone             = nonmonotone_filter,       # Our nonmonotone map component specification
                X                       = map_input,                # The dummy map input
                polynomial_type         = "probabilist's hermite",  # Polynomial type used for map component terms; doesn't really matter in the linear case
                monotonicity            = "separable monotonicity", # We enforce monotonicity through constrained optimization, not integration
                standardize_samples     = True,                     # Standardize the samples before training the map - should almost always be True
                verbose                 = False)                    # Do not print optimization updates
        
            # Start the timer for the full filtering run
            full_start  = time.time()
            
            # Print an update 
            print("EnTF (sparse)         : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
        
            # Start the filtering
            # Now handle the forward filtering pass
            for t in np.arange(0,T,1):
                
                # Interrupt the algorithm, if desired.
                if 'stop.txt' in os.listdir(root_directory):
                    raise Exception
                    
                # Copy the forecast into the analysis array
                X_a[t,:,:]  = copy.copy(X_f[t,:,:])
            
                # Start the time
                start   = time.time()
            
                # Assimilate the observations one at a time
                for idx,perm in enumerate([[0,1,2],[1,0,2],[2,1,0]]):
                    
                    # Simulate observations
                    Y_sim[t,:,idx] = copy.copy(X_a[t,:,idx]) + \
                        scipy.stats.norm.rvs(
                            loc     = 0,
                            scale   = obs_sd,
                            size    = X_a[t,:,idx].shape)
                        
                    # Create the map input
                    map_input = copy.copy(np.column_stack((
                        Y_sim[t,:,idx][:,np.newaxis],   # First dimension: simulated observation
                        X_a[t,:,:][:,perm])))           # Next D dimensions: permuted states
                        
                    # Reset the transport map with the new values
                    tm.reset(map_input)
                    
                    # Start optimizing the transport map
                    tm.optimize()
                    
                    # Composite map -------------------------------------------
        
                    # Once the map is optimized, use it to convert the samples to samples from
                    # the reference distribution X
                    norm_samples = tm.map(copy.copy(map_input))
                    
                    # Create an array with the observations
                    X_precalc = np.repeat(
                        a       = observations[t,idx].reshape((1,1)),
                        repeats = N, 
                        axis    = 0)
                    
                    # Apply the inverse map to obtain samples from (Y,Z_a)
                    ret = tm.inverse_map(
                        X_star  = X_precalc,
                        Z       = norm_samples)
                    
                    # ---------------------------------------------------------
                    
                    # Undo the permutation
                    ret = ret[:,perm]
                    
                    # Ssave the result in the analysis array
                    X_a[t,...]  = copy.copy(ret)
                    
                # Stop the clock
                end         = time.time()
                time_list   .append(end-start)
                    
                # Calculate RMSE
                RMSE = (np.mean(X_a[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
                # After the analysis step, make a forecast to the next timestep
                if t < T-1:
                    
                    # Make a Lorenz forecast
                    X_f[t+1,:,:] = rk4(
                        Z           = copy.copy(X_a[t,:,:]),
                        fun         = lorenz_dynamics,
                        t           = 0,
                        dt          = dt/2,
                        nt          = 2)
                    
                    # After the deterministic forecast, add the forecast error
                    X_f[t+1,:,:] += scipy.stats.norm.rvs(
                        loc     = 0,
                        scale   = mod_sd,
                        size    = X_f[t+1,:,:].shape)
        
            # Time the end of the full series
            full_end    = time.time()
            
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
            
            # Store the results in the output dictionary
            output_dictionary['X_f']        = X_f
            output_dictionary['X_a']        = X_a
            output_dictionary['Y_sim']      = Y_sim
            output_dictionary['RMSE_list']  = RMSE_list
            output_dictionary['time_list']  = time_list
            output_dictionary['duration']   = full_end-full_start
            
            # Write the result dictionary to a file
            pickle.dump(output_dictionary,  open(
                'EnTF_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p','wb'))
                    
            #%%
            
            # =====================================================================
            # Create the EnKS
            # =====================================================================
                    
            # Reset the random seed
            np.random.seed(rep)
        
            # Initialize the output dictionary
            output_dictionary_EnKS  = {
                'N'             : N}
            
            # Initialize an array for the smoothing samples
            X_EnKS          = np.zeros((T,N,D))
            
            # Create a matrix for the lagged RMSE quantiles
            X_s_q       = np.zeros((T,maxlag+1,5))*np.nan # Time steps, lag, quantiles 05-25-50-75-95
            
            # This projector subtracts the ensemble mean from the samples
            projector   = np.identity(N) - np.ones((N,N))/N
            
            # Start the clock for the full smoother
            full_start  = time.time()
                    
            # Copy the filtering array for the EnKS
            X_EnKS      = copy.copy(X_a)
            
            # Print an update 
            print("EnKS                  : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
        
            # Go through all time steps
            for t in range(T):
                
                # Extract the observation prediction anomalies
                A_Y_sim = np.dot(
                    projector,
                    Y_sim[t,...])
                
                # Apply the EnKS update to the past few states, up to maxlag
                for s in np.arange(np.maximum(0,t-maxlag-1),t,1):
                    
                    # Calculate the anomalies of the smoothing matrix
                    A_s     = np.dot(
                        projector,
                        X_EnKS[s,...])
                    
                    # Calculate the gain; the 1/N term cancels out
                    gain    = np.linalg.multi_dot((
                        A_s.T,
                        A_Y_sim,
                        np.linalg.inv(
                            np.dot(
                                A_Y_sim.T,
                                A_Y_sim))))
                    
                    # Calculate the signal
                    signal  = Y_sim[t,...] - observations[t,:][np.newaxis,:]
                    
                    # Update the Smoothing term
                    X_EnKS[s,...] -= np.dot(
                        gain,
                        signal.T).T
                    
                # We don't want to store all intermediate smoothing ensembles,
                # instead we just store selected quantiles
                for i,s in enumerate(np.arange(t,np.maximum(-1,t-1-maxlag),-1)):
                
                    # Calculate the RMSE of each sample
                    RMSEi   = np.sqrt(np.mean((X_EnKS[s,:,:] - synthetic_truth[s,:][np.newaxis,:])**2,axis=-1))
                
                    # Calculate and store the RMSE quantiles
                    X_s_q[t,i,0]     = np.quantile(RMSEi,   q = 0.05)
                    X_s_q[t,i,1]     = np.quantile(RMSEi,   q = 0.25)
                    X_s_q[t,i,2]     = np.quantile(RMSEi,   q = 0.50)
                    X_s_q[t,i,3]     = np.quantile(RMSEi,   q = 0.75)
                    X_s_q[t,i,4]     = np.quantile(RMSEi,   q = 0.95)
                
            # At the end of the smoothing pass, calculate all final RMSE values
            RMSE_list   = []
            
            # Calculate it for every marginal
            for t in range(T):
            
                # Calculate RMSE
                RMSE = (np.mean(X_EnKS[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
            
            # Stop the time counter
            full_end    = time.time()
            
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                
            # Store the results
            output_dictionary_EnKS['RMSE_list']     = RMSE_list
            output_dictionary_EnKS['X_s_q']         = X_s_q
            output_dictionary_EnKS['duration']      = full_end-full_start
            
            # Write the result dictionary to a file
            pickle.dump(output_dictionary_EnKS,  open(
                'EnKS_smoother_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p','wb'))
            
            # Delete the dictionary from working memory
            del output_dictionary_EnKS
                    
                    
            #%%
            
            # =====================================================================
            # Create the EnRTS
            # =====================================================================
                    
            # Reset the random seed
            np.random.seed(rep)
        
            # Initialize the output dictionary
            output_dictionary_EnRTS   = {
                'N'             : N}
            
            # Copy the filtering ensembles
            X_EnRTS     = copy.copy(X_a)

            # Start the clock
            full_start  = time.time()
            
            # Print an update 
            print("EnRTSS (single-pass)  : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
            
            # Move backwards through each time step
            for t in np.arange(T-2,-1,-1):
                
                # Calculate the anomalies
                A_t_minus   = np.dot(projector,X_a[t,...])
                A_t_plus    = np.dot(projector,X_f[t+1,...])
                
                # Calculate the gain; the 1/N term cancels out
                gain    = np.linalg.multi_dot((
                    A_t_minus.T,
                    A_t_plus,
                    np.linalg.inv(
                        np.dot(
                            A_t_plus.T,
                            A_t_plus))))
                
                # Calculate the signal
                signal  = X_f[t+1,...] - X_EnRTS[t+1,...]
                
                # Update the Smoothing term
                X_EnRTS[t,...] -= np.dot(
                    gain,
                    signal.T).T
                
            
            # At the end of the time series, stop the clock
            full_end    = time.time()
            
            # Re-calculate the RMSE list
            RMSE_list   = []
            
            # Go through all time steps
            for t in range(T):
            
                # Calculate RMSE
                RMSE = (np.mean(X_EnRTS[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                
            # Write the results into the dictionary
            output_dictionary_EnRTS['RMSE_list']    = RMSE_list
            output_dictionary_EnRTS['duration']     = full_end-full_start
            
            # Write the result dictionary to a file
            pickle.dump(output_dictionary_EnRTS,  open(
                'EnRTS_smoother_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p','wb'))
            
            # Delete the dictionary from working memory
            del output_dictionary_EnRTS
            
            
            #%%
            
            # =====================================================================
            # Create the multi-pass EnRTS
            # =====================================================================
                    
            # Reset the random seed
            np.random.seed(rep)
        
            # Initialize the output dictionary
            output_dictionary_EnRTS_mp   = {
                'N'             : N}
            
            # Create a matrix for the lagged RMSE quantiles
            X_s_q       = np.zeros((T,maxlag+1,5))*np.nan # Time steps, lag, quantiles 05-25-50-75-95
                    
            # Copy the filtering ensembles
            X_EnRTS_mp  = copy.copy(X_a)
        
            # Start the clock
            full_start  = time.time()
            
            # Print an update 
            print("EnRTSS (multi-pass)   : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
        
            # Go through all timesteps
            for t in range(T):
        
                # For the multi-pass EnRTSS, we must base future simulations off
                # previous smoothing passes; create a local copy for that purpose
                X_EnRTS_mp_orig = copy.copy(X_EnRTS_mp)
                
                # Go backwards through all timesteps
                for s in np.arange(t-1,np.maximum(-1,t-1-maxlag),-1):#np.arange(t-1,-1,-1):
                
                    # In the first operation, the training samples are based off
                    # the filtering ensembles
                    if s == t-1:
                    
                        # Calculate the anomalies
                        A_t_minus   = np.dot(projector,X_a[s,...])
                        A_t_plus    = np.dot(projector,X_f[s+1,...])
                        
                        # Calculate the gain term
                        gain    = np.linalg.multi_dot((
                            A_t_minus.T,
                            A_t_plus,
                            np.linalg.inv(
                                np.dot(
                                    A_t_plus.T,
                                    A_t_plus))))
                        
                        # Calculate the signal term
                        signal  = X_f[s+1,...] - X_EnRTS_mp[s+1,...]
                        
                    # In all later operations, the trainings samples are based
                    # off a previous smoothing ensemble
                    else:
                        
                        # Calculate the anomalies
                        A_t_minus   = np.dot(projector,X_EnRTS_mp_orig[s,...])
                        A_t_plus    = np.dot(projector,X_EnRTS_mp_orig[s+1,...])
                        
                        # Calculate the gain term
                        gain    = np.linalg.multi_dot((
                            A_t_minus.T,
                            A_t_plus,
                            np.linalg.inv(
                                np.dot(
                                    A_t_plus.T,
                                    A_t_plus))))
                        
                        # Calculate the signal term
                        signal  = X_EnRTS_mp_orig[s+1,...] - X_EnRTS_mp[s+1,...]
                    
                    # Update the Smoothing term
                    X_EnRTS_mp[s,...] -= np.dot(
                        gain,
                        signal.T).T
                    
                # Calculate the intermediate smoothing marginal quantiles
                for i,s in enumerate(np.arange(t,np.maximum(-1,t-1-maxlag),-1)):
                
                    # Calculate the individual RMSEs of all samples
                    RMSEi   = np.sqrt(np.mean((X_EnRTS_mp[s,:,:] - synthetic_truth[s,:][np.newaxis,:])**2,axis=-1))
                
                    # Store their quantiles
                    X_s_q[t,i,0]     = np.quantile(RMSEi,   q = 0.05)
                    X_s_q[t,i,1]     = np.quantile(RMSEi,   q = 0.25)
                    X_s_q[t,i,2]     = np.quantile(RMSEi,   q = 0.50)
                    X_s_q[t,i,3]     = np.quantile(RMSEi,   q = 0.75)
                    X_s_q[t,i,4]     = np.quantile(RMSEi,   q = 0.95)
                  
            # Re-simulate RMSE
            RMSE_list   = []
            
            # Go through all timesteps
            for t in range(T):
            
                # Calculate RMSE
                RMSE = (np.mean(X_EnRTS_mp[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                
            # Stop the clock
            full_end    = time.time()
                
            # Write the results into the dictionary
            output_dictionary_EnRTS_mp['RMSE_list']     = RMSE_list
            output_dictionary_EnRTS_mp['X_s_q']         = X_s_q
            output_dictionary_EnRTS_mp['duration']      = full_end-full_start
            
            # Write the result dictionary to a file
            pickle.dump(output_dictionary_EnRTS_mp,  open(
                'EnRTS_mp_smoother_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p','wb'))
            
            # Delete the dictionary from working memory
            del output_dictionary_EnRTS_mp
            
            """
            
            # raise Exception #-marked-
                    
            #%%
        
            # =====================================================================
            # Single-pass Backward Smoothing
            # =====================================================================
            
            # Reset the random seed
            np.random.seed(rep)
        
            # Nonmonotone terms
            nonmonotone = [
                [[],[0],[1],[2]],
                [[],[0],[1],[2],[3]],
                [[],[0],[1],[2],[3],[4]]]
            
            # Monotone terms
            monotone        = [
                [[3]],
                [[4]],
                [[5]]]
            
            # -------------------------------------------------------------------------
            # Prepare the backwards smoother
            # -------------------------------------------------------------------------
        
            # Initialize the array for the analysis
            X_s         = np.zeros((T,N,D))
            X_s         = copy.copy(X_a[:,:,:])
            
            # Initialize the list for the RMSE
            RMSE_list   = []
            
            # Calculate the RMSE
            RMSE = (np.mean(X_s[-1,:,:],axis=0) - synthetic_truth[-1,:])**2
            RMSE = np.mean(RMSE)
            RMSE = np.sqrt(RMSE)
            RMSE_list.append(RMSE)
            
            # Create a copy of the original smoothing data for map construction
            X_s_orig    = copy.copy(X_s)
            
            # Create the transport map input
            map_input = copy.copy(np.column_stack((
                X_f[1,:,:],   # First D dimensions: predicted states
                X_a[0,...])))    # Next D dimensions: filtered states
            
            if "tm" in globals():
                del tm
                
            # Parameterize the transport map
            tm     = transport_map(
                monotone                = monotone,                 # Our monotone map component specification
                nonmonotone             = nonmonotone,              # Our nonmonotone map component specification
                X                       = map_input,                # The dummy map input
                polynomial_type         = "probabilist's hermite",  # Polynomial type used for map component terms; doesn't really matter in the linear case
                monotonicity            = "separable monotonicity", # We enforce monotonicity through constrained optimization, not integration
                standardize_samples     = True,                     # Standardize the samples before training the map - should almost always be True
                verbose                 = False)                    # Do not print optimization updates
            
            # raise Exception
            
            full_start  = time.time()
            
            # Print an update 
            print("EnTS (BW, single-pass): seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
            
            # Within a smoothing period, make a backwards smoothing pass for smoothing_period_length
            for t in np.arange(
                    T-2,    # Start from the penultimate time T
                    -1,     # Go to time 0)
                    -1):    # Backwards
                
                # Create the transport map input
                map_input = copy.copy(np.column_stack((
                    X_f[t+1,:,:],   # First D dimensions: predicted states
                    X_a[t,...])))    # Next D dimensions: filtered states
                
                # Reset the map
                tm.reset(map_input)
                
                # Start optimizing the transport map
                tm.optimize()
                
                # Once the map is optimized, use it to convert the samples to samples from
                # the reference distribution X
                norm_samples = tm.map(map_input)
                
                # Now condition on the previous smoothing distribution
                X_star = copy.copy(X_s[t+1,...])
                
                # Then invert the map
                ret = tm.inverse_map(
                    X_star      = X_star,
                    Z           = norm_samples) # Only necessary when heuristic is deactivated
        
                # Copy the results into the smoothing array
                X_s[t,...]    = copy.copy(ret)
                
                # Calculate the RMSE
                RMSE = (np.mean(X_s[t,:,:],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
        
            
            full_end    = time.time()
            
            RMSE_list   = []
            
            for t in range(T):
            
                # Calculate RMSE
                RMSE = (np.mean(X_s[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
        
            if N in full_output:
                
                # Store the results in the output dictionary
                output_dictionary_TM_BWS    = {
                    'X_s'                   : X_s,
                    'RMSE_list'             : RMSE_list,
                    'duration'              : full_end-full_start}
                
            else:
                
                # Store the results in the output dictionary
                output_dictionary_TM_BWS    = {
                    'RMSE_list'             : RMSE_list,
                    'duration'              : full_end-full_start}
        
            # Write the result dictionary to a file
            pickle.dump(output_dictionary_TM_BWS,  open(
                'TM_BW_smoother_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p','wb') )
                
            del output_dictionary_TM_BWS
            
            #%%
        
            # =====================================================================
            # Multi-pass backward smoothing
            # =====================================================================
            
            np.random.seed(rep)
        
            nonmonotone = [
                [[],[0],[1],[2]],
                [[],[0],[1],[2],[3]],
                [[],[0],[1],[2],[3],[4]]]
            
            monotone    = [
                [[3]],
                [[4]],
                [[5]]]
            
            # -------------------------------------------------------------------------
            # Prepare the backwards smoother
            # -------------------------------------------------------------------------
        
            # Initialize the array for the analysis
            X_s         = copy.copy(X_a[:,:,:])
            # Z_ss        = np.zeros((T,maxlag+1,N,D))*np.nan
            # Z_ss[0,0,...] = copy.copy(Z_a[0,:,:])
            
            # Create a matrix for the lagged RMSE quantiles
            X_s_q       = np.zeros((T,maxlag+1,5))*np.nan # Time steps, lag, quantiles 05-25-50-75-95
            
            # Initialize the list for the RMSE
            RMSE_list   = []
            
            percentage_outside_hypersphere  = []
            
            # Create a copy of the original smoothing data for map construction
            X_s_orig    = copy.copy(X_s)
            
            # Create the transport map input
            map_input = copy.copy(np.column_stack((
                X_f[1,:,:],   # First D dimensions: predicted states
                X_a[0,...])))    # Next D dimensions: filtered states
            
            if "tm" in globals():
                del tm
                
            
            # Parameterize the transport map
            tm     = transport_map(
                monotone                = monotone,                 # Our monotone map component specification
                nonmonotone             = nonmonotone,              # Our nonmonotone map component specification
                X                       = map_input,                # The dummy map input
                polynomial_type         = "probabilist's hermite",  # Polynomial type used for map component terms; doesn't really matter in the linear case
                monotonicity            = "separable monotonicity", # We enforce monotonicity through constrained optimization, not integration
                standardize_samples     = True,                     # Standardize the samples before training the map - should almost always be True
                verbose                 = False)                    # Do not print optimization updates
            
            full_start  = time.time()
            
            # Print an update 
            print("EnTS (BW, multi-pass) : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
            
            for t in np.arange(1,T,1):
                
                # Interrupt the algorithm, if desired.
                if 'stop.txt' in os.listdir(root_directory):
                    raise Exception
                
                # # Print an update message informing us that we are assimilating the t-th 
                # # data point.
                # print('Now smoothing the states at the '+ordinal(t)+' time step.')
                
                X_s_orig    = copy.copy(X_s)
                
                # Reset the RMSE list
                RMSE_list   = []
                
                for s in np.arange(t-1,np.maximum(-1,t-1-maxlag),-1):
                    
                    if s == t-1:
                        
                        # Create the transport map input
                        map_input = copy.copy(np.column_stack((
                            X_f[s+1,:,:],       # First D dimensions: predicted states
                            X_s_orig[s,...])))  # Next D dimensions: filtered states
                        
                    else:
                
                        # Create the transport map input
                        map_input = copy.copy(np.column_stack((
                            X_s_orig[s+1,:,:],  # First D dimensions: predicted states
                            X_s_orig[s,...])))  # Next D dimensions: filtered states
                    
                    # Reset the map
                    tm.reset(map_input)
                    
                    # Start optimizing the transport map
                    tm.optimize()
                    
                    # Once the map is optimized, use it to convert the samples to samples from
                    # the reference distribution X
                    norm_samples = tm.map(map_input)
                    
                    # Now condition on the previous smoothing distribution
                    X_star = copy.copy(X_s[s+1,:,:])
                    
                    # Then invert the map
                    ret = tm.inverse_map(
                        X_star      = X_star,
                        Z           = norm_samples) # Only necessary when heuristic is deactivated
            
                    # Copy the results into the smoothing array
                    X_s[s,...]    = copy.copy(ret)
                    
                
                # Set the first RMSE value
                RMSE_list   = []
                RMSE = (np.mean(X_s[t,:,:],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
                for s in np.arange(t-1,-1,-1):
                    
                    # Calculate the RMSE
                    RMSE = (np.mean(X_s[s,:,:],axis=0) - synthetic_truth[s,:])**2
                    RMSE = np.mean(RMSE)
                    RMSE = np.sqrt(RMSE)
                    RMSE_list.append(RMSE)

                # # Save all smoothing states
                # Z_ss[t,-np.minimum(t+1,maxlag+1):,...]     = copy.copy(Z_s[np.maximum(0,t-maxlag):t+1,...])
                # # copy.copy(Z_s[-np.minimum(t+1,maxlag+1):,...])
                
                for i,s in enumerate(np.arange(t,np.maximum(-1,t-1-maxlag),-1)):
                
                    RMSEi   = np.sqrt(np.mean((X_s[s,:,:] - synthetic_truth[s,:][np.newaxis,:])**2,axis=-1))
                
                    X_s_q[t,i,0]     = np.quantile(RMSEi,   q = 0.05)
                    X_s_q[t,i,1]     = np.quantile(RMSEi,   q = 0.25)
                    X_s_q[t,i,2]     = np.quantile(RMSEi,   q = 0.50)
                    X_s_q[t,i,3]     = np.quantile(RMSEi,   q = 0.75)
                    X_s_q[t,i,4]     = np.quantile(RMSEi,   q = 0.95)
        
                # # Print the RMSE results obtained thus far
                # print(str(s).zfill(4)+'/'+str(t).zfill(4)+' Average ensemble RMSE  (N='+str(N)+') is '+str(RMSE)+ ' | Average so far: '+str(np.mean(RMSE_list)))
                
                # plt.figure()
                # plt.title(str(np.mean(RMSE_list)))
                # plt.plot(RMSE_list)
                # plt.savefig('TM_BW_mp_smoother_RMSE_N='+str(N)+'.png')
                # plt.close('all')
                
            full_end    = time.time()
            
            RMSE_list   = []
            
            for t in range(T):
            
                # Calculate RMSE
                RMSE = (np.mean(X_s[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
        
            # if N in full_output:
                
            #     # Store the results in the output dictionary
            #     output_dictionary_TM_BWS_mp = {
            #         'Z_ss'                  : Z_s_q,
            #         'RMSE_list'             : RMSE_list,
            #         'duration'              : full_end-full_start}
                
            # else:
                
            #     # Store the results in the output dictionary
            #     output_dictionary_TM_BWS_mp = {
            #         'RMSE_list'             : RMSE_list,
            #         'duration'              : full_end-full_start}
        
            # Store the results in the output dictionary
            output_dictionary_TM_BWS_mp = {
                'X_s_q'                 : X_s_q,
                'RMSE_list'             : RMSE_list,
                'duration'              : full_end-full_start}
        
            # # Declare victory
            # print('Computations finished.')
        
            # # Store the results in the output dictionary
            # output_dictionary['Z_s']                = Z_s
            # output_dictionary['Z_ss']               = Z_ss
            # output_dictionary['RMSE_list_smooth']   = RMSE_list
            # output_dictionary['duration']           = full_end-full_start
            # # output_dictionary_smoother['percentage_outside_hypersphere'] = percentage_outside_hypersphere
            
            # Write the result dictionary to a file
            pickle.dump(output_dictionary_TM_BWS_mp,  open(
                'TM_BW_mp_smoother_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p','wb'))
            
            del output_dictionary_TM_BWS_mp
            
            
            
            
            
            #%%
        
            # =====================================================================
            # Joint analysis Smoothing
            # =====================================================================
            
            np.random.seed(rep)
        
            nonmonotone = [
                [[],[0],[1],[2]],
                [[],[0],[1],[2],[3]],
                [[],[0],[1],[2],[3],[4]]]
            
            monotone    = [
                [[3]],
                [[4]],
                [[5]]]
            
            # -------------------------------------------------------------------------
            # Prepare the backwards smoother
            # -------------------------------------------------------------------------
        
            # Initialize the array for the analysis
            X_s         = copy.copy(X_a[:,:,:])
            
            # Create a matrix for the lagged RMSE quantiles
            X_s_q       = np.zeros((T,maxlag+1,5))*np.nan # Time steps, lag, quantiles 05-25-50-75-95
            
            
            
            # Z_ss        = np.zeros((T,maxlag+1,N,D))
            # Z_ss[0,0,...] = copy.copy(Z_a[0,:,:])
            
            # Initialize the list for the RMSE
            RMSE_list   = []
            
            percentage_outside_hypersphere  = []
            
            # Create a copy of the original smoothing data for map construction
            X_s_orig    = copy.copy(X_s)
            
            # Create the transport map input
            map_input = copy.copy(np.column_stack((
                X_f[1,:,:],   # First D dimensions: predicted states
                X_a[0,...])))    # Next D dimensions: filtered states
            
            if "tm" in globals():
                del tm
                
            
            # Parameterize the transport map
            tm     = transport_map(
                monotone                = monotone,                 # Our monotone map component specification
                nonmonotone             = nonmonotone,              # Our nonmonotone map component specification
                X                       = map_input,                # The dummy map input
                polynomial_type         = "probabilist's hermite",  # Polynomial type used for map component terms; doesn't really matter in the linear case
                monotonicity            = "separable monotonicity", # We enforce monotonicity through constrained optimization, not integration
                standardize_samples     = True,                     # Standardize the samples before training the map - should almost always be True
                verbose                 = False)                    # Do not print optimization updates
            
            full_start  = time.time()
            
            # Print an update 
            print("EnTS (JA)             : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
            
            for t in np.arange(1,T,1):
                
                # Interrupt the algorithm, if desired.
                if 'stop.txt' in os.listdir(root_directory):
                    raise Exception
                
                # # Print an update message informing us that we are assimilating the t-th 
                # # data point.
                # print('Now smoothing the states at the '+ordinal(t)+' time step.')
                
                # Reset the RMSE list
                RMSE_list   = []
                
                # # Set the first RMSE value
                # RMSE = (np.mean(Z_s[t,:,:],axis=0) - synthetic_truth[T_spinup+t,:])**2
                # RMSE = np.mean(RMSE)
                # RMSE = np.sqrt(RMSE)
                # RMSE_list.append(RMSE)
                
                for s in np.arange(np.maximum(0,t-maxlag-1),t,1): # Do not smooth the last state, we already updated that one during the filtering pass
                
                    # Create the transport map input
                    map_input = copy.copy(np.column_stack((
                        Y_sim[t,:,:],   # First D dimensions: predicted states
                        X_s[s,...])))    # Next D dimensions: filtered states
                    
                    # Reset the map
                    tm.reset(map_input)
                    
                    # Start optimizing the transport map
                    tm.optimize()
                    
                    # Once the map is optimized, use it to convert the samples to samples from
                    # the reference distribution X
                    norm_samples = tm.map(map_input)
                    
                    # Now condition on the previous smoothing distribution
                    X_star = copy.copy(np.repeat(
                        a       = observations[t,:][np.newaxis,:],
                        repeats = N, 
                        axis    = 0))
                    
                    # Then invert the map
                    ret = tm.inverse_map(
                        X_star      = X_star,
                        Z           = norm_samples) # Only necessary when heuristic is deactivated
            
                    # Copy the results into the smoothing array
                    X_s[s,...]    = copy.copy(ret)
                    
                    # # Calculate the RMSE
                    # RMSE = (np.mean(Z_s[s,:,:],axis=0) - synthetic_truth[T_spinup+s,:])**2
                    # RMSE = np.mean(RMSE)
                    # RMSE = np.sqrt(RMSE)
                    # RMSE_list.append(RMSE)
                    
                for i,s in enumerate(np.arange(t,np.maximum(-1,t-1-maxlag),-1)):
                
                    RMSEi   = np.sqrt(np.mean((X_s[s,:,:] - synthetic_truth[s,:][np.newaxis,:])**2,axis=-1))
                
                    X_s_q[t,i,0]     = np.quantile(RMSEi,   q = 0.05)
                    X_s_q[t,i,1]     = np.quantile(RMSEi,   q = 0.25)
                    X_s_q[t,i,2]     = np.quantile(RMSEi,   q = 0.50)
                    X_s_q[t,i,3]     = np.quantile(RMSEi,   q = 0.75)
                    X_s_q[t,i,4]     = np.quantile(RMSEi,   q = 0.95)
                    
                # Set the first RMSE value
                RMSE_list   = []
                # RMSE = (np.mean(Z_s[t,:,:],axis=0) - synthetic_truth[T_spinup+t,:])**2
                # RMSE = np.mean(RMSE)
                # RMSE = np.sqrt(RMSE)
                # RMSE_list.append(RMSE)
                
                for s in np.arange(0,t+1,1):
                    
                    # Calculate the RMSE
                    RMSE = (np.mean(X_s[s,:,:],axis=0) - synthetic_truth[s,:])**2
                    RMSE = np.mean(RMSE)
                    RMSE = np.sqrt(RMSE)
                    RMSE_list.append(RMSE)
                    
                # Save all smoothing states
                # Z_ss[t,-np.minimum(t+1,maxlag+1):,...]     = copy.copy(Z_s[-np.minimum(t+1,maxlag+1):,...])
                # Z_ss[t,:np.minimum(t+1,maxlag+1),...]     = copy.copy(Z_s[:np.minimum(t+1,maxlag+1),...])
                # Z_ss[t,-np.minimum(t+1,maxlag+1):,...]     = copy.copy(Z_s[np.maximum(0,t-maxlag):t+1,...])
                
                # for i,s in enumerate(np.arange(t,np.maximum(-1,t-1-maxlag),-1)):
                
                #     RMSEi   = np.sqrt(np.mean((Z_s[s,:,:] - synthetic_truth[np.newaxis,:])**2,axis=-1))
                
                #     Z_s_q[t,i,0]     = np.quantile(RMSEi,   q = 0.05)
                #     Z_s_q[t,i,1]     = np.quantile(RMSEi,   q = 0.25)
                #     Z_s_q[t,i,2]     = np.quantile(RMSEi,   q = 0.50)
                #     Z_s_q[t,i,3]     = np.quantile(RMSEi,   q = 0.75)
                #     Z_s_q[t,i,4]     = np.quantile(RMSEi,   q = 0.95)
                
                
                
                # copy.copy(Z_s[np.maximum(0,t-maxlag):t+1,...])
                
                
                # # Print the RMSE results obtained thus far
                # print(str(s).zfill(4)+'/'+str(t).zfill(4)+' Average ensemble RMSE  (N='+str(N)+') is '+str(RMSE)+ ' | Average so far: '+str(np.mean(RMSE_list)))
                
                # plt.figure()
                # plt.title(str(np.mean(RMSE_list)))
                # plt.plot(RMSE_list)
                # plt.savefig('TM_JA_smoother_RMSE_N='+str(N)+'.png')
                # plt.close('all')
        
            # # Declare victory
            # print('Computations finished.')
            
            full_end    = time.time()
        
            RMSE_list   = []
            
            for t in range(T):
            
                # Calculate RMSE
                RMSE = (np.mean(X_s[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
        
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
            
            # if N in full_output:
                
            #     # Store the results in the output dictionary
            #     output_dictionary_TM_JAS = {
            #         'Z_s_q'                 : Z_s_q,
            #         'RMSE_list'             : RMSE_list,
            #         'duration'              : full_end-full_start}
                
            # else:
                
            #     # Store the results in the output dictionary
            #     output_dictionary_TM_JAS = {
            #         'RMSE_list'             : RMSE_list,
            #         'duration'              : full_end-full_start}
            
            # Store the results in the output dictionary
            output_dictionary_TM_JAS = {
                'X_s_q'                 : X_s_q,
                'RMSE_list'             : RMSE_list,
                'duration'              : full_end-full_start}
        
            # # Store the results in the output dictionary
            # output_dictionary['Z_s']                = Z_s
            # output_dictionary['Z_ss']               = Z_ss
            # output_dictionary['RMSE_list_smooth']   = RMSE_list
            # output_dictionary['duration']           = full_end-full_start
            # # output_dictionary_smoother['percentage_outside_hypersphere'] = percentage_outside_hypersphere
            
            # Write the result dictionary to a file
            pickle.dump(output_dictionary_TM_JAS,  open(
                'TM_JA_smoother_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p','wb'))
            
            del output_dictionary_TM_JAS
            
            """
    