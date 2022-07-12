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
    def lorenz_dynamics(t, X, beta=8/3, rho=28, sigma=10):
        
        if len(X.shape) == 1: # Only one particle
        
            dX1ds   = - sigma*X[0] + sigma*X[1]
            dX2ds   = - X[0]*X[2] + rho*X[0] - X[1]
            dX3ds   = X[0]*X[1] - beta*X[2]
            
            dyn     = np.asarray([dX1ds, dX2ds, dX3ds])
            
        else:
            
            dX1ds   = - sigma*X[...,0] + sigma*X[...,1]
            dX2ds   = - X[...,0]*X[...,2] + rho*X[...,0] - X[...,1]
            dX3ds   = X[...,0]*X[...,1] - beta*X[...,2]
    
            dyn     = np.column_stack((dX1ds, dX2ds, dX3ds))
    
        return dyn
    
    # Finds value of y for a given x using step size h
    # and initial value y0 at x0.
    def rk4(X,fun,t=0,dt=1,nt=1):#(x0, y0, x, h):
        
        """
        Parameters
            t       : initial time
            X       : initial states
            fun     : function to be integrated
            dt      : time step length
            nt      : number of time steps
        
        """
        
        # Prepare array for use
        if len(X.shape) == 1: # We have only one particle, convert it to correct format
            X       = X[np.newaxis,:]
            
        # Go through all time steps
        for i in range(nt):
            
            # Calculate the RK4 values
            k1  = fun(t + i*dt,           X);
            k2  = fun(t + i*dt + 0.5*dt,  X + dt/2*k1);
            k3  = fun(t + i*dt + 0.5*dt,  X + dt/2*k2);
            k4  = fun(t + i*dt + dt,      X + dt*k3);
        
            # Update next value
            X   += dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        return X

    # -------------------------------------------------------------------------
    # Set up exercise
    # -------------------------------------------------------------------------
    
    # Define problem dimensions
    O                   = 3 # Observation space dimensions
    D                   = 3 # State space dimensions
    
    # Ensemble sizes
    Ns                  = [50,75,100,150,250,500,1000]
    
    # Set up time
    T                   = 1000
    dt                  = 0.1   # Time step length
    times = np.arange(T)*dt
    
    # Observation error
    obs_sd              = 2
    R                   = np.identity(O)*obs_sd**2
    
    # Number of repeats with different random seeds
    repeats             = 1000
    
    for rep in np.arange(0,repeats,1):
        
        # Set the random seed
        np.random.seed(rep)

        # Create the array for the synthetic reference
        synthetic_truth         = np.zeros((1000+T,1,D))
        
        # Initiate it with standard Gaussian samples
        synthetic_truth[0,0,:]  = scipy.stats.norm.rvs(size=3)
        
        # Simulate the spinup and simulation period
        for t in np.arange(0,1000+T-1,1):
             
            # Make a Lorenz forecast
            synthetic_truth[t+1,:,:] = rk4(
                X           = copy.copy(synthetic_truth[t,:,:]),
                fun         = lorenz_dynamics,
                t           = 0,
                dt          = dt/2,
                nt          = 2)
            
        # Remove the unnecessary particle index
        synthetic_truth     = synthetic_truth[:,0,:]
            
        # Create noisy observations by perturbing the true state
        observations        = copy.copy(synthetic_truth) + scipy.stats.norm.rvs(scale = obs_sd, size = synthetic_truth.shape)
      
        # Discard the spinup
        synthetic_truth     = synthetic_truth[1000:,:]
        observations        = observations[1000:,:]
        
        #%% 
        
        # Repeat this simulation for all ensemble sizee´s
        for N in Ns:
    

            # =================================================================
            # Create the sparse filter reference
            # =================================================================
            
            # Initialize the output dictionary with some preliminary information
            output_dictionary   = {
                'obs_sd'            : obs_sd,
                'N'                 : N,
                'synthetic_truth'   : synthetic_truth,
                'observations'      : observations}
            
            # Define the map. Note that we skip the first map component, as we
            # do not have to define or optimize it if we are only interested in
            # a conditioning operation. Otherwise, we would require four map
            # components, not three as below.
            
            # The nonmonotone part of each map is constant + linear term
            # Note that the second and third map component are conditionally
            # independent of the first argument, the observation
            nonmonotone_filter  = [
                [[],[0]],
                [[],[1]],
                [[],[1],[2]]]
        
            # The monotone part of each map component just depends linearly on
            # the last argument. 
            monotone_filter     = [
                [[1]],
                [[2]],
                [[3]]]
        
            # -----------------------------------------------------------------
            # Prepare the stochastic map filtering
            # -----------------------------------------------------------------
            
            # Reset the random seed again
            np.random.seed(rep)
            
            # Initialize an array for the ensemble
            X_a         = np.zeros((T,N,D))
            
            # Initiate it with standard Gaussian samples centered on the 
            # synthetic truth.
            X_a[0,:,:]  = scipy.stats.norm.rvs(
                loc     = synthetic_truth[0,:],
                size    = (N,D))
            
            # Initialize the array for the simulated observations
            Y_sim       = np.zeros((T,N,O))
            
            # Initialize the list for the RMSE and time
            RMSE_list           = []
            time_list           = []
            
            # Create a dummy map input to build the transport map class; we 
            # replace this later on
            map_input = np.column_stack((
                X_a[0,:,0][:,np.newaxis],   # First O dimensions: simulated observations
                X_a[0,:,:]))    # Next D dimensions: predicted states
            
            # To avoid unforseen code weirdness, delete any pre-existing 
            # transport map object
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
        
            # Print an update 
            print("EnTF (sparse)        : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
        
            # Start filtering, no spinup
            for t in np.arange(0,T,1):
        
                # Start the calculation timer
                start   = time.time()
            
                # Assimilate the observations one at a time
                for idx,perm in enumerate([[0,1,2],[1,0,2],[2,1,0]]):
                    
                    # Simulate observation predictions
                    Y_sim[t,:,idx] = copy.copy(X_a[t,:,:][:,idx]) + \
                        scipy.stats.norm.rvs(
                            loc     = 0,
                            scale   = obs_sd,
                            size    = N)
                        
                    # Create the map input
                    map_input = np.column_stack((
                        Y_sim[t,:,idx][:,np.newaxis],   # First dimension: observation prediction
                        X_a[t,:,:][:,perm]))            # Next D dimensions: permuted states
                        
                    # Reset the transport map with the new values
                    tm.reset(map_input)
                    
                    # Optimize the transport map
                    tm.optimize()
                    
                    # Composite map section -----------------------------------
        
                    # Pushforward, obtain reference samples
                    norm_samples = tm.map(map_input)
                    
                    # Create an array with conditioning values; for DA, this is
                    # just an array with copies of the observation
                    X_star = np.repeat(
                        a       = observations[t,idx].reshape((1,1)),
                        repeats = N, 
                        axis    = 0)
                    
                    # Apply the inverse map to obtain posterior samples
                    ret = tm.inverse_map(
                        X_star      = X_star,
                        Z           = norm_samples)
                    
                    # Composite map section end -------------------------------
                    
                    # Undo the permutation
                    ret = ret[:,perm]
                    
                    # Save the result in the analysis array
                    X_a[t,...]  = copy.copy(ret)
                    
                # Stop the clock and print optimization time
                end         = time.time()
                time_list   .append(end-start)
                    
                # Calculate RMSE from ensemble mean to true state
                RMSE = (np.mean(X_a[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
                # After the analysis step, make a forecast to the next timestep
                if t < T-1:
                    
                    # Make a Lorenz forecast
                    X_a[t+1,:,:] = rk4(
                        X           = copy.copy(X_a[t,:,:]),
                        fun         = lorenz_dynamics,
                        t           = 0,
                        dt          = dt/2,
                        nt          = 2)
                    
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
        
            # Store the results in the output dictionary
            output_dictionary['RMSE_list_sparse']   = copy.copy(RMSE_list)
            output_dictionary['time_list_sparse']   = copy.copy(time_list)
                    
            #%% 
        
            # =====================================================================
            # Create the dense filter reference
            # =====================================================================
            
            # For the dense EnTF, we define the lower three map components for
            # a six-dimensional distribution; that means that our first map
            # component is actually the fourth, and thus depends on the first
            # three entries. We also consider no conditional independence.
            nonmonotone_filter  = [
                [[],[0],[1],[2]],
                [[],[0],[1],[2],[3]],
                [[],[0],[1],[2],[3],[4]]]
        
            # For the monotone parts of the map components, it's business as 
            # usual.
            monotone_filter     = [
                [[3]],
                [[4]],
                [[5]]]
        
            # -------------------------------------------------------------------------
            # Prepare the stochastic map filtering
            # -------------------------------------------------------------------------
            
            # Reset the random seeds
            np.random.seed(rep)
            
            # Initialize the array for the dense filter by perturbing the true
            # state. As before.
            X_a         = np.zeros((T,N,D))
            X_a[0,:,:]  = scipy.stats.norm.rvs(
                loc     = synthetic_truth[0,:],
                size    = (N,D))
            
            # Initialize the arrays for the simulated observations
            Y_sim       = np.zeros((T,N,O))
            
            # Initialize the list for the RMSE and computation time
            RMSE_list           = []
            time_list           = []
            
            # Create a dummy map input
            map_input = np.column_stack((
                X_a[0,:,:],     # First O dimensions: observation predictions
                X_a[0,:,:]))    # Next D dimensions: predicted states
            
            # Delete any pre-existing map objects
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
            
            # Print an update 
            print("EnTF (dense)         : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
            
            # Start filtering
            for t in np.arange(0,T,1):
                    
                # Start the time
                start   = time.time()
            
                # Simulate observation predictions
                Y_sim[t,:,:] = copy.copy(X_a[t,:,:]) + \
                    scipy.stats.norm.rvs(
                        loc     = 0,
                        scale   = obs_sd,
                        size    = X_a[t,:,:].shape)
                    
                # Create the map input
                map_input = np.column_stack((
                    Y_sim[t,:,:],   # First O dimensions: observation predictions
                    X_a[t,:,:]))    # Next D dimensions: states
                    
                # Reset the transport map
                tm.reset(map_input)
                
                # Optimize the map
                tm.optimize()
                
                # Composite map section ---------------------------------------
        
                # Pushforward
                norm_samples = tm.map(copy.copy(map_input))
                
                # Create an array with the observations
                X_star = np.repeat(
                    a       = observations[t,:][np.newaxis,:],
                    repeats = N, 
                    axis    = 0)
                
                # Pullback
                ret = tm.inverse_map(
                    X_star      = X_star,
                    Z           = norm_samples)
                
                # Composite map section end -----------------------------------
                
                # Ssave the result in the analysis array
                X_a[t,...]  = copy.copy(ret)
                    
                # Stop the clock and print optimization time
                end         = time.time()
                time_list   = end-start
                    
                # Calculate RMSE
                RMSE = (np.mean(X_a[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
                # After the analysis step, make a forecast to the next timestep
                if t < T-1:
                    
                    # Make a Lorenz forecast
                    X_a[t+1,:,:] = rk4(
                        X           = copy.copy(X_a[t,:,:]),
                        fun         = lorenz_dynamics,
                        t           = 0,
                        dt          = dt/2,
                        nt          = 2)
                    
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
        
            # Store the results in the output dictionary
            output_dictionary['RMSE_list_dense']  = copy.copy(RMSE_list)
            output_dictionary['time_list_dense']  = copy.copy(time_list)
            
            #%%
            
            # =====================================================================
            # Create the empirical EnKF reference
            # =====================================================================
            
            # Reset the random seed
            np.random.seed(rep)
            
            # Initiate the ensemble. We've seen this before.
            X_EnKF          = np.zeros((T,N,D))
            X_EnKF[0,:,:]   = scipy.stats.norm.rvs(
                loc     = synthetic_truth[0,:],
                size    = (N,D))

            # Lists for RMSE and time, yep, yep.
            RMSE_list   = []
            time_list   = []
            
            # Initialize the arrays for the simulated observations
            Y_sim       = np.zeros((T,N,O))
            
            # This is a projector which subtracts the ensemble mean
            projector   = np.identity(N) - np.ones((N,N))/N

            # Print an update 
            print("EnKF (empirical)     : seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
            
            # Start filtering
            for t in np.arange(0,T,1):
                
                # Star the timer
                start   = time.time()
                
                # Sample observation predictions
                Y_sim[t,...]    = copy.copy(X_EnKF[t,...]) + scipy.stats.norm.rvs(
                    loc     = 0,
                    scale   = obs_sd,
                    size    = X_EnKF[t,:,:].shape)
                
                # Calculate the anomalies
                A_Y_sim = np.dot(
                    projector,
                    Y_sim[t,...])
                A_t     = np.dot(
                    projector,
                    X_EnKF[t,...])
                
                # Calculate the gain, the 1/N factors cancel out
                gain    = np.linalg.multi_dot((
                    A_t.T,
                    A_Y_sim,
                    np.linalg.inv(
                        np.dot(
                            A_Y_sim.T,
                            A_Y_sim))))
                
                # Calculate the signal
                signal  = Y_sim[t,...] - observations[t,:][np.newaxis,:]
                
                # Update the Smoothing term
                X_EnKF[t,...] -= np.dot(
                    gain,
                    signal.T).T
                
                # End the timer
                end     = time.time()
                time_list.append(end-start)
                
                # Calculate RMSE
                RMSE = (np.mean(X_EnKF[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
                if t < T-1:
                
                    # Make a Lorenz forecast
                    X_EnKF[t+1,:,:] = rk4(
                        X           = copy.copy(X_EnKF[t,:,:]),
                        fun         = lorenz_dynamics,
                        t           = 0,
                        dt          = dt/2,
                        nt          = 2)
                    
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                        
            # Store the results in the output dictionary
            output_dictionary['RMSE_list_EnKF_empirical']   = copy.copy(RMSE_list)
            output_dictionary['time_list_EnKF_empirical']   = copy.copy(time_list)
            
            
            #%%
        
            # =====================================================================
            # Create the semi-empirical EnKF reference
            # =====================================================================
            
            # Reset the random seed
            np.random.seed(rep)
            
            # Initialize the ensemble
            X_EnKF          = np.zeros((T,N,D))
            X_EnKF[0,:,:]   = scipy.stats.norm.rvs(
                loc     = synthetic_truth[0,:],
                size    = (N,D))

            # Lists for RMSe and time
            RMSE_list   = []
            time_list   = []
            
            # Initialize the arrays for the simulated observations
            Y_sim       = np.zeros((T,N,O))
            projector   = np.identity(N) - np.ones((N,N))/N
            
            # Print an update 
            print("EnKF (semi-empirical): seed="+str(rep).zfill(4)+" | N="+str(N).zfill(4),end="")
            
            # Start filtering
            for t in np.arange(0,T,1):
                
                # Sample observation predictions
                Y_sim[t,...]    = copy.copy(X_EnKF[t,...]) + scipy.stats.norm.rvs(
                    loc     = 0,
                    scale   = obs_sd,
                    size    = X_EnKF[t,:,:].shape)
                
                # Calculate the anomaly. The semi-empirical EnKF does not require
                # the observation prediction anomalies, it determines those in
                # closed form based on the state anomalies.
                A_t     = np.dot(
                    projector,
                    X_EnKF[t,...])
                
                # Calculate the gain term; we have to keep the 1/N term due to
                # the analytical observation error covariance matrix R
                gain    = np.linalg.multi_dot((
                    A_t.T,
                    A_t/(N-1),
                    np.linalg.inv(
                        np.dot(
                            A_t.T,
                            A_t)/(N-1) + R)))
                
                # Calculate the signal term
                signal  = Y_sim[t,...] - observations[t,:][np.newaxis,:]
                
                # Update the Smoothing term
                X_EnKF[t,...] -= np.dot(
                    gain,
                    signal.T).T
                
                # End the timer
                end     = time.time()
                time_list.append(end-start)
                
                # Calculate RMSE
                RMSE = (np.mean(X_EnKF[t,...],axis=0) - synthetic_truth[t,:])**2
                RMSE = np.mean(RMSE)
                RMSE = np.sqrt(RMSE)
                RMSE_list.append(RMSE)
                
                if t < T-1:
                
                    # Make a Lorenz forecast
                    X_EnKF[t+1,:,:] = rk4(
                        X           = copy.copy(X_EnKF[t,:,:]),
                        fun         = lorenz_dynamics,
                        t           = 0,
                        dt          = dt/2,
                        nt          = 2)
                    
            # Add the mean ensemble RMSE to the output print
            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                        
            # Store the results in the output dictionary
            output_dictionary['RMSE_list_EnKF_semiempirical']   = copy.copy(RMSE_list)
            output_dictionary['time_list_EnKF_semiempirical']   = copy.copy(time_list)
            
            #%%
            
            # Pickle the results for this random seed,
            pickle.dump(
                output_dictionary,
                open('filter_comparison_output_dictionary'+'_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p','wb'))
            

        
            