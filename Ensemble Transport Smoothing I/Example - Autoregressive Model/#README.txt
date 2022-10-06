To regenerate our results, please proceed as follows:

1) Run the Python file "run_AR_process.py". Warning: this can take a while, and generates a 2GB output file.

2) Run the Python files "process_mean_and_covariance.py" and "process_signal_and_gain.py". As the name implies, both files process the previously generated output file to calculate the quantities we will plot in the following.

3) Run the Python files "plot_mean.py", "plot_covariance.py", and "plot_signal_vs_gain.py" to generate the corresponding output figures.

The remaining Python files serve the following purposes:

transport_map_138.py
This file contains the triangular transport toolbox we used in this study, and includes everything required to define, optimize, and evaluate a transport map.

ensemble_filters.py
This file contains the function used for the Ensemble Kalman Filter.

ensemble_smoothers.py
This file contains the functions used for the Ensemble Kalman Smoother, the single- and multi-pass Ensemble Rauch-Tung-Striebel Smoother, and the multi-pass Ensemble Forward Smoother.

transport_filters.py
This file contains the function used for the Ensemble Transport Filter.

transport_smoothers.py
This file contains the functions used for the dense Ensemble Transport Smoother, the single- and multi-pass backward Ensemble Transport Smoother, and the multi-pass forward Ensemble Transport Smoother.

