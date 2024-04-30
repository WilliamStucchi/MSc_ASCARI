#Science Robotics Readme
Dependencies: matlab python 3

python packages: casadi numpy tensorflow matplotlib tqdm scipy

#Figure 2 consists of the comparison between experience human race driver and physics-based
controller. This data and analsysis is contained within the Human vs. Machine Folder

#Figure 4 consists of the experimental results from the comparison between NN and physics-based
controller on the VW GTI. This experimental data and analysis is contained with the Control/Experiment
folder. In order to generate the plots contained in this figure, in matlab run Plot_Scirob_Experiment.m

In addition to the experimental data, the real time control code is contained within the Real_Time
folder. Simulation code is also available in Control/Simulation. In order to run a simulation with
both controllers on the oval, run Simulate_Controllers.py in python. This simulation comes with
pretrained weights which are used in the model.

#Figure 5 consists of learning experiments contrasting both the physics-based model and neural
network model on both generated and recorded data.

In order to plot the results from the paper, navigate to results and run plot_results.py

All of the learning data in the paper is generated from running generate_data.py
This both generates the simulated data as well as prepares the experimental data for learning.
The results of running generate data are already stored in Model_Learning/data,
so there is no need to run generate_data.py

In order to learn models, run train_models.py
This will train models for each of the experiments shown in the results.
This will also save the trained models, and overwrite the results in the results folder.

Once again to plot the results, you can run plot_results.py in the results directory.

email nspielbe@stanford.edu for any questions.

