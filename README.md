## 🤖 SelfDriveSuite
Note:  This project is a culmination of my work for the **"ROB 535: Self-driving Cars: Perception to Control"** course, conducted from September to December 2023. The course provided a solid foundation in autonomous driving technologies, from perception to control, and this repository represents the integration and application of those concepts.
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/acc_vehicle_simulation.gif">
</div>

### 🎯 Goal
This a comprehensive project focused on developing algorithms for self-driving cars. This repository encompasses several critical components of autonomous driving technology, including:

- **Adaptive Cruise Control**: Implementing an adaptive cruise controller to maintain safe following distances and speeds.
- **Model Predictive Control (MPC)**: Designing both linear and non-linear MPC for tracking reference trajectories and optimizing vehicle paths.
- **All-Weather Scene Understanding**: Achieving robust object recognition and scene segmentation in challenging conditions, such as low visibility and adverse weather.

## ⚙️ Prerequisites
- Python libraries:
    - Numpy
    - Matplotlib
    - Scipy
    - CVXPY (for Convex optimization)

## 🛠️ Test/Demo
- Adaptive Cruise Control
    - Go to the directory 'Vehicle Control\Adaptive Cruise Control', and launch the jupyter notebook
- Trajectory tracking using Linear MPC
    - Go to the directory 'Vehicle Control\Trajectory tracking', and launch the jupyter notebook

## 📊 Results
### 📈 Adaptive Cruise Control
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/acc_vehicle_simulation.gif">
</div>
<div style="width: 60%; text-align: center;">
      <img style="width:100%" src="assets/acc_vehicle_velocity.png">
</div>

### 📈 Trajectory tracking (using Linear MPC)
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/vehicle_traj_track.gif">
</div>





