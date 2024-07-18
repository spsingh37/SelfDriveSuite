## 🤖 SelfDriveSuite
Note:  This project is a culmination of my work for the **"ROB 535: Self-driving Cars: Perception to Control"** course, conducted from September to December 2023. The course provided a solid foundation in autonomous driving technologies, from perception to control, and this repository represents the integration and application of those concepts.
<!-- <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/acc_vehicle_simulation.gif">
</div> -->
<!-- <div style="display: flex; justify-content: center;">
    <div style="margin: 10px;">
        <img src="assets/acc_vehicle_simulation.gif" alt="Simulation 1" style="width: 200px;">
    </div>
    <div style="margin: 10px;">
        <img src="assets/vehicle_overtaking.gif" alt="Simulation 2" style="width: 200px;">
    </div>
</div> -->
<p align="center">
  <img src="assets/acc_vehicle_simulation.gif" alt="Simulation 1" width="400" />
  <img src="assets/vehicle_overtaking.gif" alt="Simulation 2" width="400" />
</p>

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
    - CasADi 

## 🛠️ Test/Demo
- Adaptive Cruise Control
    - Go to the directory 'Vehicle Control\Adaptive Cruise Control', and launch the jupyter notebook
- Trajectory tracking using Linear MPC
    - Go to the directory 'Vehicle Control\Trajectory tracking', and launch the jupyter notebook
- Car Overtaking using Non-Linear MPC
    - Go to the directory 'Trajectory Optimization\CarOvertaking', and launch the jupyter notebook

## 📊 Results
### 📈 Adaptive Cruise Control
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/acc_vehicle_simulation.gif">
</div>
<div style="width: 80%; text-align: center;">
      <img style="width:80%" src="assets/acc_vehicle_velocity.png">
</div>

### 📈 Trajectory tracking (using Linear MPC)
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/vehicle_traj_track.gif">
</div>

### 📈 Car Overtaking (using Non-Linear MPC)
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/vehicle_overtaking.gif">
</div>
<div style="width: 60%; text-align: center">
      <img style="width:60%" src="assets/trajectory.png">
</div>





