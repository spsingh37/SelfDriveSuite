## 🤖 SelfDriveSuite: Vehicle Control and Scene Understanding                                                               
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
  <img src="assets/case5_traj_gif.gif" alt="Simulation 3" width="400" />
  <img src="assets/predictions_gif.gif" alt="Simulation 4" width="400" />
</p>

### 🎯 Goal
This a comprehensive project focused on developing algorithms for self-driving cars. This repository encompasses several critical components of autonomous driving technology, including:

- **Adaptive Cruise Control**: Implementing an adaptive cruise controller to maintain safe following distances and speeds.
- **Model Predictive Control (MPC)**: Designing both linear and non-linear MPC for tracking reference trajectories and optimizing vehicle paths.
- **All-Weather Scene Understanding**: Achieving robust real-time, object recognition, detection, and scene segmentation in challenging conditions, such as low visibility, and/pr fast-paced scenario, and/or adverse weather.

## ⚙️ Prerequisites
- Libraries/Frameworks:
    - Numpy
    - Matplotlib
    - Scipy
    - CVXPY (for Convex optimization)
    - CasADi 
    - pytorch

## 🛠️ Test/Demo
- Adaptive Cruise Control
    - Go to the directory 'Vehicle Control\Adaptive Cruise Control', and launch the jupyter notebook
- Trajectory tracking using Linear MPC
    - Go to the directory 'Vehicle Control\Trajectory tracking', and launch the jupyter notebook
- Car Overtaking using Non-Linear MPC
    - Go to the directory 'Trajectory Optimization\CarOvertaking', and launch the jupyter notebook
- Drag Racing using Non-Linear MPC
    - Go to the directory 'Trajectory Optimization\DragRacing', and launch the jupyter notebook
- Image Classification
    - Go to the directory 'Image Classification', and launch the jupyter notebook
- Object Detection
    - Go to the directory 'Object Detection', and run 'inference.py' or 'inference_video.py' following README there.
- Scene Segmentation
    - Go to the directory 'Scene Segmentation', and launch the jupyter notebook

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
<div style="width: 75%; text-align: center">
      <img style="width:75%" src="assets/trajectory.png">
</div>

### 📈 Drag Racing (using Non-Linear MPC)
- Case 1:
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/case1_traj_gif.gif">
</div>
<div style="height: 250%; text-align: center">
      <img style="height: 75px; width: 600px;" src="assets/case1_traj.png">
</div>

- Case 2:
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/case2_traj_gif.gif">
</div>
<div style="height: 250%; text-align: center">
      <img style="height: 100px; width: 600px;" src="assets/case2_traj.png">
</div>

- Case 3:
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/case3_traj_gif.gif">
</div>
<div style="height: 250%; text-align: center">
      <img style="height: 100px; width: 600px;" src="assets/case3_traj.png">
</div>

- Case 4:
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/case4_traj_gif.gif">
</div>
<div style="height: 250%; text-align: center">
      <img style="height: 150px; width: 600px;" src="assets/case4_traj.png">
</div>

- Case 5:
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/case5_traj_gif.gif">
</div>
<div style="height: 250%; text-align: center">
      <img style="height: 150px; width: 750px;" src="assets/case5_traj.png">
</div>

### 📈 Image Classification (Cars vs Person vs None)
- Challenging because low-resolution 32x32 blurry RGB images...still achieved 90% accuracy

<div style="width: 75%; text-align: center">
      <img style="width:75%" src="assets/image_classification.jpg">
</div>

### 📈 Object Detection (Traffic sign Detection)
- Achieved ~48 mAP with an average FPS of 43

<div style="width: 75%; text-align: center">
      <img style="width:75%" src="assets/object_detection_gif.gif">
</div>

### 📈 Scene segmentation (Comprising 14 classes from urban scenario + background class)
- Achieved 81.2 mIoU

<p align="center">
  <img src="assets/real_gif.gif" alt="Simulation 1" width="400" />
  <img src="assets/predictions_gif.gif" alt="Simulation 2" width="400" />
</p>



