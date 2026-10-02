# State-Feedback Trot Locomotion Controller for Unitree Go2

A model-based quadruped locomotion control project developed in MuJoCo, focusing on state feedback, gait scheduling, swing-leg trajectory generation, and joint-level tracking control.

## Project Overview

This project implements a trot locomotion controller for the Unitree Go2 quadruped robot in the MuJoCo simulation environment. The control framework integrates robot state estimation, gait scheduling, foothold planning, swing-leg trajectory generation, and joint-level control to achieve forward locomotion with a freely moving base.

The project emphasizes the integration of kinematics, motion planning, and feedback control in a simulated quadruped system. Experimental evaluation focuses on forward displacement, base attitude, velocity response, and joint position tracking errors.

### Key Features

* **Robot Modeling:** Unitree Go2 simulation using MuJoCo.
* **State Feedback:** Robot state information is used to support locomotion control.
* **Gait Scheduling:** Alternating diagonal-leg coordination for trot locomotion.
* **Swing-Leg Control:** Smooth swing-foot trajectory generation.
* **Foothold Planning:** Foot placement adjustment based on locomotion state.
* **Joint-Level Control:** Desired joint positions are tracked through feedback control.
* **Experimental Evaluation:** Quantitative analysis of base motion, attitude, velocity, and joint tracking performance.

### Current Experimental Results

In an 8-second flat-ground simulation, the robot achieved approximately 1.573 m of forward displacement, with an average forward velocity of 0.1952 m/s. The overall joint position tracking RMSE was 0.146610 rad.

These results demonstrate forward locomotion in the simulated environment while also revealing opportunities for improving velocity tracking and joint-level accuracy.


## Core Modules

The V3 locomotion controller consists of five modular components that coordinate gait timing, foothold adjustment, foot trajectory generation, and robot state feedback.

### 1. State Estimator

**File:** `controllers/v3_model_based/state_estimator.py`

The state estimator retrieves the robot's base state from the MuJoCo simulation, including:

* Base position in the world frame.
* Rotation matrix from the base frame to the world frame.
* Roll, pitch, and yaw angles.
* Linear velocity in the world frame.
* Angular velocity in the world frame.

The estimator also accounts for the difference between the subtree center of mass and the base origin when calculating linear velocity, providing a consistent velocity estimate for subsequent control modules.

### 2. Gait Scheduler

**File:** `controllers/v3_model_based/gait_scheduler.py`

The gait scheduler implements a diagonal trot pattern with a periodic phase variable.

* FR and RL legs operate in phase.
* FL and RR legs operate with a half-cycle phase offset.
* Each gait cycle is divided into stance and swing phases using a configurable duty factor.

The default gait period is 0.5 s, with a duty factor of 0.6, corresponding to a 60% stance phase and a 40% swing phase.

### 3. Foothold Planner

**File:** `controllers/v3_model_based/foothold_planner.py`

The foothold planner adjusts the nominal foothold according to the difference between the current and desired forward velocities.

The velocity error is defined as:

$$
e_v = v_x - v_x^{\mathrm{des}}
$$

The foothold adjustment is calculated as:

$$
\Delta x = \operatorname{clip}(k_v e_v,-\Delta x_{\max},\Delta x_{\max})
$$

where \(k_v\) is the feedback gain and \(\Delta x_{\max}\) is the maximum allowable foothold offset.

The adjustment is applied along the forward axis while preserving the nominal lateral and vertical coordinates.

This mechanism provides a simple velocity-feedback strategy for dynamic foot placement.

### 4. Foot Trajectory Generator

**File:** `controllers/v3_model_based/foot_trajectory_generator.py`

The foot trajectory generator coordinates the gait scheduler, foothold planner, and swing-leg controller to produce target positions and velocities for all four feet.

During the stance phase, the foot moves from the nominal front position toward the rear position relative to the robot body.

During the swing phase, the foot returns from the rear position to the front position while following a smooth trajectory.

Two operating modes are supported:

* **Fixed foothold mode:** Uses nominal foot placement without velocity-based adjustment.
* **Feedback-enabled mode:** Applies dynamic foothold offsets based on the robot's forward velocity error.

The foothold offset is updated at the stance-to-swing transition and locked during the corresponding swing cycle. This prevents the target landing position from changing continuously throughout the swing motion.

### 5. Swing Leg Controller

**File:** `controllers/v3_model_based/swing_leg_controller.py`

The swing-leg controller generates smooth foot trajectories using polynomial interpolation.

A fifth-order smoothstep function is used for horizontal motion:

$$
s_5(\tau)=10\tau^3-15\tau^4+6\tau^5
$$

The vertical trajectory is defined by:

$$
z(\tau)=z_0+h\,64\tau^3(1-\tau)^3
$$

where \(\tau\in[0,1]\) is the normalized swing phase and \(h\) is the foot clearance.

The trajectory provides smooth transitions at lift-off and landing, with zero endpoint velocity and acceleration for the horizontal interpolation and zero vertical velocity at both endpoints.

The default foot clearance is 0.05 m.

## Control Strategy

The locomotion controller adopts a hierarchical control architecture consisting of gait generation, state feedback, inverse kinematics, and joint-level torque control.

### 1. Three-Stage Locomotion

The simulation is divided into three phases:

* **Standing:** The robot maintains its initial standing configuration under joint-level feedback control.
* **Transition:** The desired foot positions are smoothly interpolated from the current configuration to the initial gait targets using a cubic smoothstep function. Inverse kinematics is applied to obtain the corresponding joint references.
* **Trot locomotion:** The robot executes a diagonal trot gait with velocity, attitude, and body-height feedback.

### 2. Velocity-Based Foothold Adaptation

During trot locomotion, the estimated base velocity is transformed from the world frame into the body frame. The forward velocity component is then compared with the desired velocity to adjust the foothold locations.

This feedback mechanism modifies the nominal foothold position along the forward axis, allowing the controller to adapt foot placement according to the robot's actual motion.

### 3. Body Attitude and Height Feedback

During the stance phase, the controller adjusts the vertical foot target based on body roll, pitch, height deviation, and vertical velocity.

The correction is expressed as:

$$
\Delta z =
-K_{p}^{\mathrm{pitch}}\theta\,x_f
+K_{p}^{\mathrm{roll}}\phi\,y_f
+K_{p}^{\mathrm{height}}(z_b-z_{\mathrm{des}})
+K_{d}^{\mathrm{height}}\dot{z}_b
$$

where:

* \(\phi\) and \(\theta\) denote the estimated roll and pitch angles.
* \(x_f\) and \(y_f\) are the horizontal coordinates of the desired foot position in the body frame.
* \(z_b\) is the current base height.
* \(z_{\mathrm{des}}\) is the desired base height.
* \(\dot{z}_b\) is the vertical velocity of the base.

The correction is applied to the vertical component of the desired foot position before inverse kinematics. This provides a feedback mechanism for regulating body attitude and height through stance-foot placement.

### 4. Inverse Kinematics and Joint-Level Torque Control

The desired foot positions are converted into joint-angle references using damped inverse kinematics.

The joint-level controller combines proportional-derivative feedback with gravity compensation:

$$
\tau_{\mathrm{raw}} =
K_p(q_{\mathrm{des}}-q)
-K_d\dot{q}
+\tau_{\mathrm{gravity}}
$$

The resulting torque commands are clipped to the actuator limits before being sent to the MuJoCo simulation.

This control structure combines task-space foot placement with joint-space feedback while explicitly accounting for gravity and actuator constraints.

### Control Framework Summary

Together, these modules establish a modular locomotion framework that combines periodic gait coordination, state-dependent foothold adjustment, and smooth foot trajectory generation. The generated foot targets are subsequently processed by inverse kinematics and joint-level feedback control to produce actuator commands for the simulated robot.
## Simulation Setup

### Robot and Environment

* **Robot:** Unitree Go2 quadruped
* **Simulator:** MuJoCo
* **Environment:** Flat ground without obstacles
* **Actuated joints:** 12 (3 per leg)
* **Leg order:** FR, FL, RR, RL
* **Joint types:** Hip, thigh, and calf
* **Initial standing configuration:** `[0.0, 0.8, -1.45]` rad per leg

### Gait and Feedback Parameters

| Parameter                |   Value |
| ------------------------ | ------: |
| Desired forward velocity | 0.5 m/s |
| Gait period              |   0.5 s |
| Duty factor              |     0.6 |
| Stance duration          |   0.3 s |
| Swing duration           |   0.2 s |
| Step length              |  0.15 m |
| Swing foot clearance     |  0.05 m |
| Foothold feedback gain   |    0.15 |
| Maximum foothold offset  |  0.05 m |

### Body Feedback Gains

| Feedback term             | Gain |
| ------------------------- | ---: |
| Roll                      |  1.0 |
| Pitch                     |  1.0 |
| Height                    |  2.0 |
| Vertical velocity damping |  0.2 |

### Joint-Level PD Gains

The proportional and derivative gains are configured per joint type:

| Joint | \(K_p\) | \(K_d\) |
| ----- | ------: | ------: |
| Hip   |    20.0 |     1.0 |
| Thigh |    20.0 |     1.0 |
| Calf  |    40.0 |     2.0 |

Gravity compensation is added to the PD torque, and the resulting commands are clipped to the actuator control limits defined by the MuJoCo model.

### Simulation Timeline

The simulation runs for 8 seconds and consists of three stages:

| Stage             | Time interval | Duration |
| ----------------- | ------------- | -------: |
| Standing          | 0–2 s         |      2 s |
| Smooth transition | 2–3 s         |      1 s |
| Trot locomotion   | 3–8 s         |      5 s |

The simulation timestep is loaded directly from the MuJoCo model configuration.
## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/quadruped_locomotion.git
cd quadruped_locomotion
```

### 2. Set Up the Python Environment

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Run the V3 Locomotion Simulation

From the project root directory, execute:

```bash
python scripts/run_free_trot_v3.py
```

The simulation opens the MuJoCo viewer and saves the experiment log and generated results under:

```text
outputs/free_trot_v3/
```

**Note:** The robot model path is configured relative to the project root. Run the script from that directory to ensure the model and output paths resolve correctly.

## System Architecture
flowchart TD
    A["MuJoCo Go2 Simulation"] --> B["Robot State Feedback"]
    B --> C["State Estimator"]
    C --> D["Gait Scheduler"]
    D --> E["Foothold Planner"]
    E --> F["Foot Trajectory Generator"]
    F --> G["Swing Leg Controller"]
    G --> H["Inverse Kinematics"]
    H --> I["Joint-Level Feedback Control"]
    I --> J["Joint Torque Commands"]
    J --> A
