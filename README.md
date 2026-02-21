# Kuka Med 7: VR Teleoperation in Isaac Lab

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-5.1.0-orange.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

An advanced robotics teleoperation framework for the **Kuka Med 7 LBR**, implemented within the NVIDIA Isaac Lab simulation environment. This project features immersive **Apple Vision Pro** integration via CloudXR and low-latency web streaming.

## 🌟 Key Features

- **🏥 Realistic Clinical Environment**: A fully rendered medical suite featuring a hospital bed, surgical mount, and accurate medical lighting (6500K).
- **🦾 Kuka Med 7 LBR**: High-fidelity articulation robot model with tuned PD control and stable transition between joint and Cartesian space.
- **🕶️ VR Teleoperation**: Real-time hand tracking control using Apple Vision Pro and OpenXR, with custom coordinate transformations for intuitive "Human-in-the-loop" interaction.
- **🌐 Web Streaming (M-JPEG)**: A lightweight Flask-based streaming server providing multiple camera views (Wrist & Room) viewable directly in VR browsers (Safari) or external workstations.
- **🟢 Visualization Proxy**: A virtual hand proxy for visual feedback, ensuring precise alignment between the user's physical movements and robot targets.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Isaac Sim 4.5+** and **Isaac Lab** installed. Follow the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### 2. Installation
Clone the repository and install the project in editable mode:
```bash
# From the project root
python -m pip install -e source/Kuka_Med_7
```

### 3. Usage

#### ⌨️ Keyboard Teleoperation
Run basic Cartesian control using your keyboard:
```bash
python scripts/teleop_med7.py
```

#### 🥽 VR Teleoperation (Apple Vision Pro)
1. **Setup CloudXR**: Set the environment variable to enable the external renderer:
   ```bash
   export EXTERNAL_RENDERER=cloudxr
   ```
2. **Launch with Experience File**:
   ```bash
   python scripts/teleop_med7_vr.py --experience apps/isaaclab.python.headless.cloudxr.kit
   ```
3. **Connect Vision Pro**: Open the CloudXR client on your headset and point to your workstation IP.
4. **View Streams**: Open the Vision Pro Safari browser and navigate to `http://<YOUR_IP>:5000` to see the wrist and room cameras.

## 🛠️ Project Structure

- `source/Kuka_Med_7`: Core environment and task definitions.
  - `tasks/med7/med7_env_cfg.py`: Scene and agent configuration.
  - `tasks/med7/med7_env.py`: Environment logic and physics interaction.
- `scripts/`: Executable scripts for teleoperation and testing.
  - `teleop_med7_vr.py`: The main VR entry point with Flask streaming integration.
- `assets/`: URDFs and meshes for the Kuka Med 7.

## 📖 Documentation
Detailed technical documentation on the control loops, coordinate transforms, and streaming architecture can be found in [documentation.md](documentation.md).

## 📄 License
This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
