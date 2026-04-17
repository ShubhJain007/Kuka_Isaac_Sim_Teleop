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
If startup fails (ROS 2 / `rclpy` / bridge issues), run without bone subscribers:
```bash
python scripts/teleop_med7.py --no-ros
```

#### 🥽 VR Teleoperation (Apple Vision Pro)
1. **Start CloudXR Runtime (Docker)**:
   In a dedicated terminal, start the server-side runtime:
   ```bash
   docker run -it --rm --name cloudxr-runtime \
       --user $(id -u):$(id -g) --gpus=all -e "ACCEPT_EULA=Y" \
       --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
       -p 48010:48010 -p 47998-48000:47998-48000/udp \
       -p 48005:48005/udp -p 48008:48008/udp -p 48012:48012/udp \
       nvcr.io/nvidia/cloudxr-runtime:5.0.1
   ```
2. **Setup Environment**: With CloudXR **still running**, in the terminal where you launch Isaac Lab:
   ```bash
   cd /path/to/Kuka_Med_7
   source scripts/cloudxr_env.sh
   ```
   This sets `EXTERNAL_RENDERER`, `XDG_RUNTIME_DIR`, `XR_RUNTIME_JSON`, and `LD_LIBRARY_PATH`.  
   **`docker run` must bind-mount the same host `openxr` folder** you use for `XDG_RUNTIME_DIR`. Example: if Docker uses `--mount src=$HOME/IsaacLab/openxr,dst=/openxr`, then Isaac must use `$HOME/IsaacLab/openxr/run` — `cloudxr_env.sh` **auto-detects** that path from the running `cloudxr-runtime` container. Override with `export CLOUDXR_OPENXR_ROOT=/your/openxr` before `source` if needed.  
   **If you see `ipc_cloudxr: No such file`:** start the container first, then `source scripts/cloudxr_env.sh` again. Run `bash scripts/check_cloudxr.sh` — it should show `[OK] IPC socket present`.
3. **Launch the Simulation**:
   The CloudXR experience is **not** bundled with the Isaac Lab installation — it is **this project’s** file at  
   `apps/isaaclab.python.headless.cloudxr.kit` (under the repo root). From the repo root:
   ```bash
   export EXTERNAL_RENDERER=cloudxr
   python scripts/teleop_med7_vr.py --experience "$(pwd)/apps/isaaclab.python.headless.cloudxr.kit"
   ```
   If `apps/` is missing, run `git pull` or recreate the file from the repository. **Fallback:** Isaac Lab’s stock XR headless kit, e.g.  
   `<IsaacLab>/apps/isaaclab.python.xr.openxr.headless.kit`, plus `EXTERNAL_RENDERER=cloudxr` — hand-tracking toggles may differ from this repo’s kit.
   Optional flags:
   - `--ros` — subscribe to `/bone_pose_femur` and `/bone_pose_tibia` (same bone updates as keyboard teleop).
   - `--keyboard-fallback` — also enable host keyboard (arrows, `[,]`, `P`, `K`, `1`/`2`, …) for debugging.
   - `--sensitivity` — scales hand-tracking deltas.

   **`teleop_med7_vr.py` (VR planning):** Moves the **surgical pointer / guide line / bone opacity** only; the robot joints are **held** (no arm IK). **By default everything is hand gestures:** pinch near the cyan probe to **grab & drag** the pointer; otherwise right wrist = probe motion, right thumb–index = line length (when not grabbing), left hand = opacity & gestures (see **`scripts/VR_PLANNING_WALKTHROUGH.md`**). Room MJPEG camera is an **OR-style view in front of the arm**. **`--keyboard-fallback`** opt-in for keys. Use **`teleop_med7.py`** for **keyboard arm** teleop.

4. **Connect Vision Pro**: Open the CloudXR client on your headset and point to your workstation IP.
5. **View Streams**: On Vision Pro Safari, open `http://<YOUR_IP>:5000` for MJPEG wrist/room cameras (optional alongside the CloudXR stereo view).

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
