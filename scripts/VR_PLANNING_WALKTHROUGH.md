# VR surgical planning — pointer & cameras walkthrough

This applies to `teleop_med7_vr.py` (Vision Pro + CloudXR, planning mode only — no arm IK).

## Before you start

1. Source CloudXR env: `source scripts/cloudxr_env.sh` (see `README.md`).
2. Launch with the project kit, e.g.  
   `python scripts/teleop_med7_vr.py --experience "$(pwd)/apps/isaaclab.python.headless.cloudxr.kit"`
3. On the headset, use your app’s **Play** control so “teleoperation” is active (the script prints `VR planning active`).

---

## Room camera (OR-style view)

The **room** MJPEG stream (`http://<host>:5000/room`) uses a static camera placed **in front of the arm** at roughly surgeon eye height, looking toward the knee workspace — closer to a real OR / bedside view than the old corner shot.

- **Wrist** stream is still the camera on the robot flange (`lbr_link_7`).
- If the framing feels off, adjust `room_camera` `pos` / `rot` in `source/Kuka_Med_7/Kuka_Med_7/tasks/med7/med7_env_cfg.py`.

---

## Hand stick figures in the sim

OpenXR hand joints are drawn as **cyan** (right) and **orange** (left) **line “bones”** under `/World/SurgicalPlan/HandStick/{Right,Left}`. They move with tracked hands so you can see your pose in the same world as the robot and bones.

The green **HandProxy** sphere still tracks the **right wrist** (scene object used elsewhere).

---

## Controlling the pointer (recommended workflow)

The **probe** is the small **cyan sphere** (`/World/SurgicalPlan/Probe`). The red **guide cylinder** shows depth along the probe axis.

### 1. Move without grabbing (free steering)

- Move your **right wrist** and rotate — the probe moves in **small relative steps** (Se3Rel), like nudging a tool.
- Use this for fine adjustment when you are **not** pinching a grab.

### 2. Pinch-grab (pick up the pointer and move it anywhere)

This is the “put hand over the pointer and pinch” behavior:

1. **Touch the workspace with your right pinch** (thumb and index **close together**).
2. Place the pinch **near the cyan probe** (within about **10 cm** of the probe centre — see `GRAB_RADIUS_M` in the script).
3. **Close the pinch** (pinch transition: fingers were more open, then closed). That **arms** the grab if you’re close enough.
4. **While holding the pinch**, **move your hand** in space. The probe **keeps a fixed offset** from your **pinch midpoint** (halfway between thumb tip and index tip), so it **follows your hand** through the volume.
5. **Open the pinch** to **release** the probe.

While grabbing:

- You can still **rotate** the probe with **wrist rotation** (same deltas as before, applied on top).
- The **left hand** can still add **extra rotation** (second retargeter).
- **Guide line length** is **not** driven by thumb–index distance (that would fight the pinch). Line length updates again when you are **not** grabbing.

### 3. Guide line length (when not grabbing)

- **Thumb–index separation** maps to **cylinder length** (wider → longer line in the clamped range).

### 4. Left-hand gestures (unchanged)

- **Thumb–pinky spread** → bone translucency.
- **Left pinch timing** (when the right hand is **not** in a very tight pinch): **two** short taps with a deliberate gap → save line; **~0.52–1.08 s** hold → lock toggle; **≥ ~1.42 s** hold → femur/tibia target. **0.75 s cooldown** after any of these so jitter does not chain commands. See `LeftPinchGesture` in `teleop_med7_vr.py`.

### 5. Keyboard (optional)

Add `--keyboard-fallback` to use host keys for the same controls (arrows, `P`, `K`, `[,]`, etc.).

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| Grab never starts | Move pinch **closer** to the probe; ensure **Play** is on; pinch must **close** (edge) while near the probe. |
| Grab feels jumpy | XR anchor / room calibration; reduce `GRAB_RADIUS_M` if it grabs from too far. |
| Stick figures missing | Hand tracking off in runtime; joints missing in dict — some edges skip if tips are invalid. |
| Room view wrong | Tune `room_camera` offset in `med7_env_cfg.py`. |
| No cyan probe / sim looked broken | Surgical prims are created **after** the first `env.reset()` and the parent `/World/SurgicalPlan` scope is explicit. If you use **XR Reset**, prims are re-created automatically. Check console for `[INFO] Surgical pointer: ... valid=True`. |

---

## Constants (script)

- `GRAB_RADIUS_M` — max distance from pinch midpoint to probe to allow a new grab (meters).
- Pinch “closed” threshold — `pd_r < 0.042` (thumb–index distance, meters).
