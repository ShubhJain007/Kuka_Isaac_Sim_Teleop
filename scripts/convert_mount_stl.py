"""One-shot converter: mount_with_tip.stl → USD with RigidBodyAPI + MassAPI baked in.

Produces a USD that can be spawned as a RigidObject, which is what avp_stream
needs in order to stream its pose to the Vision Pro every frame.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

SRC = "/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/med7/visual/mount_with_tip.stl"
DST_DIR = "/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/mount_with_tip_usd"

cfg = MeshConverterCfg(
    asset_path=SRC,
    usd_dir=DST_DIR,
    force_usd_conversion=True,
    scale=(0.001, 0.001, 0.001),  # mm → m
    rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
    ),
    mass_props=schemas_cfg.MassPropertiesCfg(mass=0.001),
    collision_props=None,
)

conv = MeshConverter(cfg)
print(f"[DONE] USD: {conv.usd_path}")
simulation_app.close()
