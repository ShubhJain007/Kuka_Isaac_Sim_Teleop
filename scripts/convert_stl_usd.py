import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert STL to USD using Isaac Lab.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

def main():
    cfg = MeshConverterCfg(
        asset_path='/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/Draw_Left_Femur_Plan_Array_V2.STL',
        usd_dir='/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/femur_cut_usd',
        force_usd_conversion=True,
        scale=(0.001, 0.001, 0.001), # Convert mm to meters
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(), #otherwise no rigid body properties only the mesh
    )

    # MeshConverter handles conversion during __init__
    convertor = MeshConverter(cfg)
    print(f"[INFO] Done! USD asset saved at: {convertor.usd_path}")

if __name__ == "__main__":
    main()
    simulation_app.close()