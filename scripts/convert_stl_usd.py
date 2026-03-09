import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert STL to USD using Isaac Lab.")
parser.add_argument("--asset_path", type=str, required=True, help="Path to the source STL file.")
parser.add_argument("--usd_dir", type=str, required=True, help="Directory to save the converted USD file.")
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
        asset_path=args_cli.asset_path,
        usd_dir=args_cli.usd_dir,
        force_usd_conversion=True,
        scale=(0.001, 0.001, 0.001),  # Convert mm to meters
        rigid_props=None,      # No physics body — pure visual mesh
        collision_props=None,  # No collision mesh
    )

    # MeshConverter handles conversion during __init__
    convertor = MeshConverter(cfg)
    print(f"[INFO] Done! USD asset saved at: {convertor.usd_path}")

if __name__ == "__main__":
    main()
    simulation_app.close()