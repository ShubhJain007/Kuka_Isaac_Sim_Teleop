from isaacsim import SimulationApp
sim = SimulationApp({'headless': True})

import omni.client
from omni.isaac.core.utils.nucleus import get_assets_root_path

root = get_assets_root_path()
hospital_props_path = f"{root}/Isaac/Props/Hospital"
res, _ = omni.client.stat(hospital_props_path)
print(f"Hospital Props Folder exists: {res == omni.client.Result.OK}")

def list_recursive(folder, depth=0):
    if depth > 3: return
    result, entries = omni.client.list(folder)
    if result != omni.client.Result.OK:
        return
    for entry in entries:
        child_path = f"{folder}/{entry.relative_path}"
        # Check for folder using bitwise AND with known folder flag values
        # ItemFlags.CAN_HAVE_CHILDREN is typically 1
        if entry.flags & 1: 
            list_recursive(child_path, depth + 1)
        elif entry.relative_path.endswith(".usd"):
            low = entry.relative_path.lower()
            if "bed" in low or "stool" in low or "chair" in low or "hospital" in low:
                print(child_path)

list_recursive(f"{root}/Isaac/Props")

