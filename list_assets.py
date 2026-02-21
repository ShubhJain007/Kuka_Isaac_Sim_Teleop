from isaacsim import SimulationApp
sim = SimulationApp({'headless': True})

import omni.client
from omni.isaac.core.utils.nucleus import get_assets_root_path

root = get_assets_root_path()
props_path = f"{root}/Isaac/Props"

def list_recursive(folder, depth=0):
    if depth > 3: return
    result, entries = omni.client.list(folder)
    if result != omni.client.Result.OK:
        return
    for entry in entries:
        child_path = f"{folder}/{entry.relative_path}"
        if entry.flags & omni.client.ItemFlags.IS_FOLDER:
            list_recursive(child_path, depth + 1)
        elif entry.relative_path.endswith(".usd"):
            low = entry.relative_path.lower()
            if "bed" in low or "stool" in low or "chair" in low or "hospital" in low:
                print(child_path)

list_recursive(props_path)
sim.close()
