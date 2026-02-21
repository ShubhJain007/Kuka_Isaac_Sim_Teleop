from isaacsim import SimulationApp
sim = SimulationApp({'headless': True})

import omni.client
from omni.isaac.core.utils.nucleus import get_assets_root_path

root = get_assets_root_path()

def scan(path, depth=0):
    if depth > 5: return
    res, entries = omni.client.list(path)
    if res != omni.client.Result.OK:
        return
    for e in entries:
        full = f"{path}/{e.relative_path}"
        if e.flags & 1: # Folder
            scan(full, depth + 1)
        elif e.relative_path.endswith(".usd"):
            low = e.relative_path.lower()
            if "hospital" in low or "bed" in low or "stool" in low or "chair" in low:
                print(full)

print("--- SCANNING Environments ---")
scan(f"{root}/Isaac/Environments")
print("--- SCANNING Props ---")
scan(f"{root}/Isaac/Props")

sim.close()
