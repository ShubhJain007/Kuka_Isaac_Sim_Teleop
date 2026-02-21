from isaacsim import SimulationApp
sim = SimulationApp({'headless': True})

import omni.client
from omni.isaac.core.utils.nucleus import get_assets_root_path

root = get_assets_root_path()
res, entries = omni.client.list(f"{root}/Isaac/Props")

with open("props_list.txt", "w") as f:
    if res == omni.client.Result.OK:
        for e in entries:
            f.write(f"{e.relative_path}\n")
    else:
        f.write(f"FAILED to list: {res}\n")

sim.close()
