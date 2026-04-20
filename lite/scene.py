"""
Lightweight scene: loads robot URDF, bone meshes, surgical prims.
Kinematic-only — joint angles applied via forward kinematics, no PhysX.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Kuka Med7 kinematic chain from URDF.
# Each entry: (joint_name, parent_link, child_link, origin_xyz, origin_rpy, axis)
_JOINT_CHAIN = [
    ("lbr_A1", "lbr_link_0", "lbr_link_1", (0, 0, 0.1475),       (0, 0, 0), (0, 0, 1)),
    ("lbr_A2", "lbr_link_1", "lbr_link_2", (0, -0.0105, 0.1925),  (0, 0, 0), (0, 1, 0)),
    ("lbr_A3", "lbr_link_2", "lbr_link_3", (0, 0.0105, 0.2075),   (0, 0, 0), (0, 0, 1)),
    ("lbr_A4", "lbr_link_3", "lbr_link_4", (0, 0.0105, 0.1925),   (0, 0, 0), (0, -1, 0)),
    ("lbr_A5", "lbr_link_4", "lbr_link_5", (0, -0.0105, 0.2075),  (0, 0, 0), (0, 0, 1)),
    ("lbr_A6", "lbr_link_5", "lbr_link_6", (0, -0.0707, 0.1925),  (0, 0, 0), (0, 1, 0)),
    ("lbr_A7", "lbr_link_6", "lbr_link_7", (0, 0.0707, 0.091),    (0, 0, 0), (0, 0, 1)),
]

# Fixed joint at end
_EE_ORIGIN = (0, 0, 0.035)


@dataclass
class SceneConfig:
    urdf_path: str = str(_PROJECT_ROOT / "med7.urdf")
    femur_usd: str = str(_PROJECT_ROOT / "femur_cut_usd" / "Draw_Left_Femur_Plan_Array_V2.usd")
    tibia_usd: str = str(_PROJECT_ROOT / "tibia_cut_usd" / "tibia_cut.usd")

    robot_base_height: float = 0.50
    mount_size: tuple = (0.40, 0.40, 0.50)
    bed_size: tuple = (2.0, 1.0, 0.55)
    bed_offset: tuple = (1.3, 0.0, 0.275)

    femur_init_pos: tuple = (1.3, 0.0, 0.8)
    tibia_init_pos: tuple = (1.5, 0.0, 0.8)

    probe_radius: float = 0.005
    guide_line_radius: float = 0.0012

    joint_names: list = field(default_factory=lambda: [
        "lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4",
        "lbr_A5", "lbr_A6", "lbr_A7",
    ])


class Med7Scene:
    """Pure USD scene — no physics, purely kinematic."""

    def __init__(self, cfg: SceneConfig | None = None):
        self.cfg = cfg or SceneConfig()
        self._stage = None
        self._link_prim_paths: dict[str, str] = {}
        self._joint_angles = np.zeros(7)
        self._fk_logged = False

    def setup(self, stage):
        """Build the full scene on the given USD stage."""
        self._stage = stage

        self._create_lights(stage)
        self._create_ground(stage)
        self._create_mount(stage)
        self._create_bed(stage)
        self._load_robot(stage)
        self._load_bones(stage)
        self._create_surgical_prims(stage)

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _create_lights(self, stage):
        from pxr import UsdLux
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.GetIntensityAttr().Set(800.0)
        dome.GetColorAttr().Set((0.92, 0.95, 1.0))

        dist = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        dist.GetIntensityAttr().Set(2000.0)
        dist.GetColorAttr().Set((1.0, 0.97, 0.88))
        dist.GetAngleAttr().Set(0.53)

    def _create_ground(self, stage):
        from pxr import UsdGeom, Gf
        plane = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
        s = 50.0
        plane.GetPointsAttr().Set([
            Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, -s, 0),
            Gf.Vec3f(s, s, 0), Gf.Vec3f(-s, s, 0),
        ])
        plane.GetFaceVertexCountsAttr().Set([4])
        plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        plane.GetDisplayColorAttr().Set([(0.78, 0.80, 0.82)])

    def _create_mount(self, stage):
        from pxr import UsdGeom, Gf
        sx, sy, sz = self.cfg.mount_size
        cube = UsdGeom.Cube.Define(stage, "/World/RobotMount")
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, sz / 2.0))
        cube.GetDisplayColorAttr().Set([(0.72, 0.78, 0.82)])

    def _create_bed(self, stage):
        from pxr import UsdGeom, Gf
        sx, sy, sz = self.cfg.bed_size
        bx, by, bz = self.cfg.bed_offset
        cube = UsdGeom.Cube.Define(stage, "/World/HospitalBed")
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        xf.AddTranslateOp().Set(Gf.Vec3d(bx, by, bz))
        cube.GetDisplayColorAttr().Set([(0.0, 0.0, 0.0)])

    def _load_robot(self, stage):
        """Import URDF for visual meshes, then track link prim paths for FK."""
        from pxr import UsdGeom, Gf

        urdf_path = self.cfg.urdf_path
        if not os.path.isfile(urdf_path):
            print(f"[WARN] URDF not found: {urdf_path}")
            return

        import omni.kit.app
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

        from isaacsim.asset.importer.urdf import _urdf as urdf_interface
        urdf_api = urdf_interface.acquire_urdf_interface()

        import_config = urdf_interface.ImportConfig()
        import_config.fix_base = True
        import_config.make_default_prim = False
        import_config.create_physics_scene = False

        urdf_dir = os.path.dirname(urdf_path)
        urdf_file = os.path.basename(urdf_path)
        parsed = urdf_api.parse_urdf(urdf_dir, urdf_file, import_config)
        result = urdf_api.import_robot(urdf_dir, urdf_file, parsed, import_config)
        print(f"[SCENE] URDF import result: {result}")

        # Find the robot root prim — importer uses the URDF robot name
        robot_prim = None
        for candidate in ["/med7", "/World/med7", "/World/Robot", f"/{urdf_file.replace('.urdf', '')}"]:
            p = stage.GetPrimAtPath(candidate)
            if p and p.IsValid():
                robot_prim = p
                print(f"[SCENE] Robot prim found at: {candidate}")
                break

        if robot_prim is None:
            # Fallback: search stage for any prim containing lbr_link_0
            for prim in stage.Traverse():
                if prim.GetName() == "lbr_link_0":
                    robot_prim = prim.GetParent()
                    print(f"[SCENE] Robot prim found via link search: {robot_prim.GetPath()}")
                    break

        if robot_prim is None:
            print("[WARN] Robot prim not found after URDF import")
            print("[DEBUG] Top-level prims:")
            for child in stage.GetPseudoRoot().GetChildren():
                print(f"  {child.GetPath()}")
            return

        # Raise robot to sit on top of mount
        xf = UsdGeom.Xformable(robot_prim)
        existing_ops = xf.GetOrderedXformOps()
        if existing_ops:
            for op in existing_ops:
                if op.GetOpName() == "xformOp:translate":
                    op.Set(Gf.Vec3d(0.0, 0.0, self.cfg.robot_base_height))
                    break
            else:
                mat = Gf.Matrix4d()
                mat.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, self.cfg.robot_base_height))
                xf.MakeMatrixXform().Set(mat)
        else:
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, self.cfg.robot_base_height))

        # Find link prims by name for FK
        for prim in stage.Traverse():
            name = prim.GetName()
            if name.startswith("lbr_link_"):
                self._link_prim_paths[name] = str(prim.GetPath())

        found = [n for n in self._link_prim_paths if n in
                 [j[2] for j in _JOINT_CHAIN] + ["lbr_link_0"]]
        print(f"[SCENE] Robot loaded, {len(found)} links found for FK")

    def _load_bones(self, stage):
        from pxr import UsdGeom, Gf

        UsdGeom.Scope.Define(stage, "/World/Bones")

        for name, usd_path, init_pos in [
            ("femur", self.cfg.femur_usd, self.cfg.femur_init_pos),
            ("tibia", self.cfg.tibia_usd, self.cfg.tibia_init_pos),
        ]:
            prim_path = f"/World/Bones/{name}"
            if os.path.isfile(usd_path):
                prim = stage.DefinePrim(prim_path)
                prim.GetReferences().AddReference(usd_path)
                mat = Gf.Matrix4d()
                mat.SetTranslateOnly(Gf.Vec3d(*init_pos))
                UsdGeom.Xformable(prim).MakeMatrixXform().Set(mat)
                print(f"[SCENE] {name} loaded at {prim_path}")
            else:
                print(f"[WARN] Bone USD not found: {usd_path}")

    def _create_surgical_prims(self, stage):
        from pxr import UsdGeom, Gf

        UsdGeom.Scope.Define(stage, "/World/SurgicalPlan")

        probe = UsdGeom.Sphere.Define(stage, "/World/SurgicalPlan/Probe")
        probe.GetRadiusAttr().Set(self.cfg.probe_radius)
        probe.GetDisplayColorAttr().Set([(0.0, 1.0, 1.0)])

        line = UsdGeom.Cylinder.Define(stage, "/World/SurgicalPlan/GuideLine")
        line.GetRadiusAttr().Set(self.cfg.guide_line_radius)
        line.GetHeightAttr().Set(0.2)
        line.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
        UsdGeom.Imageable(line.GetPrim()).MakeInvisible()

        stylus_xf = UsdGeom.Xform.Define(stage, "/World/SurgicalPlan/StylusTracker")
        tip = UsdGeom.Sphere.Define(stage, "/World/SurgicalPlan/StylusTracker/StylusTip")
        tip.GetRadiusAttr().Set(0.005)
        tip.GetDisplayColorAttr().Set([(0.0, 1.0, 0.4)])
        UsdGeom.Xformable(tip.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.045))

        aline = UsdGeom.Cylinder.Define(stage, "/World/SurgicalPlan/StylusTracker/ApproachLine")
        aline.GetRadiusAttr().Set(0.0008)
        aline.GetHeightAttr().Set(0.045)
        aline.GetDisplayColorAttr().Set([(0.0, 1.0, 0.4)])
        UsdGeom.Xformable(aline.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.045 / 2))

        UsdGeom.Imageable(stylus_xf.GetPrim()).MakeInvisible()

    # ------------------------------------------------------------------
    # Runtime: forward kinematics (no physics)
    # ------------------------------------------------------------------

    def set_joint_positions(self, joint_angles: dict[str, float]):
        """Apply joint angles via forward kinematics — pure xform, no PhysX.

        Sets each link's LOCAL transform (origin + joint rotation).
        The USD hierarchy handles parent-child composition automatically.
        """
        from pxr import UsdGeom, Gf

        if not self._link_prim_paths:
            return

        if not self._fk_logged:
            print(f"[FK] Applying joints: { {k: round(v, 3) for k, v in joint_angles.items()} }")
            self._fk_logged = True

        for jname, parent_link, child_link, origin_xyz, origin_rpy, axis in _JOINT_CHAIN:
            angle = joint_angles.get(jname, 0.0)

            # Local = origin translation, then joint rotation about axis
            T_origin = np.eye(4)
            T_origin[:3, 3] = origin_xyz

            T_rot = np.eye(4)
            axis_np = np.array(axis, dtype=np.float64)
            if abs(angle) > 1e-10:
                T_rot[:3, :3] = R.from_rotvec(axis_np * angle).as_matrix()

            T_local = T_origin @ T_rot

            child_path = self._link_prim_paths.get(child_link)
            if child_path is None:
                continue
            self._set_matrix_xform(child_path, T_local)

    def _set_matrix_xform(self, prim_path: str, T: np.ndarray):
        """Set a prim's transform from a 4x4 numpy matrix.

        USD Gf.Matrix4d uses row-vector convention (transpose of numpy).
        """
        from pxr import UsdGeom, Gf

        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        M = T.T  # transpose: numpy column-vector → USD row-vector
        mat = Gf.Matrix4d(
            M[0, 0], M[0, 1], M[0, 2], M[0, 3],
            M[1, 0], M[1, 1], M[1, 2], M[1, 3],
            M[2, 0], M[2, 1], M[2, 2], M[2, 3],
            M[3, 0], M[3, 1], M[3, 2], M[3, 3],
        )
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(mat)

    # ------------------------------------------------------------------
    # Runtime: prim pose setters
    # ------------------------------------------------------------------

    def set_prim_transform(self, prim_path: str, pos, quat_wxyz):
        """Set a prim's world transform from position + quaternion (w,x,y,z)."""
        w, x, y, z = (float(v) for v in quat_wxyz)
        rot = R.from_quat([x, y, z, w])  # scipy uses xyzw
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
        self._set_matrix_xform(prim_path, T)

    def set_bone_pose(self, bone_name: str, pos, quat_wxyz):
        self.set_prim_transform(f"/World/Bones/{bone_name}", pos, quat_wxyz)

    def set_probe_pose(self, pos, quat_wxyz):
        self.set_prim_transform("/World/SurgicalPlan/Probe", pos, quat_wxyz)

    def set_guide_line(self, pos, quat_wxyz, height: float, visible: bool = True):
        from pxr import UsdGeom
        prim = self._stage.GetPrimAtPath("/World/SurgicalPlan/GuideLine")
        if not prim or not prim.IsValid():
            return
        self.set_prim_transform("/World/SurgicalPlan/GuideLine", pos, quat_wxyz)
        h_attr = prim.GetAttribute("height")
        if h_attr.IsValid():
            h_attr.Set(float(height))
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if visible else img.MakeInvisible()

    def set_stylus_pose(self, mat_4x4: np.ndarray, visible: bool = True):
        from pxr import UsdGeom
        prim = self._stage.GetPrimAtPath("/World/SurgicalPlan/StylusTracker")
        if not prim or not prim.IsValid():
            return
        self._set_matrix_xform("/World/SurgicalPlan/StylusTracker", mat_4x4)
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if visible else img.MakeInvisible()

    def set_bone_opacity(self, bone_name: str, opacity: float):
        from pxr import UsdGeom, UsdShade
        prim = self._stage.GetPrimAtPath(f"/World/Bones/{bone_name}")
        if not prim or not prim.IsValid():
            return
        self._set_opacity_recursive(prim, opacity)

    def _set_opacity_recursive(self, prim, opacity: float):
        from pxr import UsdGeom, UsdShade, Sdf
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Set([opacity])
        binding = UsdShade.MaterialBindingAPI(prim)
        if binding:
            material = binding.GetDirectBinding().GetMaterial()
            if material:
                shader, _, _ = material.ComputeSurfaceSource()
                if shader:
                    inp = shader.GetInput("opacity")
                    if inp:
                        inp.Set(opacity)
                    else:
                        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        for child in prim.GetChildren():
            self._set_opacity_recursive(child, opacity)
