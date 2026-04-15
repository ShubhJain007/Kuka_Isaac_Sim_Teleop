# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""USD bone mesh visualization: Poisson surface reconstruction + ICP tracking.

First point cloud → Open3D Poisson mesh → written to UsdGeom.Mesh once.
Subsequent point clouds → ICP vs. reference → update prim transform only.
"""

from __future__ import annotations

import time
import numpy as np


_cached_widths: dict[tuple[str, int], object] = {}


def create_or_update_pointcloud_prim(
    stage,
    prim_path: str,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    point_size: float = 0.003,
) -> bool:
    """Create or update a USD Points prim (legacy, used for local rendering only)."""
    from pxr import UsdGeom, Vt

    if points.shape[0] == 0:
        return False

    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        points_geom = UsdGeom.Points.Define(stage, prim_path)
        prim = points_geom.GetPrim()
    else:
        points_geom = UsdGeom.Points(prim)

    n = points.shape[0]
    pts_f32 = np.ascontiguousarray(points, dtype=np.float32)
    vt_points = Vt.Vec3fArray(pts_f32.tolist())
    points_geom.GetPointsAttr().Set(vt_points)

    if colors is not None and colors.shape[0] == n:
        cols_f32 = np.ascontiguousarray(colors, dtype=np.float32)
        points_geom.GetDisplayColorAttr().Set(Vt.Vec3fArray(cols_f32.tolist()))

    cache_key = (prim_path, n)
    if cache_key not in _cached_widths:
        _cached_widths[cache_key] = Vt.FloatArray([point_size] * n)
    points_geom.GetWidthsAttr().Set(_cached_widths[cache_key])

    return True


def clear_pointcloud_prim(stage, prim_path: str) -> bool:
    """Remove points from a USD Points prim (set to empty array)."""
    from pxr import UsdGeom, Vt

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False

    points_geom = UsdGeom.Points(prim)
    points_attr = points_geom.GetPointsAttr()
    points_attr.Set(Vt.Vec3fArray([]))

    return True


# ---------------------------------------------------------------------------
#  BoneMeshManager — Poisson surface reconstruction + ICP pose tracking
# ---------------------------------------------------------------------------

class BoneMeshManager:
    """Manages one bone: reconstructs a mesh once, then tracks via ICP.

    Usage:
        mgr = BoneMeshManager("femur", prim_path, color=(0.0, 0.8, 0.0))
        # every time a new PointCloud2 arrives:
        mgr.update(stage, new_points)
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        color: tuple[float, float, float] = (0.9, 0.9, 0.9),
        poisson_depth: int = 6,
        icp_max_dist: float = 0.02,
    ):
        self.name = name
        self.prim_path = prim_path
        self.color = color
        self._poisson_depth = poisson_depth
        self._icp_max_dist = icp_max_dist

        self._ref_pcd = None        # open3d.geometry.PointCloud (reference)
        self._mesh_written = False   # True after first reconstruction
        self._update_count = 0
        self._o3d = None             # lazy import

    @property
    def mesh_ready(self) -> bool:
        return self._mesh_written

    def _get_o3d(self):
        if self._o3d is None:
            import open3d
            self._o3d = open3d
        return self._o3d

    # ------------------------------------------------------------------
    def update(self, stage, points: np.ndarray) -> None:
        """Process a new point cloud.  First call reconstructs mesh,
        subsequent calls compute ICP transform."""
        if points.shape[0] < 20:
            return
        self._update_count += 1

        if not self._mesh_written:
            self._reconstruct_and_write(stage, points)
        else:
            self._icp_update(stage, points)

    # ------------------------------------------------------------------
    def _reconstruct_and_write(self, stage, points: np.ndarray) -> None:
        o3d = self._get_o3d()
        t0 = time.time()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self._poisson_depth)

        # Trim low-density outlier vertices
        dens = np.asarray(densities)
        thresh = np.quantile(dens, 0.02)
        mask = dens < thresh
        mesh.remove_vertices_by_mask(mask)
        mesh.compute_vertex_normals()

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)

        if verts.shape[0] == 0 or faces.shape[0] == 0:
            print(f"[BoneMesh] {self.name}: Poisson reconstruction produced empty mesh, skipping")
            return

        self._write_mesh_to_usd(stage, verts, faces)

        self._ref_pcd = pcd
        self._mesh_written = True

        dt_ms = (time.time() - t0) * 1000
        print(f"[BoneMesh] {self.name}: reconstructed mesh from {points.shape[0]} pts "
              f"→ {verts.shape[0]} verts, {faces.shape[0]} faces ({dt_ms:.0f} ms)")

    # ------------------------------------------------------------------
    def _write_mesh_to_usd(self, stage, verts: np.ndarray, faces: np.ndarray) -> None:
        from pxr import UsdGeom, UsdShade, Vt, Gf, Sdf

        prim = stage.GetPrimAtPath(self.prim_path)
        if prim.IsValid():
            mesh_geom = UsdGeom.Mesh(prim)
        else:
            mesh_geom = UsdGeom.Mesh.Define(stage, self.prim_path)

        mesh_geom.GetPointsAttr().Set(Vt.Vec3fArray(verts.tolist()))
        mesh_geom.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
        mesh_geom.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
        mesh_geom.GetSubdivisionSchemeAttr().Set("none")

        n_verts = verts.shape[0]
        vert_colors = Vt.Vec3fArray([Gf.Vec3f(*self.color)] * n_verts)
        mesh_geom.GetDisplayColorAttr().Set(vert_colors)

        ext_min = verts.min(axis=0)
        ext_max = verts.max(axis=0)
        mesh_geom.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(*ext_min.tolist()), Gf.Vec3f(*ext_max.tolist()),
        ]))

        try:
            mesh_geom.GetDoubleSidedAttr().Set(True)
        except Exception:
            pass

        # Bind a UsdPreviewSurface material — RealityKit ignores displayColor
        # and requires a proper material to render the mesh.
        mat_path = self.prim_path + "/BoneMaterial"
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            material = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path + "/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*self.color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.85)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(mesh_geom.GetPrim()).Bind(material)

    # ------------------------------------------------------------------
    def _icp_update(self, stage, points: np.ndarray) -> None:
        o3d = self._get_o3d()
        t0 = time.time()

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # ICP: find T that maps *reference* → *new* cloud
        result = o3d.pipelines.registration.registration_icp(
            source=self._ref_pcd,
            target=new_pcd,
            max_correspondence_distance=self._icp_max_dist,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        T = np.array(result.transformation)  # 4×4 column-major math convention
        fitness = result.fitness

        if fitness < 0.3:
            if self._update_count <= 5:
                print(f"[BoneMesh] {self.name}: ICP fitness too low ({fitness:.2f}), skipping")
            return

        self._apply_transform(stage, T)

        dt_ms = (time.time() - t0) * 1000
        if self._update_count <= 3 or self._update_count % 100 == 0:
            print(f"[BoneMesh] {self.name}: ICP update #{self._update_count}: "
                  f"fitness={fitness:.3f} ({dt_ms:.1f} ms)")

    # ------------------------------------------------------------------
    def _apply_transform(self, stage, T: np.ndarray) -> None:
        """Apply a 4×4 rigid transform to the mesh prim."""
        from pxr import UsdGeom, Gf

        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return

        # USD Gf.Matrix4d is row-major (transpose of standard math convention)
        gf_mat = Gf.Matrix4d(
            T[0, 0], T[1, 0], T[2, 0], T[3, 0],
            T[0, 1], T[1, 1], T[2, 1], T[3, 1],
            T[0, 2], T[1, 2], T[2, 2], T[3, 2],
            T[0, 3], T[1, 3], T[2, 3], T[3, 3],
        )
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(gf_mat)
