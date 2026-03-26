"""
CPU-based replacements for cumesh operations, for ROCm/AMD GPU systems.

cumesh is CUDA-only (no ROCm builds). This module provides drop-in replacements
using trimesh, xatlas, and pyfqmr for CPU-based mesh processing.

On Strix Halo APUs with unified memory, CPU<->GPU copies are essentially free,
so CPU-based mesh processing has minimal overhead.
"""

import logging
import numpy as np
import torch
import trimesh as _trimesh

log = logging.getLogger("trellis2.rocm_mesh_ops")


class CuMeshCompat:
    """CPU-based drop-in for cumesh.CuMesh using trimesh."""

    def __init__(self):
        self._mesh = None
        self._vertices = None
        self._faces = None

    def init(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Initialize from torch tensors (matches cumesh.CuMesh.init)."""
        if isinstance(vertices, torch.Tensor):
            self._vertices = vertices.detach().cpu().float()
        else:
            self._vertices = torch.tensor(vertices, dtype=torch.float32)

        if isinstance(faces, torch.Tensor):
            self._faces = faces.detach().cpu().int()
        else:
            self._faces = torch.tensor(faces, dtype=torch.int32)

        self._rebuild_trimesh()

    def _rebuild_trimesh(self):
        """Rebuild the internal trimesh object from current verts/faces."""
        self._mesh = _trimesh.Trimesh(
            vertices=self._vertices.numpy(),
            faces=self._faces.numpy(),
            process=False,
        )

    @property
    def num_vertices(self):
        return len(self._mesh.vertices)

    @property
    def num_faces(self):
        return len(self._mesh.faces)

    @property
    def num_boundaries(self):
        """Approximate boundary count using trimesh edge detection."""
        edges = self._mesh.edges_sorted
        edge_counts = {}
        for e in edges:
            key = tuple(e)
            edge_counts[key] = edge_counts.get(key, 0) + 1
        boundary_edges = sum(1 for c in edge_counts.values() if c == 1)
        return boundary_edges

    @property
    def num_boundary_loops(self):
        """Count boundary loops."""
        outline = self._mesh.outline()
        if outline is None or len(outline.entities) == 0:
            return 0
        return len(outline.entities)

    # These are no-ops for the CPU path — cumesh builds internal data structures
    def get_edges(self):
        pass

    def get_boundary_info(self):
        pass

    def get_vertex_edge_adjacency(self):
        pass

    def get_vertex_boundary_adjacency(self):
        pass

    def get_manifold_boundary_adjacency(self):
        pass

    def read_manifold_boundary_adjacency(self):
        pass

    def get_boundary_connected_components(self):
        pass

    def get_boundary_loops(self):
        pass

    def fill_holes(self, max_hole_perimeter=0.03):
        """Fill holes in the mesh. Uses trimesh hole filling."""
        try:
            _trimesh.repair.fill_holes(self._mesh)
            self._vertices = torch.tensor(self._mesh.vertices, dtype=torch.float32)
            self._faces = torch.tensor(self._mesh.faces, dtype=torch.int32)
            log.info(f"fill_holes: {self.num_vertices} verts, {self.num_faces} faces")
        except Exception as e:
            log.warning(f"fill_holes failed (non-fatal): {e}")

    def remove_faces(self, face_mask: torch.Tensor):
        """Remove faces indicated by boolean mask."""
        if isinstance(face_mask, torch.Tensor):
            face_mask = face_mask.detach().cpu().numpy()
        keep = ~face_mask.astype(bool)
        self._mesh.update_faces(keep)
        self._mesh.remove_unreferenced_vertices()
        self._vertices = torch.tensor(self._mesh.vertices, dtype=torch.float32)
        self._faces = torch.tensor(self._mesh.faces, dtype=torch.int32)

    def unify_face_orientations(self):
        """Unify face winding so normals are consistent."""
        _trimesh.repair.fix_normals(self._mesh)
        _trimesh.repair.fix_winding(self._mesh)
        self._vertices = torch.tensor(self._mesh.vertices, dtype=torch.float32)
        self._faces = torch.tensor(self._mesh.faces, dtype=torch.int32)
        log.info("Unified face orientations (CPU)")

    def simplify(self, target_face_count, verbose=False, options=None):
        """Simplify mesh to target face count using quadric decimation."""
        if options is None:
            options = {}

        current_faces = len(self._mesh.faces)
        if current_faces <= target_face_count:
            if verbose:
                log.info(f"simplify: already at {current_faces} faces (<= {target_face_count}), skipping")
            return

        try:
            # Try pyfqmr first (faster, better quality)
            import pyfqmr
            mesh_simplifier = pyfqmr.Simplify()
            mesh_simplifier.setMesh(self._mesh.vertices, self._mesh.faces)
            mesh_simplifier.simplify_mesh(
                target_count=target_face_count,
                aggressiveness=7,
                preserve_border=True,
                verbose=10 if verbose else 0,
            )
            new_verts, new_faces, _ = mesh_simplifier.getMesh()
        except ImportError:
            # Fall back to fast_simplification (used by trimesh)
            try:
                import fast_simplification
                target_reduction = 1.0 - (target_face_count / current_faces)
                target_reduction = max(0.0, min(1.0, target_reduction))
                new_verts, new_faces = fast_simplification.simplify(
                    self._mesh.vertices.astype(np.float32),
                    self._mesh.faces.astype(np.int32),
                    target_reduction=target_reduction,
                )
            except ImportError:
                log.warning("Neither pyfqmr nor fast_simplification available, skipping simplify")
                return

        self._vertices = torch.tensor(new_verts, dtype=torch.float32)
        self._faces = torch.tensor(new_faces, dtype=torch.int32)
        self._rebuild_trimesh()

        if verbose:
            log.info(f"simplify: {current_faces} -> {self.num_faces} faces")

    def uv_unwrap(self, compute_charts_kwargs=None, return_vmaps=False, verbose=False):
        """UV unwrap using xatlas."""
        import xatlas

        if compute_charts_kwargs is None:
            compute_charts_kwargs = {}

        verts_np = self._mesh.vertices.astype(np.float32)
        faces_np = self._mesh.faces.astype(np.uint32)

        # Map cumesh kwargs to xatlas ChartOptions
        chart_options = xatlas.ChartOptions()
        if "threshold_cone_half_angle_rad" in compute_charts_kwargs:
            # xatlas uses max_cost, not cone angle directly — approximate
            angle = compute_charts_kwargs["threshold_cone_half_angle_rad"]
            chart_options.normal_deviation_weight = 2.0 * (1.0 - np.cos(angle))
        if "refine_iterations" in compute_charts_kwargs:
            pass  # xatlas handles this internally
        if "global_iterations" in compute_charts_kwargs:
            pass
        if "smooth_strength" in compute_charts_kwargs:
            pass

        pack_options = xatlas.PackOptions()

        atlas = xatlas.Atlas()
        atlas.add_mesh(verts_np, faces_np)
        atlas.generate(chart_options=chart_options, pack_options=pack_options, verbose=verbose)

        vmapping, new_faces, uvs = atlas[0]

        out_vertices = torch.tensor(verts_np[vmapping], dtype=torch.float32)
        out_faces = torch.tensor(new_faces.astype(np.int32), dtype=torch.int32)
        out_uvs = torch.tensor(uvs, dtype=torch.float32)
        out_vmaps = torch.tensor(vmapping.astype(np.int64), dtype=torch.long)

        if return_vmaps:
            return out_vertices, out_faces, out_uvs, out_vmaps
        return out_vertices, out_faces, out_uvs

    def compute_vertex_normals(self):
        """Compute vertex normals using trimesh."""
        # trimesh auto-computes vertex_normals
        self._vertex_normals = torch.tensor(
            self._mesh.vertex_normals.copy(), dtype=torch.float32
        )

    def read_vertex_normals(self):
        """Return vertex normals as a tensor."""
        if self._vertex_normals is None:
            self.compute_vertex_normals()
        return self._vertex_normals

    def read(self):
        """Return (vertices, faces) as torch tensors on CPU."""
        return self._vertices.clone(), self._faces.clone()

    # Internal attribute
    _vertex_normals = None


class cuBVHCompat:
    """CPU-based drop-in for cumesh.cuBVH using trimesh proximity."""

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        if isinstance(vertices, torch.Tensor):
            verts_np = vertices.detach().cpu().numpy()
        else:
            verts_np = np.asarray(vertices)

        if isinstance(faces, torch.Tensor):
            faces_np = faces.detach().cpu().numpy()
        else:
            faces_np = np.asarray(faces)

        self._mesh = _trimesh.Trimesh(
            vertices=verts_np.astype(np.float64),
            faces=faces_np.astype(np.int64),
            process=False,
        )
        self._proximity = _trimesh.proximity.ProximityQuery(self._mesh)

    def unsigned_distance(self, points: torch.Tensor, return_uvw=False):
        """Compute unsigned distance from points to mesh surface.

        Returns:
            distances: [N] unsigned distances
            face_ids: [N] closest face indices (if return_uvw)
            uvw: [N, 3] barycentric coordinates on closest face (if return_uvw)
        """
        device = points.device if isinstance(points, torch.Tensor) else 'cpu'

        if isinstance(points, torch.Tensor):
            pts_np = points.detach().cpu().numpy().astype(np.float64)
        else:
            pts_np = np.asarray(points, dtype=np.float64)

        closest_points, distances, face_ids = self._proximity.on_surface(pts_np)
        distances = torch.tensor(distances, dtype=torch.float32, device=device)

        if not return_uvw:
            return distances

        face_ids_tensor = torch.tensor(face_ids, dtype=torch.long, device=device)

        # Compute barycentric coordinates
        tri_verts = self._mesh.vertices[self._mesh.faces[face_ids]]  # [N, 3, 3]
        v0 = tri_verts[:, 0]
        v1 = tri_verts[:, 1]
        v2 = tri_verts[:, 2]

        # Barycentric from closest point
        cp = closest_points.astype(np.float64)
        e0 = v1 - v0
        e1 = v2 - v0
        v = cp - v0

        d00 = np.sum(e0 * e0, axis=1)
        d01 = np.sum(e0 * e1, axis=1)
        d11 = np.sum(e1 * e1, axis=1)
        d20 = np.sum(v * e0, axis=1)
        d21 = np.sum(v * e1, axis=1)

        denom = d00 * d11 - d01 * d01
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

        bary_v = (d11 * d20 - d01 * d21) / denom
        bary_w = (d00 * d21 - d01 * d20) / denom
        bary_u = 1.0 - bary_v - bary_w

        uvw = torch.tensor(
            np.stack([bary_u, bary_v, bary_w], axis=1),
            dtype=torch.float32,
            device=device,
        )

        return distances, face_ids_tensor, uvw


class _RemeshingCompat:
    """Placeholder for cumesh.remeshing operations."""

    @staticmethod
    def remesh_narrow_band_dc(
        vertices, faces, center=None, scale=None, resolution=512,
        band=1.0, project_back=0.0, verbose=False, bvh=None,
    ):
        """Remesh using narrow-band dual contouring.

        This is a simplified CPU fallback — returns the mesh mostly unchanged
        since proper dual contouring remeshing requires the CUDA kernels.
        The mesh will still work but topology won't be as clean.
        """
        log.warning(
            "remesh_narrow_band_dc: using CPU fallback (passthrough). "
            "Install cumesh for GPU-accelerated remeshing."
        )
        if isinstance(vertices, torch.Tensor):
            return vertices.cpu(), faces.cpu()
        return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32)


# Module-level aliases to match cumesh API
CuMesh = CuMeshCompat
cuBVH = cuBVHCompat
remeshing = _RemeshingCompat()
