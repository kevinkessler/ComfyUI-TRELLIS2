"""
Pure PyTorch replacement for o_voxel operations, for ROCm/AMD GPU systems.

o_voxel is CUDA-only. This module reimplements flexible_dual_grid_to_mesh
and to_glb using pure PyTorch operations.

The flexible_dual_grid_to_mesh function implements a variant of Dual Contouring:
- Each active voxel has a dual vertex positioned by a learned offset
- Surface crossings on edges between voxels generate quad faces
- Quads connect the 4 voxels sharing an intersected edge
- Quads are triangulated using a learned diagonal parameter
"""

import logging
import numpy as np
import torch

log = logging.getLogger("trellis2.rocm_voxel_ops")


def flexible_dual_grid_to_mesh(
    coords,          # [N, 3] integer voxel coordinates
    vertex_offsets,  # [N, 3] per-voxel vertex position offsets
    intersected,     # [N, 3] edge intersection flags (bool in inference, float in training)
    quad_lerp,       # [N, 1] quad interpolation weight
    aabb=None,       # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    grid_size=512,   # resolution
    train=False,     # whether in training mode
):
    """Convert sparse voxel grid to triangle mesh using Flexible Dual Grid.

    This is a dual contouring variant where:
    1. Each voxel cell has a dual vertex at (cell_coord + offset) * voxel_size + origin
    2. For each edge between adjacent cells marked as intersected, a quad face is
       created from the 4 cells sharing that edge
    3. Quads are split into triangles using the quad_lerp parameter

    Args:
        coords: [N, 3] integer voxel coordinates
        vertex_offsets: [N, 3] per-voxel vertex position offsets (sigmoid-scaled)
        intersected: [N, 3] per-voxel edge intersection flags for +x, +y, +z edges
        quad_lerp: [N, 1] controls quad triangulation diagonal choice
        aabb: axis-aligned bounding box [[min], [max]], defaults to unit cube
        grid_size: grid resolution (int)
        train: if True, returns extra data for training

    Returns:
        (vertices, faces) tuple of torch tensors
    """
    device = coords.device

    if aabb is None:
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    voxel_size = (aabb[1] - aabb[0]) / grid_size

    # 1. Compute vertex positions for each voxel
    # pos = aabb[0] + (coord + offset) * voxel_size
    vertex_positions = aabb[0] + (coords.float() + vertex_offsets) * voxel_size

    # 2. Build a coordinate-to-index lookup
    # Pack coords into a single int64 for hashing
    N = coords.shape[0]
    cx = coords[:, 0].long()
    cy = coords[:, 1].long()
    cz = coords[:, 2].long()

    # Use a large prime-based hash to map 3D coords to flat keys
    # Ensure no collisions within typical grid sizes (up to 2048)
    PRIME_Y = 2053
    PRIME_Z = 2053 * 2053
    coord_keys = cx * PRIME_Z + cy * PRIME_Y + cz

    # Build hash map: coord_key -> vertex index
    # Using a tensor-based approach for GPU compatibility
    coord_to_idx = {}
    keys_np = coord_keys.cpu().numpy()
    for i in range(N):
        coord_to_idx[int(keys_np[i])] = i

    # 3. For each edge direction, find valid quads
    # Edge intersection semantics:
    #   intersected[:, 0] = x-edge: surface crosses between voxel and its +x neighbor
    #   intersected[:, 1] = y-edge: surface crosses between voxel and its +y neighbor
    #   intersected[:, 2] = z-edge: surface crosses between voxel and its +z neighbor
    #
    # For a dual-contouring quad, each intersected edge is surrounded by 4 cells.
    # For x-edge at (i,j,k): cells (i,j,k), (i,j-1,k), (i,j,k-1), (i,j-1,k-1)
    # For y-edge at (i,j,k): cells (i,j,k), (i-1,j,k), (i,j,k-1), (i-1,j,k-1)
    # For z-edge at (i,j,k): cells (i,j,k), (i-1,j,k), (i,j-1,k), (i-1,j-1,k)

    # The 4 neighboring cell offsets for each edge direction (relative to the cell
    # that owns the edge):
    # For x-edge: the other 3 cells are at (0,-1,0), (0,0,-1), (0,-1,-1) relative
    # For y-edge: the other 3 cells are at (-1,0,0), (0,0,-1), (-1,0,-1) relative
    # For z-edge: the other 3 cells are at (-1,0,0), (0,-1,0), (-1,-1,0) relative

    quad_offsets = {
        0: [(0, 0, 0), (0, -1, 0), (0, 0, -1), (0, -1, -1)],  # x-edge
        1: [(0, 0, 0), (-1, 0, 0), (0, 0, -1), (-1, 0, -1)],  # y-edge
        2: [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0)],  # z-edge
    }

    all_faces = []
    cx_np = cx.cpu().numpy()
    cy_np = cy.cpu().numpy()
    cz_np = cz.cpu().numpy()

    if isinstance(intersected, torch.Tensor):
        if intersected.dtype == torch.bool:
            inter_np = intersected.cpu().numpy()
        else:
            inter_np = (intersected > 0).cpu().numpy()
    else:
        inter_np = np.array(intersected) > 0

    ql_np = quad_lerp.detach().cpu().float().numpy().flatten() if isinstance(quad_lerp, torch.Tensor) else np.array(quad_lerp).flatten()

    for edge_dir in range(3):
        offsets = quad_offsets[edge_dir]
        # Find voxels with this edge intersected
        active = np.where(inter_np[:, edge_dir])[0]

        for idx in active:
            ix, iy, iz = int(cx_np[idx]), int(cy_np[idx]), int(cz_np[idx])

            # Find the 4 cells forming the quad
            cell_indices = []
            valid = True
            for dx, dy, dz in offsets:
                key = (ix + dx) * PRIME_Z + (iy + dy) * PRIME_Y + (iz + dz)
                cell_idx = coord_to_idx.get(key, -1)
                if cell_idx == -1:
                    valid = False
                    break
                cell_indices.append(cell_idx)

            if not valid:
                continue

            c0, c1, c2, c3 = cell_indices

            # Triangulate quad: split into 2 triangles
            # Use quad_lerp to choose diagonal
            ql_val = ql_np[idx]

            if ql_val > 0.5:
                # Diagonal c0-c3
                all_faces.append([c0, c1, c3])
                all_faces.append([c0, c3, c2])
            else:
                # Diagonal c1-c2
                all_faces.append([c0, c1, c2])
                all_faces.append([c1, c3, c2])

    if len(all_faces) == 0:
        log.warning("flexible_dual_grid_to_mesh: no faces generated")
        faces = torch.zeros((0, 3), dtype=torch.int64, device=device)
    else:
        faces = torch.tensor(all_faces, dtype=torch.int64, device=device)

    return vertex_positions, faces


def to_glb(
    vertices,
    faces,
    attr_volume,
    coords,
    attr_layout,
    aabb,
    voxel_size,
    decimation_target=500000,
    texture_size=2048,
    remesh=True,
    remesh_band=1.0,
    remesh_project=0.0,
    verbose=False,
):
    """Convert mesh + voxel data to a textured trimesh object (for GLB export).

    This is a convenience function that chains: simplify -> UV unwrap -> texture bake.
    Replacement for o_voxel.postprocess.to_glb.
    """
    import trimesh as _trimesh
    from .rocm_mesh_ops import CuMeshCompat, cuBVHCompat
    from .rocm_grid_sample import grid_sample_3d

    device = vertices.device

    if verbose:
        log.info(f"to_glb: {vertices.shape[0]} verts, {faces.shape[0]} faces")

    # 1. Simplify
    mesh_compat = CuMeshCompat()
    mesh_compat.init(vertices, faces)

    if remesh:
        mesh_compat.unify_face_orientations()

    mesh_compat.fill_holes()
    mesh_compat.unify_face_orientations()
    mesh_compat.simplify(decimation_target, verbose=verbose)
    mesh_compat.unify_face_orientations()

    simp_verts, simp_faces = mesh_compat.read()

    if verbose:
        log.info(f"to_glb: simplified to {simp_verts.shape[0]} verts, {simp_faces.shape[0]} faces")

    # 2. UV unwrap
    out_verts, out_faces, out_uvs, out_vmaps = mesh_compat.uv_unwrap(return_vmaps=True, verbose=verbose)

    # Compute normals
    mesh_compat.compute_vertex_normals()
    out_normals = mesh_compat.read_vertex_normals()[out_vmaps]

    out_verts_np = out_verts.float().numpy()
    out_faces_np = out_faces.numpy()
    out_uvs_np = out_uvs.float().numpy()
    out_normals_np = out_normals.float().numpy()

    if verbose:
        log.info(f"to_glb: UV unwrapped to {out_verts_np.shape[0]} verts")

    # 3. Simple vertex color baking from voxel data
    # Query PBR attributes at vertex positions
    aabb_t = torch.tensor(aabb, dtype=torch.float32, device=device)
    voxel_size_t = torch.tensor([voxel_size] * 3, dtype=torch.float32, device=device)
    grid_size_t = ((aabb_t[1] - aabb_t[0]) / voxel_size_t).round().int()

    query_pts = torch.tensor(out_verts_np, dtype=torch.float32, device=device)
    grid = ((query_pts - aabb_t[0]) / voxel_size_t).reshape(1, -1, 3)

    coords_with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)
    shape = torch.Size([1, attr_volume.shape[1], *grid_size_t.tolist()])

    vertex_attrs = grid_sample_3d(attr_volume, coords_with_batch, shape, grid, mode='trilinear')

    # Extract PBR channels
    base_color = vertex_attrs[:, attr_layout['base_color']].clamp(0, 1).cpu().float().numpy()
    vertex_colors = (base_color * 255).astype(np.uint8)

    # Build vertex color array (RGBA)
    alpha = vertex_attrs[:, attr_layout.get('alpha', slice(5, 6))].clamp(0, 1).cpu().float().numpy()
    alpha_u8 = (alpha * 255).astype(np.uint8)
    vertex_colors_rgba = np.concatenate([vertex_colors, alpha_u8], axis=1)

    # 4. Build trimesh
    result = _trimesh.Trimesh(
        vertices=out_verts_np,
        faces=out_faces_np,
        vertex_normals=out_normals_np,
        vertex_colors=vertex_colors_rgba,
        process=False,
    )

    # Attach UVs
    result.visual = _trimesh.visual.TextureVisuals(uv=out_uvs_np)

    if verbose:
        log.info(f"to_glb: complete ({len(result.vertices)} verts, {len(result.faces)} faces)")

    return result
