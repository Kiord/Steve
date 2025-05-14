import numpy as np
from numba import njit, int32, float32
from collections import defaultdict
from numba.typed import List
from time import time
from constants import EPS

@njit
def surface_area(bmin, bmax):
    d = bmax - bmin
    return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

@njit
def compute_bounds_centroid(centroids, indices):
    bmin = centroids[indices[0]].copy()
    bmax = bmin.copy()
    for i in indices[1:]:
        for k in range(3):
            bmin[k] = min(bmin[k], centroids[i][k])
            bmax[k] = max(bmax[k], centroids[i][k])
    return bmin, bmax

@njit
def sah_split_qbvh(centroids, indices, axis, bucket_count=16):
    N = len(indices)
    if N <= 1:
        return [indices]

    bmin, bmax = compute_bounds_centroid(centroids, indices)
    extent = bmax[axis] - bmin[axis]
    if extent == 0:
        return [indices]

    bucket_indices = np.full((bucket_count, N), -1, dtype=np.int32)
    bucket_counts = np.zeros(bucket_count, dtype=np.int32)

    # Fill buckets
    for idx in indices:
        offset = (centroids[idx][axis] - bmin[axis]) / extent
        b = min(bucket_count - 1, int(offset * bucket_count))
        i = bucket_counts[b]
        bucket_indices[b, i] = idx
        bucket_counts[b] += 1

    best_cost = 1e10
    best_split = None

    def compute_sah(groups):
        total_sa = surface_area(bmin, bmax)
        cost = 0.0
        for g in groups:
            if len(g) == 0:
                continue
            gmin, gmax = compute_bounds_centroid(centroids, g)
            cost += (surface_area(gmin, gmax) / total_sa) * len(g)
        return cost

    for i in range(1, bucket_count - 2):
        for j in range(i + 1, bucket_count - 1):
            for k in range(j + 1, bucket_count):
                groups = []
                for group_range in [(0, i), (i, j), (j, k), (k, bucket_count)]:
                    start, end = group_range
                    flat = []
                    for b in range(start, end):
                        count = bucket_counts[b]
                        for n in range(count):
                            flat.append(bucket_indices[b, n])
                    groups.append(np.array(flat, dtype=np.int32))
                cost = compute_sah(groups)
                if cost < best_cost:
                    best_cost = cost
                    best_split = groups

    return best_split if best_split is not None else [indices]



@njit
def _build_qbvh_sah(centroids, triangle_count, max_leaf_size=4, bucket_count=16):
    max_nodes = 2 * triangle_count
    is_leaf = np.zeros(max_nodes, dtype=int32)
    child_idx = np.full((max_nodes, 4), -1, dtype=int32)
    child_count = np.zeros(max_nodes, dtype=int32)
    aabb_min = np.full((max_nodes, 3), np.inf, dtype=float32)
    aabb_max = np.full((max_nodes, 3), -np.inf, dtype=float32)
    output_triangles_order = np.empty(triangle_count, dtype=int32)

    node_ptr = 1
    reorder_ptr = 0

    stack = List()
    stack.append((0, np.arange(triangle_count, dtype=int32)))

    while len(stack) > 0:
        node_id, indices = stack.pop()
        count = len(indices)

        # Compute bounds
        bmin, bmax = compute_bounds_centroid(centroids, indices)
        aabb_min[node_id] = bmin
        aabb_max[node_id] = bmax

        if count <= max_leaf_size:
            is_leaf[node_id] = 1
            child_count[node_id] = count
            for i in range(count):
                output_triangles_order[reorder_ptr] = indices[i]
                child_idx[node_id, i] = reorder_ptr
                reorder_ptr += 1
        else:
            axis = 0
            extents = bmax - bmin
            if extents[1] > extents[0] and extents[1] > extents[2]:
                axis = 1
            elif extents[2] > extents[0]:
                axis = 2

            splits = sah_split_qbvh(centroids, indices, axis, bucket_count)
            child_count[node_id] = len(splits)

            for i in range(len(splits)):
                split = splits[i]
                if len(split) == 0:
                    continue
                cid = node_ptr
                node_ptr += 1
                child_idx[node_id, i] = cid
                stack.append((cid, split))

    return (
        is_leaf[:node_ptr],
        child_idx[:node_ptr],
        child_count[:node_ptr],
        aabb_min[:node_ptr],
        aabb_max[:node_ptr],
        output_triangles_order
    )

@njit
def _prepare_triangle_data(triangles):
    N = len(triangles)
    tri_centroids = np.zeros((N, 3), dtype=float32)
    tri_aabb_min = np.zeros((N, 3), dtype=float32)
    tri_aabb_max = np.zeros((N, 3), dtype=float32)
    for n in range(N):
        tri_centroids[n] = (triangles[n, 0] + triangles[n, 1] + triangles[n, 2]) / 3.0
        for k in range(3):
            tri_aabb_min[n, k] = min(
                triangles[n][0][k],
                triangles[n][1][k],
                triangles[n][2][k]
            )
            tri_aabb_max[n, k] = max(
                triangles[n][0][k],
                triangles[n][1][k],
                triangles[n][2][k]
            )
    return tri_centroids, tri_aabb_min, tri_aabb_max
    # sorted_by_axis = np.zeros((3,N), dtype=np.int32)
    # for k in range(3):
    #     sorted_by_axis[k] = np.argsort(tri_centroids[:, k]).astype(np.int32)
    # return tri_centroids, tri_aabb_min, tri_aabb_max, sorted_by_axis



@njit(cache=True)
def _build_binary_sah_bvh(triangles, max_leaf_size=4):
    tri_centroids, tri_aabb_min, tri_aabb_max = _prepare_triangle_data(triangles)

    N = len(tri_centroids)
    max_nodes = 2 * N
    is_leaf = np.zeros(max_nodes, dtype=int32)
    left_child = np.full(max_nodes, -1, dtype=int32)
    right_child = np.full(max_nodes, -1, dtype=int32)

    aabb_min = np.full((max_nodes, 3), np.inf, dtype=float32)
    aabb_max = np.full((max_nodes, 3), -np.inf, dtype=float32)

    depth = np.full(max_nodes, -1, dtype=int32)

    output_triangles_order = np.empty(N, dtype=int32)
    reorder_ptr = 0

    stack = [(0, np.arange(N, dtype=int32), 0)]  # (node_id, triangle indices, depth)
    node_ptr = 1

    def compute_surface_area(min_corner, max_corner):
        d = max_corner - min_corner
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    def compute_bounds(indices):
        bmin = tri_aabb_min[indices[0]].copy()
        bmax = tri_aabb_max[indices[0]].copy()
        for i in range(1, len(indices)):
            idx = indices[i]
            for k in range(3):
                bmin[k] = min(bmin[k], tri_aabb_min[idx][k])
                bmax[k] = max(bmax[k], tri_aabb_max[idx][k])
        return bmin, bmax

    while len(stack) > 0:
        node_id, node_indices, node_depth = stack.pop()
        depth[node_id] = node_depth
        count = len(node_indices)

        bmin, bmax = compute_bounds(node_indices)
        aabb_min[node_id] = bmin
        aabb_max[node_id] = bmax

        if count <= max_leaf_size:
            is_leaf[node_id] = 1
            left_child[node_id] = reorder_ptr      # start index in output_triangles_order
            right_child[node_id] = count           # number of triangles in this leaf
            for i in range(count):
                output_triangles_order[reorder_ptr] = node_indices[i]
                reorder_ptr += 1
        else:
            best_cost = np.inf
            best_split = -1

            for axis in range(3):
                sort_keys = np.empty(count, dtype=float32)
                for i in range(count):
                    sort_keys[i] = tri_centroids[node_indices[i]][axis]
                sort_indices = np.argsort(sort_keys)
                sorted_ids = node_indices[sort_indices]

                left_bounds = np.full((count - 1, 2, 3), 0.0, dtype=float32)
                right_bounds = np.full((count - 1, 2, 3), 0.0, dtype=float32)

                lbmin = tri_aabb_min[sorted_ids[0]].copy()
                lbmax = tri_aabb_max[sorted_ids[0]].copy()
                for i in range(1, count):
                    idx = sorted_ids[i-1]
                    for k in range(3):
                        lbmin[k] = min(lbmin[k], tri_aabb_min[idx][k])
                        lbmax[k] = max(lbmax[k], tri_aabb_max[idx][k])
                    left_bounds[i-1, 0] = lbmin
                    left_bounds[i-1, 1] = lbmax

                rbmin = tri_aabb_min[sorted_ids[-1]].copy()
                rbmax = tri_aabb_max[sorted_ids[-1]].copy()
                for i in range(count - 2, -1, -1):
                    idx = sorted_ids[i+1]
                    for k in range(3):
                        rbmin[k] = min(rbmin[k], tri_aabb_min[idx][k])
                        rbmax[k] = max(rbmax[k], tri_aabb_max[idx][k])
                    right_bounds[i, 0] = rbmin
                    right_bounds[i, 1] = rbmax

                total_area = compute_surface_area(bmin, bmax)
                for i in range(1, count):
                    l_area = compute_surface_area(left_bounds[i-1, 0], left_bounds[i-1, 1])
                    r_area = compute_surface_area(right_bounds[i-1, 0], right_bounds[i-1, 1])
                    cost = (i * l_area + (count - i) * r_area) / total_area
                    if cost < best_cost:
                        best_cost = cost
                        best_split = i
                        best_sorted_ids = sorted_ids

            left_ids = best_sorted_ids[:best_split]
            right_ids = best_sorted_ids[best_split:]

            left_id = node_ptr
            node_ptr += 1
            right_id = node_ptr
            node_ptr += 1

            left_child[node_id] = left_id
            right_child[node_id] = right_id

            stack.append((right_id, right_ids, node_depth+1))
            stack.append((left_id, left_ids, node_depth+1))

    return (
        is_leaf[:node_ptr],
        left_child[:node_ptr],
        right_child[:node_ptr],
        aabb_min[:node_ptr],
        aabb_max[:node_ptr],
        depth[:node_ptr],
        output_triangles_order
    )


@njit
def reduce_min_bounds(bounds):
    n = bounds.shape[0]
    out = bounds[0].copy()
    for i in range(1, n):
        for k in range(3):
            out[k] = min(out[k], bounds[i, k])
    return out

@njit
def reduce_max_bounds(bounds):
    n = bounds.shape[0]
    out = bounds[0].copy()
    for i in range(1, n):
        for k in range(3):
            out[k] = max(out[k], bounds[i, k])
    return out


@njit(cache=True)
def _build_binary_binned_sah_bvh(triangles, max_leaf_size=4, num_bins=16):
    tri_centroids, tri_aabb_min, tri_aabb_max = _prepare_triangle_data(triangles)

    N = len(tri_centroids)
    max_nodes = 2 * N
    is_leaf = np.zeros(max_nodes, dtype=int32)
    left_child = np.full(max_nodes, -1, dtype=int32)
    right_child = np.full(max_nodes, -1, dtype=int32)
    depth = np.full(max_nodes, -1, dtype=int32)

    aabb_min = np.full((max_nodes, 3), np.inf, dtype=float32)
    aabb_max = np.full((max_nodes, 3), -np.inf, dtype=float32)

    output_triangles_order = np.empty(N, dtype=int32)
    reorder_ptr = 0
    stack = [(0, np.arange(N, dtype=int32), 0)]
    node_ptr = 1

    def compute_surface_area(min_corner, max_corner):
        d = max_corner - min_corner
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    def compute_bounds(indices):
        bmin = tri_aabb_min[indices[0]].copy()
        bmax = tri_aabb_max[indices[0]].copy()
        for i in range(1, len(indices)):
            idx = indices[i]
            for k in range(3):
                bmin[k] = min(bmin[k], tri_aabb_min[idx][k])
                bmax[k] = max(bmax[k], tri_aabb_max[idx][k])
        return bmin, bmax

    while len(stack) > 0:
        node_id, node_indices, node_depth = stack.pop()
        depth[node_id] = node_depth
        count = len(node_indices)

        bmin, bmax = compute_bounds(node_indices)
        aabb_min[node_id] = bmin
        aabb_max[node_id] = bmax

        if count <= max_leaf_size:
            is_leaf[node_id] = 1
            left_child[node_id] = reorder_ptr
            right_child[node_id] = count
            for i in range(count):
                output_triangles_order[reorder_ptr] = node_indices[i]
                reorder_ptr += 1
        else:
            best_cost = np.inf
            best_split = -1
            best_axis = 0
            bin_edges = np.zeros(num_bins + 1, dtype=float32)
            bin_counts = np.zeros((3, num_bins), dtype=int32)
            bin_bounds_min = np.full((3, num_bins, 3), np.inf, dtype=float32)
            bin_bounds_max = np.full((3, num_bins, 3), -np.inf, dtype=float32)

            extent = bmax - bmin
            extent[extent == 0.0] = 1e-5  # avoid div0

            for axis in range(3):
                # Reset bins
                bin_counts[axis, :] = 0
                bin_bounds_min[axis, :, :] = np.inf
                bin_bounds_max[axis, :, :] = -np.inf

                for idx in node_indices:
                    c = tri_centroids[idx]
                    offset = (c[axis] - bmin[axis]) / extent[axis]
                    b = min(num_bins - 1, int(offset * num_bins))
                    bin_counts[axis, b] += 1
                    for k in range(3):
                        bin_bounds_min[axis, b, k] = min(bin_bounds_min[axis, b, k], tri_aabb_min[idx][k])
                        bin_bounds_max[axis, b, k] = max(bin_bounds_max[axis, b, k], tri_aabb_max[idx][k])

                total_area = compute_surface_area(bmin, bmax)

                for i in range(1, num_bins):
                    l_count = np.sum(bin_counts[axis, :i])
                    r_count = np.sum(bin_counts[axis, i:])

                    if l_count == 0 or r_count == 0:
                        continue

                    lmin = reduce_min_bounds(bin_bounds_min[axis, :i])
                    lmax = reduce_max_bounds(bin_bounds_max[axis, :i])
                    rmin = reduce_min_bounds(bin_bounds_min[axis, i:])
                    rmax = reduce_max_bounds(bin_bounds_max[axis, i:])

                    l_area = compute_surface_area(lmin, lmax)
                    r_area = compute_surface_area(rmin, rmax)

                    cost = (l_count * l_area + r_count * r_area) / total_area
                    if cost < best_cost:
                        best_cost = cost
                        best_split = i
                        best_axis = axis

            # Partition triangles into left/right bins
            left_ids = []
            right_ids = []
            for idx in node_indices:
                c = tri_centroids[idx]
                offset = (c[best_axis] - bmin[best_axis]) / extent[best_axis]
                b = min(num_bins - 1, int(offset * num_bins))
                if b < best_split:
                    left_ids.append(idx)
                else:
                    right_ids.append(idx)

            left_ids = np.array(left_ids, dtype=int32)
            right_ids = np.array(right_ids, dtype=int32)

            left_id = node_ptr
            node_ptr += 1
            right_id = node_ptr
            node_ptr += 1

            left_child[node_id] = left_id
            right_child[node_id] = right_id

            stack.append((right_id, right_ids, node_depth + 1))
            stack.append((left_id, left_ids, node_depth + 1))

    return (
        is_leaf[:node_ptr],
        left_child[:node_ptr],
        right_child[:node_ptr],
        aabb_min[:node_ptr],
        aabb_max[:node_ptr],
        depth[:node_ptr],
        output_triangles_order
    )

@njit(cache=True)
def _build_binary_median_bvh(triangles, max_leaf_size=4, num_bins=16):
    tri_centroids, tri_aabb_min, tri_aabb_max = _prepare_triangle_data(triangles)

    N = len(tri_centroids)
    max_nodes = 2 * N
    is_leaf = np.zeros(max_nodes, dtype=int32)
    left_child = np.full(max_nodes, -1, dtype=int32)
    right_child = np.full(max_nodes, -1, dtype=int32)
    depth = np.full(max_nodes, -1, dtype=int32)

    aabb_min = np.full((max_nodes, 3), np.inf, dtype=float32)
    aabb_max = np.full((max_nodes, 3), -np.inf, dtype=float32)

    output_triangles_order = np.empty(N, dtype=int32)
    reorder_ptr = 0
    stack = [(0, np.arange(N, dtype=int32), 0)]
    node_ptr = 1

    def compute_surface_area(min_corner, max_corner):
        d = max_corner - min_corner
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    def compute_bounds(indices):
        bmin = tri_aabb_min[indices[0]].copy()
        bmax = tri_aabb_max[indices[0]].copy()
        for i in range(1, len(indices)):
            idx = indices[i]
            for k in range(3):
                bmin[k] = min(bmin[k], tri_aabb_min[idx][k])
                bmax[k] = max(bmax[k], tri_aabb_max[idx][k])
        return bmin, bmax

    while len(stack) > 0:
        node_id, node_indices, node_depth = stack.pop()
        depth[node_id] = node_depth
        count = len(node_indices)

        bmin, bmax = compute_bounds(node_indices)
        aabb_min[node_id] = bmin
        aabb_max[node_id] = bmax

        if count <= max_leaf_size:
            is_leaf[node_id] = 1
            left_child[node_id] = reorder_ptr
            right_child[node_id] = count
            for i in range(count):
                output_triangles_order[reorder_ptr] = node_indices[i]
                reorder_ptr += 1
        else:

            best_cost = np.inf

            right_ids = None
            left_ids = None

            for axis in range(3):
                centroids_axis = np.empty(len(node_indices), dtype=tri_centroids.dtype)
                for i in range(len(node_indices)):
                    centroids_axis[i] = tri_centroids[node_indices[i], axis]

                tri_order = np.argsort(centroids_axis)
                right_candidates = node_indices[tri_order[:count//2]]
                right_min_bounds, right_max_bounds = compute_bounds(right_candidates)
                left_candidates = node_indices[tri_order[count//2:]]
                left_min_bounds, left_max_bounds = compute_bounds(left_candidates)
                
                right_area = compute_surface_area(right_min_bounds, right_max_bounds)
                left_area = compute_surface_area(left_min_bounds, left_max_bounds)
                cost = left_area + right_area
                if cost < best_cost:
                    best_cost = cost
                    right_ids = right_candidates
                    left_ids = left_candidates

            left_id = node_ptr
            node_ptr += 1
            right_id = node_ptr
            node_ptr += 1

            left_child[node_id] = left_id
            right_child[node_id] = right_id

            stack.append((right_id, right_ids, node_depth + 1))
            stack.append((left_id, left_ids, node_depth + 1))

    return (
        is_leaf[:node_ptr],
        left_child[:node_ptr],
        right_child[:node_ptr],
        aabb_min[:node_ptr],
        aabb_max[:node_ptr],
        depth[:node_ptr],
        output_triangles_order
    )

def build_bvh(triangles, max_leaf_size=4, bvh_type='binned'):

    bt = time()
    if bvh_type == 'binned':
        bvh_tuple = _build_binary_binned_sah_bvh(triangles, max_leaf_size)
    elif bvh_type == 'sweep':
        bvh_tuple =  _build_binary_sah_bvh(triangles, max_leaf_size)
    elif bvh_type == 'median':
        bvh_tuple =  _build_binary_median_bvh(triangles, max_leaf_size)
    else:
        raise ValueError(f"Wrong bvh_type ({bvh_type})")
    is_leaf, left_child, right_child, aabb_min, aabb_max, depth, output_triangles_order = bvh_tuple

    bvh_dict = dict(
        is_leaf=is_leaf, left_child=left_child, right_child=right_child, 
        aabb_min=aabb_min, aabb_max=aabb_max, depth=depth, 
        max_leaf_size=max_leaf_size, bvh_type=bvh_type,
        output_triangles_order=output_triangles_order
    )
    print(time() - bt)
    return bvh_dict


def print_bvh_summary(bvh_dict):
    """
    Print summary statistics for a binary BVH:
    - Total/internal/leaf node counts
    - Depth histogram and max depth
    - Internal node area ratio stats
    - Leaf triangle count mean/std/min/max
    """
    is_leaf, left_child, right_child, aabb_min, aabb_max, depth, max_leaf_size, type, output_triangles_order = bvh_dict.values()
    num_nodes = len(is_leaf)

    def compute_surface_area(min_corner, max_corner):
        d = max_corner - min_corner
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    depth_histogram = defaultdict(int)
    area_ratios = []
    leaf_triangle_counts = []

    def traverse(node_id, depth, tri_ptr):
        depth_histogram[depth] += 1
        if is_leaf[node_id]:
            count = right_child[node_id]
            leaf_triangle_counts.append(count)
            tri_ptr[0] += count
            return

        parent_area = compute_surface_area(aabb_min[node_id], aabb_max[node_id])
        l = left_child[node_id]
        r = right_child[node_id]
        if l != -1 and r != -1:
            left_area = compute_surface_area(aabb_min[l], aabb_max[l])
            right_area = compute_surface_area(aabb_max[r], aabb_max[r])
            if parent_area > 0:
                area_ratios.append((left_area + right_area) / parent_area)
        if l != -1:
            traverse(l, depth + 1, tri_ptr)
        if r != -1:
            traverse(r, depth + 1, tri_ptr)

    triangle_ptr = [0]
    traverse(0, 0, triangle_ptr)

    print("=== BVH Statistics ===")
    print(f"Total nodes: {num_nodes}")
    print(f"Leaf nodes: {np.sum(is_leaf)}")
    print(f"Internal nodes: {np.sum(is_leaf == 0)}")
    max_depth = max(depth_histogram.keys()) if depth_histogram else 0
    print(f"Max depth: {max_depth}")
    print("Depth histogram:")
    for depth in sorted(depth_histogram.keys()):
        print(f"  Depth {depth}: {depth_histogram[depth]} nodes")

    if area_ratios:
        print("\nInternal node area ratio (child area sum / parent area):")
        print(f"  Average: {np.mean(area_ratios):.4f}")
        print(f"  Std Dev: {np.std(area_ratios):.4f}")
        print(f"  Minimum: {np.min(area_ratios):.4f}")
        print(f"  Maximum: {np.max(area_ratios):.4f}")

    if leaf_triangle_counts:
        print("\nTriangles per leaf node:")
        print(f"  Average: {np.mean(leaf_triangle_counts):.2f}")
        print(f"  Std Dev: {np.std(leaf_triangle_counts):.2f}")
        print(f"  Minimum: {np.min(leaf_triangle_counts)}")
        print(f"  Maximum: {np.max(leaf_triangle_counts)}")

    print(f"type : {type}")
    print(f"Max leaf size : {max_leaf_size}")

def aabbs_to_mesh(aabb_min, aabb_max):
    N = aabb_min.shape[0]

    # 8 cube corners (unit cube)
    cube_offsets = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ], dtype=np.float32)  # (8, 3)

    # 12 faces (2 triangles per face)
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], dtype=np.int32)  # (12, 3)

    # Compute box sizes: (N, 3)
    sizes = aabb_max - aabb_min  # (N, 3)

    # Expand sizes to match each box's 8 vertices
    # (N, 1, 3) * (1, 8, 3) → (N, 8, 3)
    box_vertices = aabb_min[:, None, :] + cube_offsets[None, :, :] * sizes[:, None, :]

    # Reshape to flat (N*8, 3)
    vertices = box_vertices.reshape(-1, 3)

    # Repeat faces for each box with offset
    face_offsets = (np.arange(N) * 8)[:, None, None]  # (N, 1, 1)
    faces = cube_faces[None, :, :] + face_offsets  # (N, 12, 3)
    faces = faces.reshape(-1, 3)  # (N*12, 3)

    return tm.Trimesh(vertices, faces)

if __name__ == '__main__':
    import trimesh as tm
    from time import time
    mesh = tm.load_mesh('data/meshes/bunny.stl')
    print(mesh)
    bvh_dict    = build_bvh(mesh.triangles.copy(), max_leaf_size=4, bvh_type='median')
    bvh_dict    = build_bvh(mesh.triangles.copy(), max_leaf_size=4, bvh_type='median')
    bvh_dict    = build_bvh(mesh.triangles.copy(), max_leaf_size=4, bvh_type='median')
    is_leaf, left_child, right_child, aabb_min, aabb_max, depth, max_leaf_size, binned, output_triangles_order = bvh_dict.values()
    
    print_bvh_summary(bvh_dict)

    import pyvista as pv
    plotter = pv.Plotter()
    max_depth = depth.max()

    def cb(value):
        d = int(round(value))
        if d > max_depth:
            mask = is_leaf == 1
        else:
            mask = depth == d
        aabb_mesh = aabbs_to_mesh(aabb_min[mask], aabb_max[mask])
        plotter.add_mesh(aabb_mesh, name='aabb', color=(1.0,0,0), opacity=0.1)

    plotter.add_mesh(mesh, opacity=0.5)#style='wireframe', edge_color=(0,1.0,0))

    plotter.add_slider_widget(cb, (0, max_depth + 1), 0, interaction_event='always')

    plotter.show()

