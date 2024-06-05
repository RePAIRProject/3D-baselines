import copy
import math

import torch
import numpy as np
from .loss import _valid_mean
from .chamfer import chamfer_distance
from .rotation import Rotation3D
from .transforms import transform_pc

import trimesh
from scipy.spatial import distance
from pyntcloud import PyntCloud
import pandas as pd
import open3d as o3d
from vedo import Points, Mesh, show, mag
from vedo.pyplot import histogram
from scipy.spatial import Delaunay
import pymeshlab

def precision(adj_pred, adj_true, areas_matrix):

    both = torch.tensor(np.logical_and(adj_pred, adj_true))
    both_areas = torch.sum(both * areas_matrix)
    true_areas = torch.sum(adj_true * areas_matrix)
    return both_areas / true_areas if true_areas > 0 else 0

def recall(adj_pred, adj_true, areas_matrix):
    both = torch.tensor(np.logical_and(adj_pred, adj_true))
    both_areas = torch.sum(both * areas_matrix)
    pred_areas = torch.sum(adj_pred * areas_matrix)
    return both_areas / pred_areas if pred_areas > 0 else 0
    
def f1(adj_pred, adj_true, areas_matrix):
    _prescision = precision(adj_pred, adj_true, areas_matrix)
    _recall = recall(adj_pred, adj_true, areas_matrix)
    return 2 * _prescision * _recall / (_prescision + _recall) if _prescision + _recall > 0 else 0

def create_mesh_from_points(pcd):
    # Create a Delaunay triangulation of the points
    #tri = Delaunay(points_array)
    # Create a Mesh from the triangulated points
    #mesh = Mesh([points_array, tri.simplices])
    #return Points.generate_mesh(points_array)
    # points = Points(points_array)
    # return pcd.reconstruct_surface(dims=100, radius=0.02)
    
    dists1 = []
    for p1 in pcd.coordinates:
        q1 = pcd.closest_point(p1, n=2)[1]
        dists1.append(mag(p1 - q1))
    histo1 = histogram(dists1, bins=25).clone2d()
    radius = histo1.mean * 10
    
    m = pcd.generate_delaunay3d(radius=radius)
    
    return m.tomesh().compute_normals()
    
    # return fill_holes(m.tomesh().compute_normals())

def compute_volume_value(vol1):
    dx1, dy1, dz1 = vol1.spacing() # voxel size
    counts1 = np.unique(vol1.pointdata[0], return_counts=True)
    n01, n11 = counts1[1]
    vol_value1 = dx1*dy1*dz1 * n11
    return vol_value1

def vedo2pymesh(vd_mesh):

    # m = pymeshlab.Mesh(vertex_matrix=vd_mesh.points(), face_matrix=vd_mesh.faces(), v_normals_matrix=vd_mesh.pointdata["Normals"], v_color_matrix=np.insert(vd_mesh.pointdata["RGB"]/255, 3, 1, axis=1))
    # vd_pcd.compute_normals_with_pca()
    m = pymeshlab.Mesh(vertex_matrix=vd_mesh.vertices, face_matrix=vd_mesh.faces(), v_normals_matrix=vd_mesh.point_normals)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)

    return ms

def pymesh2vedo(mlab_mesh):
    # color = mlab_mesh.vertex_color_matrix()[:, 0:-1]
    reco_mesh = Mesh(mlab_mesh)
    # reco_mesh.pointdata["RGB"] = (color * 255).astype(np.uint8)
    # reco_mesh.pointdata["Normals"] = mlab_mesh.vertex_normal_matrix().astype(np.float32)
    # reco_mesh.pointdata.select("RGB")

    return reco_mesh
    
def create_mesh(mesh_):
    m = o3d2pymesh(mesh_)
    m.generate_surface_reconstruction_screened_poisson(depth=8, pointweight=1, preclean=True)

    mlab_mesh = m.current_mesh()

    reco_mesh = pymesh2o3d(mlab_mesh)

    return reco_mesh

def o3d2vedo(o3d_mesh):
    m = Mesh([np.array(o3d_mesh.vertices), np.array(o3d_mesh.triangles)])

    # you could also check whether normals and color are present in order to port with the above vertices/faces
    return m

def vedo2open3d(vd_mesh):
    """
    Return an `open3d.geometry.TriangleMesh` version of
    the current mesh.

    Returns
    ---------
    open3d : open3d.geometry.TriangleMesh
      Current mesh as an open3d object.
    """
    # create from numpy arrays
    o3d_mesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vd_mesh.vertices))

    # I need to add some if check here in case color and normals info are not existing
    # o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vd_mesh.pointdata["RGB"]/255)
    # o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vd_mesh.pointdata["Normals"])

    return o3d_mesh

def o3d2pymesh(o3d_mesh):
    m = pymeshlab.Mesh(vertex_matrix=np.array(o3d_mesh.vertices), face_matrix=np.array(o3d_mesh.triangles),
                       v_normals_matrix=np.array(o3d_mesh.vertex_normals))#, v_color_matrix=np.insert(np.array(o3d_mesh.vertex_colors), 3, 1, axis=1))

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)

    return ms

def pymesh2o3d(pymesh_):
    # create from numpy arrays
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(pymesh_.vertex_matrix()),
        triangles=o3d.utility.Vector3iVector(pymesh_.face_matrix()))

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(pymesh_.vertex_color_matrix()[:, 0:-1])
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(pymesh_.vertex_normal_matrix())

    return o3d_mesh

def fill_holes(mesh_):
    m = vedo2pymesh(mesh_)
    m.meshing_close_holes(maxholesize=30, newfaceselected=False)

    mlab_mesh = m.current_mesh()

    reco_mesh = pymesh2vedo(mlab_mesh)

    return reco_mesh

def ball_pivoting_get_mesh(pcd):
    # radii = [0.005, 0.01, 0.02, 0.04]
    
    dists1 = []
    for p1 in pcd.coordinates:
        q1 = pcd.closest_point(p1, n=2)[1]
        dists1.append(mag(p1 - q1))
    # histo1 = histogram(dists1, bins=25).clone2d()
    # radius = histo1.mean * 10
    
    radii = histogram(dists1, bins=25).bins.tolist() * 10
    
    point_cloud = vedo2open3d(pcd)
    
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(50)
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, o3d.utility.DoubleVector(radii))
    
    
    return o3d2vedo(create_mesh(rec_mesh))


def Volume(points_array):
    # Create a Points object from the numpy array
    points = Points(points_array.cpu())
    
    # Create a mesh from the points
    # Note: This assumes the points form a closed surface (like a convex hull)
    mesh = Mesh(points).triangulate()
    
    # Compute the volume of the mesh
    volume = mesh.volume()
    
    return volume


def denormalize_point_cloud(normalized_points, min_vals, max_vals):
    # Calculate the range for each axis
    ranges = max_vals - min_vals
    # Find the axis with the greatest range
    max_range_axis = np.argmax(ranges)
    # Calculate the scaling factor
    scaling_factor = 1.0 / ranges[max_range_axis]
    # Calculate the midpoints
    midpoints = (min_vals + max_vals) / 2.0
    # Reverse the normalization

    denormalized_points = [(p / scaling_factor) + midpoints for p in normalized_points]
    
    return denormalized_points

@torch.no_grad()
def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, min_vals, max_vals, gt_adjacency=None, max_num_part=44):
    """Compute the `Part Accuracy` in the paper.

    We compute the per-part chamfer distance, and the distance lower than a
        threshold will be considered as correct.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], accuracy per data in the batch
    """
    adj_dist_th = 0.5

    B, P = pts.shape[:2]

    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)


    #------------- start
    precision_list = []
    recall_list = []
    f1_list = []
    Qpos_list = []
    

    for (pts_i, pts1_i, pts2_i, valids_i, min_vals_i, max_vals_i) in zip(pts.cpu(), pts1.cpu(), pts2.cpu(), valids, min_vals, max_vals):

        valids_i = valids_i.squeeze()
        valid_indices = (valids_i == 1).nonzero(as_tuple=True)[0]
        pts_i =pts_i[valid_indices]
        pts1_i =pts1_i[valid_indices]
        pts2_i =pts2_i[valid_indices]

        denormalized_points = denormalize_point_cloud(pts_i, min_vals_i.cpu(), max_vals_i.cpu())
        pts_denormed = torch.stack([torch.tensor(p) for p in denormalized_points])
        pts_denormed = torch.tensor(pts_denormed, dtype=torch.float32)

        denormalized_points1 = denormalize_point_cloud(pts1_i, min_vals_i.cpu(), max_vals_i.cpu())
        pts1_i = torch.stack([torch.tensor(p) for p in denormalized_points1])
        pts1_i = torch.tensor(pts1_i, dtype=torch.float32)

        denormalized_points2 = denormalize_point_cloud(pts2_i, min_vals_i.cpu(), max_vals_i.cpu())
        pts2_i = torch.stack([torch.tensor(p) for p in denormalized_points2])
        pts2_i = torch.tensor(pts2_i, dtype=torch.float32)
        #breakpoint()

        num_parts = pts_denormed.shape[0]
        gt_adjacency = torch.zeros((max_num_part, max_num_part))

        for part_i in range(num_parts):
            for part_j in range(num_parts):
                dist = distance.cdist(pts_denormed[part_i], pts_denormed[part_j]).min()
                if dist < adj_dist_th:
                    gt_adjacency[part_i, part_j] = 1
        
        # calculate the volume
    
        volumes = torch.zeros(gt_adjacency.shape[0])
        
        
        for idx, pc in enumerate(pts1_i):
            df = pd.DataFrame(pc.cpu(), columns=['x', 'y', 'z'])
            #print(f'df: {df}')
            #print(f'Duplicates: {df[df.duplicated()]}')
            pynt = PyntCloud(df)
            voxel_id = pynt.add_structure('convex_hull')
            volume = pynt.structures[voxel_id].volume
            #print(f"volume: {volume}")
            volumes[idx] = volume
        
        volumes_array = torch.tensor(volumes[:, np.newaxis] + volumes[np.newaxis, :])
    
        pred_adjacency = torch.zeros_like(gt_adjacency)
    
        for part_i in range(num_parts):
            for part_j in range(num_parts):
                dist = distance.cdist(pts1_i[part_i].cpu(), pts1_i[part_j].cpu()).min()
                if dist < adj_dist_th:
                    pred_adjacency[part_i, part_j] = 1
        
        # pred_adjacency = torch.tensor(gt_adjacency) 
        prec_calc = precision(pred_adjacency, gt_adjacency, volumes_array)
        recall_calc = recall(pred_adjacency, gt_adjacency, volumes_array)
        f1_calc = f1(pred_adjacency, gt_adjacency, volumes_array)

        precision_list.append(prec_calc.item())
        recall_list.append(recall_calc.item())
        f1_list.append(f1_calc.item())
    
                    
        intersection_volume_lst = []
        for t, (i, j) in enumerate(zip(pts1_i.cpu(), pts2_i.cpu())):
            
            pcd1 = Points(i)
            pcd2 = Points(j).c('r')
        
            msh1 = ball_pivoting_get_mesh(pcd1)
            msh2 = ball_pivoting_get_mesh(pcd2)
            try:
                surf1 = msh1
                surf1.color("blue5").alpha(0.1)
                vol1 = surf1.binarize()
                vol_value1 = compute_volume_value(vol1)
                surf2 = msh2
                surf2.color("red5").alpha(0.1)
                vol2 = surf2.binarize()
                vol_value2 = compute_volume_value(vol2)
                
                vol = vol1.operation('and', vol2)
                vol_value = compute_volume_value(vol)
                
                intersection_volume_lst.append((vol_value / vol_value2, vol_value2))
            except:
                intersection_volume_lst.append((0, vol_value2))
        
        total_volume = sum(x[1] for x in intersection_volume_lst)
        volume_weights = [x[1] / total_volume for x in intersection_volume_lst]
        
        Qpos = 0
        
        for (i, j), w in zip(intersection_volume_lst, volume_weights):
            Qpos += w * i
        
        Qpos_list.append(float(Qpos))


    #------------- end

    pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
    pts2 = pts2.flatten(0, 1)
    dist1, dist2 = chamfer_distance(pts1, pts2)  # [B*P, N]
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(B, P).type_as(pts)

    # part with CD < `thre` is considered correct
    thre = 0.01
    acc = (loss_per_data < thre) & (valids == 1)
    # the official code is doing avg per-shape acc (not per-part)
    acc = acc.sum(-1) / (valids == 1).sum(-1)

    return acc, torch.FloatTensor(precision_list).cuda(), torch.FloatTensor(recall_list).cuda(), torch.FloatTensor(f1_list).cuda(), torch.FloatTensor(Qpos_list).cuda()


@torch.no_grad()
def calc_connectivity_acc(trans, rot, contact_points):
    """Compute the `Connectivity Accuracy` in the paper.

    We transform pre-computed connected point pairs using predicted pose, then
        we compare the distance between them.
    Distance lower than a threshold will be considered as correct.

    Args:
        trans: [B, P, 3]
        rot: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        contact_points: [B, P, P, 4], pairwise contact matrix.
            First item is 1 --> two parts are connecting, 0 otherwise.
            Last three items are the contacting point coordinate.

    Returns:
        [B], accuracy per data in the batch
    """
    B, P, _ = trans.shape
    thre = 0.01
    # get torch.Tensor of rotation for simplicity
    rot_type = rot.rot_type
    rot = rot.rot

    def get_min_l2_dist(points1, points2, trans1, trans2, rot1, rot2):
        """Compute the min L2 distance between two set of points."""
        # points1/2: [num_contact, num_symmetry, 3]
        # trans/rot: [num_contact, 3/4/(3, 3)]
        points1 = transform_pc(trans1, rot1, points1, rot_type=rot_type)
        points2 = transform_pc(trans2, rot2, points2, rot_type=rot_type)
        dist = ((points1[:, :, None] - points2[:, None, :])**2).sum(-1)
        return dist.min(-1)[0].min(-1)[0]  # [num_contact]

    # find all contact points
    mask = (contact_points[..., 0] == 1)  # [B, P, P]
    # points1 = contact_points[mask][..., 1:]
    # TODO: more efficient way of getting paired contact points?
    points1, points2, trans1, trans2, rot1, rot2 = [], [], [], [], [], []
    for b in range(B):
        for i in range(P):
            for j in range(P):
                if mask[b, i, j]:
                    points1.append(contact_points[b, i, j, 1:])
                    points2.append(contact_points[b, j, i, 1:])
                    trans1.append(trans[b, i])
                    trans2.append(trans[b, j])
                    rot1.append(rot[b, i])
                    rot2.append(rot[b, j])
    points1 = torch.stack(points1, dim=0)  # [n, 3]
    points2 = torch.stack(points2, dim=0)  # [n, 3]
    # [n, 3/4/(3, 3)], corresponding translation and rotation
    trans1, trans2 = torch.stack(trans1, dim=0), torch.stack(trans2, dim=0)
    rot1, rot2 = torch.stack(rot1, dim=0), torch.stack(rot2, dim=0)
    points1 = torch.stack(get_sym_point_list(points1), dim=1)  # [n, sym, 3]
    points2 = torch.stack(get_sym_point_list(points2), dim=1)  # [n, sym, 3]
    dist = get_min_l2_dist(points1, points2, trans1, trans2, rot1, rot2)
    acc = (dist < thre).sum().float() / float(dist.numel())

    # the official code is doing avg per-contact_point acc (not per-shape)
    # so we tile the `acc` to [B]
    acc = torch.ones(B).type_as(trans) * acc
    return acc


def get_sym_point(point, x, y, z):
    """Get the symmetry point along one or many of xyz axis."""
    point = copy.deepcopy(point)
    if x == 1:
        point[..., 0] = -point[..., 0]
    if y == 1:
        point[..., 1] = -point[..., 1]
    if z == 1:
        point[..., 2] = -point[..., 2]
    return point


def get_sym_point_list(point, sym=None):
    """Get all poissible symmetry point as a list.
    `sym` is a list indicating the symmetry axis of point.
    """
    if sym is None:
        sym = [1, 1, 1]
    else:
        if not isinstance(sym, (list, tuple)):
            sym = sym.tolist()
        sym = [int(i) for i in sym]
    point_list = []
    for x in range(sym[0] + 1):
        for y in range(sym[1] + 1):
            for z in range(sym[2] + 1):
                point_list.append(get_sym_point(point, x, y, z))

    return point_list


@torch.no_grad()
def trans_metrics(trans1, trans2, valids, metric):
    """Evaluation metrics for transformation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    if metric == 'mse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def rot_metrics(rot1, rot2, valids, metric):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    deg1 = rot1.to_euler(to_degree=True)  # [B, P, 3]
    deg2 = rot2.to_euler(to_degree=True)
    diff1 = (deg1 - deg2).abs()
    diff2 = 360. - (deg1 - deg2).abs()
    # since euler angle has the discontinuity at 180
    diff = torch.minimum(diff1, diff2)
    if metric == 'mse':
        metric_per_data = diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = diff.pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = diff.abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data
