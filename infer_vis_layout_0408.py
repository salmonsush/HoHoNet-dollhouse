# for infer_layout
import os, sys, time, glob
import argparse
import importlib
from tqdm import tqdm
from imageio import imread, imwrite
import torch
import numpy as np
import matplotlib.pyplot as plt 
from lib.config import config, update_config
import cv2

# for vis_layout
import json
import open3d as o3d
from PIL import Image
from scipy.signal import correlate2d
from scipy.ndimage import shift
from lib.misc.post_proc import np_coor2xy, np_coorx2u, np_coory2v
from eval_layout import layout_2_depth

# Database
door_list = [
    {'id': 0, 'room': 0, 'door_point': (290,339), 'direction': 'L'},
    {'id': 1, 'room': 0, 'door_point': (290,526), 'direction': 'F'},
    {'id': 2, 'room': 1, 'door_point': (295,960), 'direction': 'B'},
    {'id': 3, 'room': 2, 'door_point': (295,727), 'direction': 'R'},
]

Localization = [
    {'spot': 0, 'pair': ('0R', '2R')},
    {'spot': 1, 'pair': ('0L', '1B')},
]

def get_vertices(cor_id, xs, ys, zs):
    cor_id = cor_id.tolist()
    points = np.empty((0, 3))
    for point in cor_id:
        x = xs[point[0], point[1]]
        y = ys[point[0], point[1]]
        z = zs[point[0], point[1]]
        points = np.vstack((points, np.array([x, y, z])))
    return points

def Getdoor(x, y, points):
    xyz = points[x-1:x+1, y-1:y+1]
    
    # door_coorx = xs[x, y]
    # door_coory = ys[x, y]
    # door_coorz = zs[x, y]
    # vertex_i = np.array([door_coorx, door_coory, door_coorz])

    # door_coorx = xs[x, y-1]
    # door_coory = ys[x, y-1]
    # door_coorz = zs[x, y-1]
    # vertex_j = np.array([door_coorx, door_coory, door_coorz])

    # door_coorx = xs[x+1, y]
    # door_coory = ys[x+1, y]
    # door_coorz = zs[x+1, y]
    # vertex_k = np.array([door_coorx, door_coory, door_coorz])

    # door_coorx, door_coory, door_coorz = vertex_i

    v1 = xyz[0,1] - xyz[0,0]
    v2 = xyz[1,0] - xyz[0,0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    return xyz[0,0], normal

def transition(xyz1, xyz2, points2):
    trans_matrix = xyz1- xyz2
    points2[:,:,:3] = points2[:,:,:3] + trans_matrix
    xyz2 = xyz2 + trans_matrix
    return points2, xyz2

def rotate(vector1, vector2, points2):
    # Define the vectors
    vector1 = -vector1
    # Calculate the cross product between the two vectors
    cross_product = np.cross(vector1, vector2)

    # Calculate the dot product between the two vectors
    dot_product = np.dot(vector1, vector2)

    # Calculate the norm of the cross product
    cross_norm = np.linalg.norm(cross_product)

    # Calculate the rotation angle
    angle = np.arctan2(cross_norm, dot_product)

    # Calculate the rotation axis
    axis = cross_product / cross_norm

    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(angle) + axis[0]**2*(1-np.cos(angle)), 
                                axis[0]*axis[1]*(1-np.cos(angle)) - axis[2]*np.sin(angle), 
                                axis[0]*axis[2]*(1-np.cos(angle)) + axis[1]*np.sin(angle)],
                                [axis[1]*axis[0]*(1-np.cos(angle)) + axis[2]*np.sin(angle), 
                                np.cos(angle) + axis[1]**2*(1-np.cos(angle)), 
                                axis[1]*axis[2]*(1-np.cos(angle)) - axis[0]*np.sin(angle)],
                                [axis[2]*axis[0]*(1-np.cos(angle)) - axis[1]*np.sin(angle), 
                                axis[2]*axis[1]*(1-np.cos(angle)) + axis[0]*np.sin(angle), 
                                np.cos(angle) + axis[2]**2*(1-np.cos(angle))]])

    # Apply the rotation matrix to vector2
    new_vector2 = np.dot(rotation_matrix, vector2)

    print("Original vector1:", vector1)
    print("Original vector2:", vector2)
    print("Rotated vector2:", new_vector2)

    # Rotate points2 to be the same orientation as points1
    points2[:,:,:3] = np.dot(points2[:,:,:3], rotation_matrix)

    return points2, new_vector2

def postprocessing_cor_id(cor_id):
    # var = y_bon_[0][:-1] - y_bon_[0][1:]
    # plt.plot(range(len(var)), var)
    # plt.show()

    # empty_rgb = np.zeros_like(rgb, np.float32)
    
    # x_coord = np.arange(y_bon_.shape[1])
    # y_coord_ceil = y_bon_[0].astype(int)
    # y_coord_floor = y_bon_[1].astype(int)
    # empty_rgb[y_coord_ceil, x_coord] = [255,255,255]
    # empty_rgb[y_coord_floor, x_coord] = [255,255,255]
    
    # img = empty_rgb
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    
    # # @@ method 1 -> fail @@@
    # # dst = cv2.cornerHarris(gray,5,3,0.04)
    # # ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    # # dst = np.uint8(dst)
    # # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # # corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # # for i in range(1, len(corners)):
    # #     print(corners[i])
    # # img[dst>0.1*dst.max()]=[0,255,255]
    
    # # @@ method 2 -> fail @@@
    # # detect corners with the goodFeaturesToTrack function.
    # corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10)
    # corners = np.int0(corners)
    # # we iterate through each corner, 
    # # making a circle at each point that we think is a corner.
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(img, (x, y), 3, 255, -1)
    
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows    
    return cor_id

def save_txt(cor_id, y_cor_, y_bon_, args, fname):
    # save .txt
    with open(os.path.join(args.out, f'{fname}.layout.txt'), 'w') as f:
        for u, v in cor_id:
            f.write(f'{u:.1f} {v:.1f}\n')
    with open(os.path.join(args.out, f'{fname}.y_bon_.txt'), 'w') as f:
        for i in range(len(y_bon_[0])):
            f.write(f'{y_bon_[0, i]:.1f} {y_bon_[1, i]:.1f}\n')
    with open(os.path.join(args.out, f'{fname}.y_cor_.txt'), 'w') as f:
        for u in y_cor_:
            f.write(f'{u:.1f}\n')

def save_img(rgb, cor_id, y_bon_, y_cor_, args, fname):
    
    # save .png               
    plt.figure(figsize=(24,10))
    
    plt.subplot(221)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title('Raw image')
    # plt.savefig(os.path.join(args.out, f'{fname}.img.png'), bbox_inches='tight')
    
    plt.subplot(222)
    plt.imshow(np.concatenate([
    (y_cor_ * 255).reshape(1,-1,1).repeat(30, 0).repeat(3, 2).astype(np.uint8), rgb[30:]], axis=0))
    plt.plot(np.arange(y_bon_.shape[1]), y_bon_[0], 'r-')
    plt.plot(np.arange(y_bon_.shape[1]), y_bon_[1], 'r-')
    plt.scatter(cor_id[:, 0], cor_id[:, 1], marker='x', c='b')
    plt.axis('off')
    plt.title('y_bon_ (red) / y_cor_ (up-most bar) / cor_id (blue x)')
    # plt.savefig(os.path.join(args.out, f'{fname}.img.png'), bbox_inches='tight')

    plt.subplot(223)
    depth_img = layout_2_depth(cor_id, *rgb.shape[:2])
    plt.imshow(depth_img, cmap='inferno_r')
    plt.axis('off')
    plt.title('rendered depth from the estimated layout (cor_id)')
    # plt.savefig(os.path.join(args.out, f'{fname}.result.png'), bbox_inches='tight')
    
    plt.subplot(224)
    # empty_rgb = np.zeros_like(rgb, np.float32)
    # plt.imshow(np.concatenate([
    #     (y_cor_ * 255).reshape(1,-1,1).repeat(30, 0).repeat(3, 2).astype(np.uint8), empty_rgb[30:]], axis=0))
    plt.imshow(depth_img, cmap='inferno_r')
    plt.plot(np.arange(y_bon_.shape[1]), y_bon_[0], 'r-')
    plt.plot(np.arange(y_bon_.shape[1]), y_bon_[1], 'r-')
    plt.scatter(cor_id[:, 0], cor_id[:, 1], marker='x', c='b')
    plt.axis('off')
    plt.title('y_bon_ (red) / y_cor_ (up-most bar) / cor_id (blue x)')
    
    plt.savefig(os.path.join(args.out, f'{fname}.result.png'), bbox_inches='tight')
    plt.show()

def save_ply(points, faces, args, fname):
    # Dump results ply
    if args.out:
        ply_header = '\n'.join([
            'ply',
            'format ascii 1.0',
            f'element vertex {len(points):d}',
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            f'element face {len(faces):d}',
            'property list uchar int vertex_indices',
            'end_header',
        ])
        with open(os.path.join(args.out, f'{fname}.layout.ply'), 'w') as f:
            f.write(ply_header)
            f.write('\n')
            for x, y, z, r, g, b in points:
                f.write(f'{x:.2f} {y:.2f} {z:.2f} {r:.0f} {g:.0f} {b:.0f}\n')
            for i, j, k in faces:
                f.write(f'3 {i:d} {j:d} {k:d}\n')

def vis_3d(points, faces, cor_id):
    # if not args.no_vis:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points[:, :3])
    mesh.vertex_colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    draw_geometries = [mesh]

    H, W = points.shape[:2]
    # Show wireframe
    # if not args.ignore_wireframe:
    # Convert cor_id to 3d xyz
    N = len(cor_id) // 2
    floor_z = -1.6
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
    c = np.sqrt((floor_xy**2).sum(1))
    v = np_coory2v(cor_id[0::2, 1], H)
    ceil_z = (c * np.tan(v)).mean()

    # Prepare wireframe in open3d
    assert N == len(floor_xy)
    wf_points = [[x, y, floor_z] for x, y in floor_xy] +\
                [[x, y, ceil_z] for x, y in floor_xy]
    wf_lines = [[i, (i+1)%N] for i in range(N)] +\
            [[i+N, (i+1)%N+N] for i in range(N)] +\
            [[i, i+N] for i in range(N)]
    wf_colors = [[1, 0, 0] for i in range(len(wf_lines))]
    wf_line_set = o3d.geometry.LineSet()
    wf_line_set.points = o3d.utility.Vector3dVector(wf_points)
    wf_line_set.lines = o3d.utility.Vector2iVector(wf_lines)
    wf_line_set.colors = o3d.utility.Vector3dVector(wf_colors)
    draw_geometries.append(wf_line_set)

    o3d.visualization.draw_geometries(draw_geometries, mesh_show_back_face=True) 
       
def infer(rgb):
    # Run inference
    with torch.no_grad():                
        x = torch.from_numpy(rgb).permute(2,0,1)[None].float() / 255.
        x = x.to(device)
        output = net.infer(x)
        cor_id = output['cor_id']
        y_bon_ = output['y_bon_']
        y_cor_ = output['y_cor_']        
                    
        # Reading source (texture img, cor_id txt)
        # equirect_texture = np.array(Image.open(path))
        equirect_texture = rgb
        H, W = equirect_texture.shape[:2]
        # if os.path.join(args.out, f'{fname}.layout.txt').endswith('json'):
        #     with open(os.path.join(args.out, f'{fname}.layout.txt')) as f:
        #         inferenced_result = json.load(f)
        #     cor_id = np.array(inferenced_result['uv'], np.float32)
        #     cor_id[:, 0] *= W
        #     cor_id[:, 1] *= H
        # else:
        #     cor_id = np.loadtxt(os.path.join(args.out, f'{fname}.layout.txt')).astype(np.float32)
        
        
        # @@@@ TODO @@@@@@@@@@@@@@@@@@@@@@@@ POST PROCESSING - cor_id @@@@@@@@@@@@@@@@@@@@@@@@
        cor_id = postprocessing_cor_id(cor_id)
        # @@@@ TODO @@@@@@@@@@@@@@@@@@@@@@@@ POST PROCESSING - cor_id @@@@@@@@@@@@@@@@@@@@@@@@


        depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, return_mask=True)
        # xs, ys, zs = cvt_cor2layout(cor_id, H, W, depth)
        coorx, coory = np.meshgrid(np.arange(W), np.arange(H))           
        
        us = - np_coorx2u(coorx, W)
        vs = np_coory2v(coory, H)
        zs = depth * np.sin(vs)
        cs = depth * np.cos(vs)
        xs = cs * np.sin(us)
        ys = -cs * np.cos(us)            
        
        # Aggregate mask
        mask = np.ones_like(floor_mask)
        # if args.ignore_floor:
        #     mask &= ~floor_mask
        # if not args.show_ceiling:
        #     mask &= ~ceil_mask
        # if args.ignore_wall:
        #     mask &= ~wall_mask
        mask &= ~ceil_mask

        # Prepare ply's points and faces
        xyzrgb = np.concatenate([
            xs[...,None], ys[...,None], zs[...,None],
            equirect_texture], -1)
        xyzrgb = np.concatenate([xyzrgb, xyzrgb[:,[0]]], 1)            
        
        # # @@@@@@@@@@@ xyz_door @@@@@@@@@@@
        # xyzrgb[door_Y, door_X, 3:6] = door_color
        # # @@@@@@@@@@@ xyz_door @@@@@@@@@@@
        
        mask = np.concatenate([mask, mask[:,[0]]], 1)
        lo_tri_template = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 1]])
        up_tri_template = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 1]])
        ma_tri_template = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]])
        lo_mask = (correlate2d(mask, lo_tri_template, mode='same') == 3)
        up_mask = (correlate2d(mask, up_tri_template, mode='same') == 3)
        ma_mask = (correlate2d(mask, ma_tri_template, mode='same') == 3) & (~lo_mask) & (~up_mask)
        ref_mask = (
            lo_mask | (correlate2d(lo_mask, np.flip(lo_tri_template, (0,1)), mode='same') > 0) |\
            up_mask | (correlate2d(up_mask, np.flip(up_tri_template, (0,1)), mode='same') > 0) |\
            ma_mask | (correlate2d(ma_mask, np.flip(ma_tri_template, (0,1)), mode='same') > 0)
        )
        points = xyzrgb[ref_mask]
        

        ref_id = np.full(ref_mask.shape, -1, np.int32)
        ref_id[ref_mask] = np.arange(ref_mask.sum())
        faces_lo_tri = np.stack([
            ref_id[lo_mask],
            ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
            ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
        ], 1)
        faces_up_tri = np.stack([
            ref_id[up_mask],
            ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
            ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
        ], 1)
        faces_ma_tri = np.stack([
            ref_id[ma_mask],
            ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
            ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
        ], 1)
        faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])      

    return points, xyzrgb, faces, cor_id, y_cor_, y_bon_, us, vs, xs, ys, zs
  
if __name__ == '__main__':
    # Parse args & config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', default="config/mp3d_layout/HOHO_layout_aug_efficienthc_Transen1_resnet34.yaml")
    parser.add_argument('--pth', default="ckpt/mp3d_layout_HOHO_layout_aug_efficienthc_Transen1_resnet34/ep300.pth")
    parser.add_argument('--out', default="output/")
    parser.add_argument('--inp', default="assets/006/")
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    device = 'cpu'
    print(torch.cuda.is_available())
    
    # Parse input paths
    rgb_lst = glob.glob(args.inp + "/*")
    if len(rgb_lst) == 0:
        print('No images found')
        import sys; sys.exit()

    # Init model
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs)
    # net.load_state_dict(torch.load(args.pth, map_location=device))map_location=torch.device('cpu')
    net.load_state_dict(torch.load(args.pth, map_location=torch.device('cpu')))
    net = net.eval().to(device)
    
    points_list = []
    door_xyz_list = []
    door_normal_list = []

    # for문 밖에서 door_list 선언
    door_list = [{'id': 0, 'room': 0, 'door_point': np.array([290, 339]), 'direction': 'L'},
                 {'id': 1, 'room': 0, 'door_point': np.array([290, 526]), 'direction': 'F'},
                 {'id': 2, 'room': 1, 'door_point': np.array([295, 960]), 'direction': 'B'},
                 {'id': 3, 'room': 2, 'door_point': np.array([295, 727]), 'direction': 'R'}]
    idx = 0
    for path in tqdm(rgb_lst):
        fname = os.path.splitext(os.path.split(path)[1])[0]
        print(f"\n({idx}th image) {fname} is loaded......")
        rgb = cv2.imread(path)
        rgb = cv2.resize(rgb, (1024, 512), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
               
        # @@@ TODO @@@ 1. Door segmentation @@@@@
        
        # door_left_up = (490, 247)
        # door_right_up =(522, 247)       
        # door_left_dw = (490, 332)    
        # door_right_dw =  (522, 332)           
        # cv2.line(rgb, door_left_up, door_left_dw, color=[0,255,255])
        # cv2.line(rgb, door_left_up, door_right_up, color=[0,255,255])
        # cv2.line(rgb, door_right_dw, door_left_dw, color=[0,255,255])
        # cv2.line(rgb, door_right_dw, door_right_up, color=[0,255,255])
        
        # door_x = np.arange(510, 540, 5)
        # door_y = np.arange(247, 332, 5)
        # door_X, door_Y = np.meshgrid(door_x, door_y)
        # door_color = [255, 255, 0]
        # rgb[door_Y, door_X] = door_color
        
        # @@@@@@ 2. 3D room layout estimation @@@@@@
        points, xyzrgb, faces, cor_id, y_cor_, y_bon_, us, vs, xs, ys, zs = infer(rgb)

        print(xyzrgb[256, 512, :3])

        # save points as .ply
        save_ply(points, faces, args, fname)
        # save txt
        save_txt(cor_id, y_cor_, y_bon_, args, fname)
        # save_image and plt.show()
        save_img(rgb, cor_id, y_bon_, y_cor_, args, fname)
        # # visualize
        # vis_3d(points, faces, cor_id)
        
        # @@@@@@ 3. Door Loacalization
        # points1 = get_vertices(cor_id, xs, ys, zs)
        door_xyz, door_normal = Getdoor(door_list[idx,0], door_list[idx,1], xyzrgb[:,:,:3])
        
        door_normal_list.append(door_normal)
        points_list.append(xyzrgb)
        door_xyz_list.append(door_xyz)

        idx += 1 
    
    
    # @@@@@@@@@@@@@@@@@@@@@@ Here @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    src, ref = 2, 0
    points_list[src], door_xyz_list[src] = transition(door_xyz_list[ref], door_xyz_list[src], points_list[src])
    points_list[src], door_normal_list[src] = rotate(door_normal_list[ref], door_normal_list[src],  points_list[src])
       
    src, ref = 1, 0
    points_list[src], door_xyz_list[src] = transition(door_xyz_list[ref], door_xyz_list[src], points_list[2])
    points_list[src], door_normal_list[src] = rotate(door_normal_list[ref], door_normal_list[2],  points_list[2])            
        
    total =  [points_list[i] for i in range(len(points_list))]
    regit_xyzrgb = np.concatenate(total, axis=1) 
    # @@@@@@@@@@@@@@@@@@@@@@ Here @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   
    
    
    pcd = o3d.geometry.PointCloud()
    np_points = regit_xyzrgb[:,:,:3].reshape(-1, 3)
    np_colors = regit_xyzrgb[:,:,3:].reshape(-1, 3)/255
    
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)  
    o3d.io.write_point_cloud("./total_xyzrgb.ply", pcd)
    # pcd_load = o3d.io.read_point_cloud("../../TestData/sync.ply")
    
    
    
    o3d.visualization.draw_geometries([pcd])
    # @@@ TODO @@@ 3. camera localization @@@@@

        
