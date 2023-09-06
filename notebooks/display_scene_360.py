import os
os.chdir('../')
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from BlendedNeRF.run_BlendedNeRF import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()


def init_scene(config_file_path):
    parser = config_parser()
    args = parser.parse_args("--config " + config_file_path)
    hwf, K, poses, images, i_train, i_val, i_test, near, far, render_poses = load_data(args)

    hwf = [args.sample_scale, args.sample_scale, hwf[2]]

    bds_dict = {
        'near': near,
        'far': far,
    }

    # scale focal length according to sample scale size
    f_sample_scale = hwf[2] * (args.sample_scale / hwf[1])
    hwf_sample_scale = [args.sample_scale, args.sample_scale, f_sample_scale]

    K = np.array([
        [hwf[2], 0, 0.5 * hwf[1]],
        [0, hwf[2], 0.5 * hwf[0]],
        [0, 0, 1]
    ])

    _, render_kwargs_test, _, _, _ = create_nerf(args)

    render_kwargs_test.update(bds_dict)

    if torch.cuda.is_available():
        render_kwargs_test['box_points'] = torch.load(args.box_points_path)
    else:
        render_kwargs_test['box_points'] = torch.load(args.box_points_path, map_location=torch.device('cpu'))
    # init scene origin
    render_kwargs_train = {'hwf': hwf_sample_scale, 'box_points': render_kwargs_test['box_points']}
    ema_scene_origin_world, ema_scene_origin_ndc = init_scene_origin(args, render_kwargs_train)

    render_kwargs_test['scene_origin'] = ema_scene_origin_world.value

    return args, render_kwargs_test, K, hwf


def render_single_pose(c2w,hwf,K,chunk,render_kwargs,render_factor):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
    return rgb, disp, extras['rgb_map_in_box'],extras


def get_pixels(points,pose,K, depth_map):
    if torch.cuda.is_available():
        pixels = torch.Tensor(points.shape[0],2).to(torch.int32)
        c2w = torch.Tensor(pose[:3, :3])
        ct = torch.Tensor(pose[:3, -1])
    else:
        pixels = torch.Tensor(points.shape[0],2).to(torch.int32).cpu()
        c2w = torch.Tensor(pose[:3, :3]).cpu()
        ct = torch.Tensor(pose[:3, -1]).cpu()

    for i,point in enumerate(points):
        point_camera = torch.linalg.inv(c2w).matmul((point - ct))
        z_val = torch.clone(-point_camera[2])
        point_camera /= z_val
        point_pixel_x = (K[0][0] * point_camera[0] + K[0][2])
        point_pixel_y = (K[1][1] * (-point_camera[1]) + K[1][2])
        pixels[i, 0] = point_pixel_x.to(torch.int32)
        pixels[i, 1] = point_pixel_y.to(torch.int32)

    filtered_pixels = []
    dists = torch.linalg.norm(points.cpu()-pose[:3,-1],dim=-1)
    for i, p in enumerate(pixels):
        W = int(K[0][2]*2)
        p = torch.clip(p,0,W-1)
        if depth_map[p[0],p[1]] == 0 or dists[i] < depth_map[p[0],p[1]]:
            filtered_pixels.append(p)
    if len(filtered_pixels) > 0:
        pixels = torch.stack(filtered_pixels)
        return pixels
    else:
        return None


def compute_box_points(x_range,y_range,z_range,out_file_name = None):
    steps = 2000
    # find coordinates of 3d box
    xs = torch.linspace(0,0.2,steps=steps)
    ys = torch.zeros_like(xs)
    zs = torch.zeros_like(xs)
    x_axis = torch.stack([xs,ys,zs],dim=-1)

    ys = torch.linspace(0,0.2,steps=steps)
    xs = torch.zeros_like(ys)
    zs = torch.zeros_like(ys)
    y_axis = torch.stack([xs,ys,zs],dim=-1)

    zs = torch.linspace(0,0.2,steps=steps)
    xs = torch.zeros_like(zs)
    ys = torch.zeros_like(zs)
    z_axis = torch.stack([xs,ys,zs],dim=-1)
    ##################################################################################
    #######################box lines parallel to x axis###############################
    low_x, high_x = x_range
    low_y, high_y = y_range
    low_z, high_z = z_range
    xs = torch.linspace(low_x,high_x,steps=steps)
    ys = torch.full(xs.size(),high_y)
    zs = torch.full(xs.size(),low_z)
    parallel_x_back_low_line = torch.stack([xs,ys,zs],dim=-1)

    xs = torch.linspace(low_x,high_x,steps=steps)
    ys = torch.full(xs.size(),high_y)
    zs = torch.full(xs.size(),high_z)
    parallel_x_back_high_line = torch.stack([xs,ys,zs],dim=-1)

    xs = torch.linspace(low_x,high_x,steps=steps)
    ys = torch.full(xs.size(),low_y)
    zs = torch.full(xs.size(),low_z)
    parallel_x_front_low_line = torch.stack([xs,ys,zs],dim=-1)

    xs = torch.linspace(low_x,high_x,steps=steps)
    ys = torch.full(xs.size(),low_y)
    zs = torch.full(xs.size(),high_z)
    parallel_x_front_high_line = torch.stack([xs,ys,zs],dim=-1)
    #######################box lines parallel to y axis###############################
    ys = torch.linspace(low_y,high_y,steps=steps)
    xs = torch.full(ys.size(),high_x)
    zs = torch.full(ys.size(),low_z)
    parallel_y_back_low_line = torch.stack([xs,ys,zs],dim=-1)

    ys = torch.linspace(low_y,high_y,steps=steps)
    xs = torch.full(ys.size(),high_x)
    zs = torch.full(ys.size(),high_z)
    parallel_y_back_high_line = torch.stack([xs,ys,zs],dim=-1)

    ys = torch.linspace(low_y,high_y,steps=steps)
    xs = torch.full(ys.size(),low_x)
    zs = torch.full(ys.size(),low_z)
    parallel_y_front_low_line = torch.stack([xs,ys,zs],dim=-1)

    ys = torch.linspace(low_y,high_y,steps=steps)
    xs = torch.full(ys.size(),low_x)
    zs = torch.full(ys.size(),high_z)
    parallel_y_front_high_line = torch.stack([xs,ys,zs],dim=-1)
    #######################box lines parallel to z axis###############################
    zs = torch.linspace(low_z,high_z,steps=steps)
    xs = torch.full(xs.size(),low_x)
    ys = torch.full(xs.size(),high_y)
    parallel_z_back_left_line = torch.stack([xs,ys,zs],dim=-1)

    zs = torch.linspace(low_z,high_z,steps=steps)
    xs = torch.full(xs.size(),high_x)
    ys = torch.full(xs.size(),high_y)
    parallel_z_back_right_line = torch.stack([xs,ys,zs],dim=-1)

    zs = torch.linspace(low_z,high_z,steps=steps)
    xs = torch.full(xs.size(),low_x)
    ys = torch.full(xs.size(),low_y)
    parallel_z_front_left_line = torch.stack([xs,ys,zs],dim=-1)

    zs = torch.linspace(low_z,high_z,steps=steps)
    xs = torch.full(xs.size(),high_x)
    ys = torch.full(xs.size(),low_y)
    parallel_z_front_right_line = torch.stack([xs,ys,zs],dim=-1)

    box = torch.cat([parallel_x_back_low_line,parallel_x_back_high_line,parallel_x_front_low_line,parallel_x_front_high_line,
                     parallel_y_back_low_line,parallel_y_back_high_line,parallel_y_front_low_line,parallel_y_front_high_line,
                     parallel_z_back_left_line, parallel_z_back_right_line,parallel_z_front_left_line,parallel_z_front_right_line],dim=0)

    p_back_low_left = torch.tensor([low_x,high_y,low_z])
    p_back_low_right = torch.tensor([high_x,high_y,low_z])
    p_back_high_left = torch.tensor([low_x,high_y,high_z])
    p_back_high_right = torch.tensor([high_x,high_y,high_z])
    p_front_low_left = torch.tensor([low_x,low_y,low_z])
    p_front_low_right = torch.tensor([high_x,low_y,low_z])
    p_front_high_left = torch.tensor([low_x,low_y,high_z])
    p_front_high_right = torch.tensor([high_x,low_y,high_z])
    box_points = torch.stack([p_front_high_left,p_front_high_right,p_front_low_right,p_front_low_left,p_back_high_left,p_back_high_right,p_back_low_right,p_back_low_left,],dim=0)
    if out_file_name != None:
        torch.save(box_points,f'{out_file_name}.pt')
    return x_axis,y_axis,z_axis,box, box_points


def plot_box(frames,frames_poses, box_points, K, x_axis ,y_axis, z_axis, depth_map_full_frame):
    for pose, frame in zip(frames_poses, frames):
        x_axis_pixels = get_pixels(x_axis, pose, K,depth_map_full_frame)
        y_axis_pixels = get_pixels(y_axis, pose, K,depth_map_full_frame)
        z_axis_pixels = get_pixels(z_axis, pose, K,depth_map_full_frame)
        box_pixels = get_pixels(box_points,pose,K,depth_map_full_frame)

        if x_axis_pixels is not None:
            x_axis_pixels = x_axis_pixels.cpu().detach().numpy()
            frame[x_axis_pixels[:,1],x_axis_pixels[:,0]] = np.array([1, 0, 0])
        if y_axis_pixels is not None:
            y_axis_pixels = y_axis_pixels.cpu().detach().numpy()
            frame[y_axis_pixels[:,1],y_axis_pixels[:,0]] = np.array([0, 1, 0])
        if z_axis_pixels is not None:
            z_axis_pixels = z_axis_pixels.cpu().detach().numpy()
            frame[z_axis_pixels[:,1],z_axis_pixels[:,0]] = np.array([0, 0, 1])
        if box_pixels is not None:
            box_pixels = box_pixels.cpu().detach().numpy()
            W = int(K[0][2] * 2)
            H = int(K[1][2] * 2)
            box_pixels    = np.clip(box_pixels,0,np.min([W-1,H-1]))
            frame[box_pixels[:,1],box_pixels[:,0]] = np.array([0.6, 0, 0.3])
        print("Full image:")
        figure(num=None,figsize=(30,30),dpi=30,facecolor='w',edgecolor='k')
        plt.imshow(frame)
        plt.show()


def plot_scene(args, render_kwargs_test,hwf,K, theta,phi,radius, create_box=False, box_center=None, box_edges_size=None, box_path=None):
    with torch.no_grad():
        pose = pose_spherical(theta=theta, phi=phi, radius=radius)

        render_kwargs_test['render_in_box'] = True
        render_kwargs_test['update_bounds'] = False
        render_kwargs_test['render_full_frame'] = True
        render_kwargs_test['blend'] = args.blend
        render_kwargs_test['use_dist_blend'] = args.use_dist_blend
        render_kwargs_test['dist_blend_alpha'] = args.dist_blend_alpha
        render_kwargs_test['is360Scene'] = args.is360Scene

        if create_box:
            x_range = (-box_edges_size[0]/2 + box_center[0], box_center[0] + box_edges_size[0]/2)
            y_range = (-box_edges_size[1]/2 + box_center[1], box_center[1] + box_edges_size[1]/2)
            z_range = (-box_edges_size[2]/2 + box_center[2], box_center[2] + box_edges_size[2]/2)
        else:
            box_points = render_kwargs_test['box_points']
            x_range = (box_points[:, 0].min().item(), box_points[:, 0].max().item())
            y_range = (box_points[:, 1].min().item(), box_points[:, 1].max().item())
            z_range = (box_points[:, 2].min().item(), box_points[:, 2].max().item())

        x_axis, y_axis, z_axis, cube_range, box_points = compute_box_points(x_range, y_range, z_range)
        if create_box:
            render_kwargs_test['box_points'] = box_points

        x_center = (box_points[:, 0].min() + box_points[:, 0].max()) / 2
        y_center = (box_points[:, 1].min() + box_points[:, 1].max()) / 2
        z_center = (box_points[:, 2].min() + box_points[:, 2].max()) / 2
        box_points_center = torch.stack([x_center, y_center, z_center])

        # compute radius
        box_height = torch.linalg.norm(box_points[0] - box_points[1]).cpu()
        box_width = torch.linalg.norm(box_points[0] - box_points[4]).cpu()
        box_depth = torch.linalg.norm(box_points[0] - box_points[3]).cpu()
        afov = 2 * np.arctan(args.sample_scale / (2 * hwf[2]))
        max_edge = np.max([box_height, box_width, box_depth])
        radius = (max_edge) / (2 * np.tan(afov / 2))

        # sample pose
        pose[:3, -1] = pose[:3, -1] + box_points_center

        rgb, disp, rgb_map_in_box, extras = render_single_pose(pose, hwf, K, args.chunk, render_kwargs_test, 0)

        plot_box(rgb[None, :].detach().cpu().numpy(), pose[None, :].detach().cpu().numpy(), cube_range, K, x_axis, y_axis, z_axis,
                 extras['depth_map_full_frame'])

        print(f"In box image:")
        figure(num=None, figsize=(30, 30), dpi=30, facecolor='w', edgecolor='k')
        plt.imshow(extras["rgb_map_in_box"].cpu().detach().numpy())
        plt.show()

    if create_box:
        torch.save(box_points, box_path)