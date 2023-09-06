import os
os.chdir('../')
import matplotlib.pyplot as plt
import numpy as np
import torch
from BlendedNeRF.run_BlendedNeRF import *
from loaders.load_llff import viewmatrix, render_path_spiral
from matplotlib.pyplot import figure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()


def init_scene(config_file_path):
    parser = config_parser()
    args = parser.parse_args("--config " + config_file_path)
    hwf, K, poses, images, i_train, i_val, i_test, near, far, render_poses = load_data(args)
    W, H = (int(args.sample_scale), int(args.sample_scale))
    f_sample_scale = hwf[2] * (args.sample_scale / hwf[1])
    hwf = [H, W, f_sample_scale]

    if K is None:
        K = np.array([
            [f_sample_scale, 0, 0.5 * W],
            [0, f_sample_scale, 0.5 * H],
            [0, 0, 1]
        ])

    # Create nerf model
    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)

    bds_dict = {
        'near': near,
        'far': far,
    }

    if torch.cuda.is_available():
        render_kwargs_test['box_points'] = torch.load(args.box_points_path)
    else:
        render_kwargs_test['box_points'] = torch.load(args.box_points_path, map_location=torch.device('cpu'))

    render_kwargs_test.update(bds_dict)

    return args, render_kwargs_test, hwf, K


def render_single_pose(c2w, hwf, K, chunk, render_kwargs, render_factor):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor
    rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
    return rgb, disp, acc, extras


def get_pixels(points, pose, hwf, K, depth_map):
    if torch.cuda.is_available():
        pixels = torch.Tensor(points.shape[0], 2).to(torch.int32)
        c2w = torch.Tensor(pose[:3, :3])
        ct = torch.Tensor(pose[:3, -1])
    else:
        pixels = torch.Tensor(points.shape[0], 2).to(torch.int32).cpu()
        c2w = torch.Tensor(pose[:3, :3]).cpu()
        ct = torch.Tensor(pose[:3, -1]).cpu()

    for i, point in enumerate(points):
        point_camera = torch.linalg.inv(c2w).matmul((point - ct))
        z_val = torch.clone(-point_camera[2])
        point_camera /= z_val
        point_pixel_x = (K[0][0] * point_camera[0] + K[0][2])
        point_pixel_y = (K[1][1] * (-point_camera[1]) + K[1][2])
        pixels[i, 0] = point_pixel_x.to(torch.int32)
        pixels[i, 1] = point_pixel_y.to(torch.int32)
    filtered_pixels = []
    # convert to ndc
    dists = torch.linalg.norm(points.cpu() - pose[:3, -1], dim=-1)
    for i, p in enumerate(pixels):
        if (p[0] > (hwf[1] - 1) or p[0] < 0) or (p[1] > (hwf[0] - 1) or p[1] < 0):
            continue
        if depth_map[p[0], p[1]] == 0 or dists[i] < depth_map[p[0], p[1]]:
            filtered_pixels.append(p)
        filtered_pixels.append(p)
    if len(filtered_pixels) > 0:
        pixels = torch.stack(filtered_pixels)
        return pixels
    else:
        return None


def compute_box_points(x_range, y_range, z_range, out_file_name=None):
    # find coordinates of 3d box
    xs = torch.linspace(0, 0.2, steps=1000)
    ys = torch.zeros_like(xs)
    zs = torch.zeros_like(xs)
    x_axis = torch.stack([xs, ys, zs], dim=-1)

    ys = torch.linspace(0, 0.2, steps=1000)
    xs = torch.zeros_like(ys)
    zs = torch.zeros_like(ys)
    y_axis = torch.stack([xs, ys, zs], dim=-1)

    zs = torch.linspace(0, 0.2, steps=1000)
    xs = torch.zeros_like(zs)
    ys = torch.zeros_like(zs)
    z_axis = torch.stack([xs, ys, zs], dim=-1)
    ##################################################################################
    #######################box lines parallel to x axis###############################
    low_x, high_x = x_range
    low_y, high_y = y_range
    low_z, high_z = z_range
    xs = torch.linspace(low_x, high_x, steps=1000)
    ys = torch.full(xs.size(), high_y)
    zs = torch.full(xs.size(), low_z)
    parallel_x_back_low_line = torch.stack([xs, ys, zs], dim=-1)

    xs = torch.linspace(low_x, high_x, steps=1000)
    ys = torch.full(xs.size(), high_y)
    zs = torch.full(xs.size(), high_z)
    parallel_x_back_high_line = torch.stack([xs, ys, zs], dim=-1)

    xs = torch.linspace(low_x, high_x, steps=1000)
    ys = torch.full(xs.size(), low_y)
    zs = torch.full(xs.size(), low_z)
    parallel_x_front_low_line = torch.stack([xs, ys, zs], dim=-1)

    xs = torch.linspace(low_x, high_x, steps=1000)
    ys = torch.full(xs.size(), low_y)
    zs = torch.full(xs.size(), high_z)
    parallel_x_front_high_line = torch.stack([xs, ys, zs], dim=-1)
    #######################box lines parallel to y axis###############################
    ys = torch.linspace(low_y, high_y, steps=1000)
    xs = torch.full(ys.size(), high_x)
    zs = torch.full(ys.size(), low_z)
    parallel_y_back_low_line = torch.stack([xs, ys, zs], dim=-1)

    ys = torch.linspace(low_y, high_y, steps=1000)
    xs = torch.full(ys.size(), high_x)
    zs = torch.full(ys.size(), high_z)
    parallel_y_back_high_line = torch.stack([xs, ys, zs], dim=-1)

    ys = torch.linspace(low_y, high_y, steps=1000)
    xs = torch.full(ys.size(), low_x)
    zs = torch.full(ys.size(), low_z)
    parallel_y_front_low_line = torch.stack([xs, ys, zs], dim=-1)

    ys = torch.linspace(low_y, high_y, steps=1000)
    xs = torch.full(ys.size(), low_x)
    zs = torch.full(ys.size(), high_z)
    parallel_y_front_high_line = torch.stack([xs, ys, zs], dim=-1)
    #######################box lines parallel to z axis###############################
    zs = torch.linspace(low_z, high_z, steps=1000)
    xs = torch.full(xs.size(), low_x)
    ys = torch.full(xs.size(), high_y)
    parallel_z_back_left_line = torch.stack([xs, ys, zs], dim=-1)

    zs = torch.linspace(low_z, high_z, steps=1000)
    xs = torch.full(xs.size(), high_x)
    ys = torch.full(xs.size(), high_y)
    parallel_z_back_right_line = torch.stack([xs, ys, zs], dim=-1)

    zs = torch.linspace(low_z, high_z, steps=1000)
    xs = torch.full(xs.size(), low_x)
    ys = torch.full(xs.size(), low_y)
    parallel_z_front_left_line = torch.stack([xs, ys, zs], dim=-1)

    zs = torch.linspace(low_z, high_z, steps=1000)
    xs = torch.full(xs.size(), high_x)
    ys = torch.full(xs.size(), low_y)
    parallel_z_front_right_line = torch.stack([xs, ys, zs], dim=-1)

    box = torch.cat([parallel_x_back_low_line, parallel_x_back_high_line, parallel_x_front_low_line, parallel_x_front_high_line,
                     parallel_y_back_low_line, parallel_y_back_high_line, parallel_y_front_low_line, parallel_y_front_high_line,
                     parallel_z_back_left_line, parallel_z_back_right_line, parallel_z_front_left_line, parallel_z_front_right_line], dim=0)

    p_back_low_left = torch.tensor([low_x, high_y, low_z])
    p_back_low_right = torch.tensor([high_x, high_y, low_z])
    p_back_high_left = torch.tensor([low_x, high_y, high_z])
    p_back_high_right = torch.tensor([high_x, high_y, high_z])
    p_front_low_left = torch.tensor([low_x, low_y, low_z])
    p_front_low_right = torch.tensor([high_x, low_y, low_z])
    p_front_high_left = torch.tensor([low_x, low_y, high_z])
    p_front_high_right = torch.tensor([high_x, low_y, high_z])
    box_points = torch.stack(
        [p_front_high_left, p_front_high_right, p_front_low_right, p_front_low_left, p_back_high_left, p_back_high_right, p_back_low_right, p_back_low_left, ],
        dim=0)
    if out_file_name != None:
        torch.save(box_points, f'{out_file_name}.pt')
    return x_axis, y_axis, z_axis, box, box_points


def plot_box(frames, frames_poses, box_points,hwf, K, x_axis, y_axis, z_axis, depth_map):
    for pose, frame in zip(frames_poses, frames):
        x_axis_pixels = get_pixels(x_axis, pose, hwf, K,depth_map).detach().cpu().numpy()
        y_axis_pixels = get_pixels(y_axis, pose, hwf, K,depth_map).detach().cpu().numpy()
        z_axis_pixels = get_pixels(z_axis, pose, hwf, K,depth_map).detach().cpu().numpy()
        box_pixels = get_pixels(box_points, pose, hwf, K, depth_map).detach().cpu().numpy()
        frame[x_axis_pixels[:, 1], x_axis_pixels[:, 0]] = np.array([1, 0, 0])
        frame[y_axis_pixels[:, 1], y_axis_pixels[:, 0]] = np.array([0, 1, 0])
        frame[z_axis_pixels[:, 1], z_axis_pixels[:, 0]] = np.array([0, 0, 1])
        frame[box_pixels[:, 1], box_pixels[:, 0]] = np.array([0.6, 0, 0.3])
        figure(num=None, figsize=(20, 20), dpi=20, facecolor='w', edgecolor='k')
        plt.imshow(frame)
        plt.show()


def ndc2world(points_ndc, hwf):
    points_world = points_ndc.clone()
    points_world[:, 2] = 2 / (points_ndc[:, 2] - 1)
    points_world[:, 0] = (-points_ndc[:, 0] * points_world[:, 2] * hwf[0]) / 2 / hwf[2]
    points_world[:, 1] = (-points_ndc[:, 1] * points_world[:, 2] * hwf[1]) / 2 / hwf[2]
    return points_world


def plot_scene(args, render_kwargs_test, hwf, K, zoom, create_box=False, box_center=None, box_edges_size=None,  box_path="box_points/my_box.pt",
               pose_idx=0, sample_random_poses=False):
    render_kwargs_test['render_in_box'] = True
    render_kwargs_test['is360Scene'] = False
    render_kwargs_test['render_full_frame'] = True
    render_kwargs_test['data_type'] = 'llff'
    render_kwargs_test['sample_scale'] = args.sample_scale
    render_kwargs_test['hwf'] = hwf
    render_kwargs_test['blend'] = args.blend
    render_kwargs_test['use_dist_blend'] = args.use_dist_blend
    render_kwargs_test['dist_blend_alpha'] = args.dist_blend_alpha

    if create_box:
        x_range = (-box_edges_size[0] / 2 + box_center[0], box_center[0] + box_edges_size[0] / 2)
        y_range = (-box_edges_size[1] / 2 + box_center[1], box_center[1] + box_edges_size[1] / 2)
        z_range = (-box_edges_size[2] / 2 + box_center[2], box_center[2] + box_edges_size[2] / 2)
        box_range_ndc = torch.tensor([[x_range[0], y_range[0], z_range[0]], [x_range[1], y_range[1], z_range[1]]])
    else:
        box_points = render_kwargs_test['box_points']
        box_range_ndc = torch.tensor([[box_points[:, 0].min(), box_points[:, 1].min(), box_points[:, 2].min()],
                                      [box_points[:, 0].max(), box_points[:, 1].max(), box_points[:, 2].max()]])
    # compute ranges
    x_range = (box_range_ndc[0, 0], box_range_ndc[1, 0])
    y_range = (box_range_ndc[0, 1], box_range_ndc[1, 1])
    z_range = (box_range_ndc[0, 2], box_range_ndc[1, 2])
    x_axis, y_axis, z_axis, box_range_ndc, box_points = compute_box_points(x_range, y_range, z_range)
    # transform ranges to world coordinates
    box_range_world = ndc2world(box_range_ndc, hwf)
    x_axis = ndc2world(x_axis, hwf)
    y_axis = ndc2world(y_axis, hwf)
    z_axis = ndc2world(z_axis, hwf)

    # compute box points in world coordinates
    x_center = (box_points[:, 0].min() + box_points[:, 0].max()) / 2
    y_center = (box_points[:, 1].min() + box_points[:, 1].max()) / 2
    z_center = (box_points[:, 2].min() + box_points[:, 2].max()) / 2
    box_center_ndc = torch.stack([x_center, y_center, z_center])
    box_points_world = ndc2world(box_points, hwf)
    x_center = (box_points_world[:, 0].min() + box_points_world[:, 0].max()) / 2
    y_center = (box_points_world[:, 1].min() + box_points_world[:, 1].max()) / 2
    z_center = (box_points_world[:, 2].min() + box_points_world[:, 2].max()) / 2
    box_center_world = torch.stack([x_center, y_center, z_center])

    if create_box:
        render_kwargs_test['box_points'] = box_points

    ema_scene_origin = EMA(box_center_world, decay=0.9995)
    ema_scene_origin_ndc = EMA(box_center_ndc, decay=0.9995)
    render_kwargs_test['scene_origin'] = ema_scene_origin.value
    render_kwargs_test['scene_origin_ndc'] = ema_scene_origin_ndc.value

    c2w = np.array([[1, 0, 0, 0, hwf[0]], [0, 1, 0, 0, hwf[1]], [0, 0, 1, 0, hwf[2]]])
    up = np.array([0, 1, 0])
    focal = 407.5658

    box_width = torch.norm(box_points_world[0] - box_points_world[1]).cpu()
    box_height = torch.norm(box_points_world[0] - box_points_world[4]).cpu()
    box_depth = torch.norm(box_points_world[0] - box_points_world[3]).cpu()

    zoom = torch.tensor(zoom)
    x_factor = args.rads_x_factor * (torch.exp(2 * zoom))
    y_factor = args.rads_y_factor * (torch.exp(2 * zoom))

    rads = torch.tensor([box_width * x_factor, box_height * y_factor, 0.1]).cpu().detach().numpy()

    zdelta = 0
    zrate = 0.5
    N_rots = 2
    N_views = 120
    z_shift = 0

    offset = np.array([0, 0.4 * box_height, z_shift])

    render_poses = render_path_spiral(c2w, up, rads, hwf[2], zdelta, zrate, N_rots, N_views, offset)
    render_poses = np.array(render_poses).astype(np.float64)

    if sample_random_poses:
        sampled_poses = render_poses[np.random.choice(120, 3), :3, :4]
    else:
        sampled_poses = render_poses[[pose_idx], :3, :4]
    with torch.no_grad():
        for pos in sampled_poses:
            pose = torch.tensor(pos).to(device)
            # center camera around scene origin
            pose[:3, -1] = pose[:3, -1] + box_center_world.to(device)
            afov = 2 * np.arctan(hwf[1] / (2 * hwf[2]))  # FOV angle
            max_edge = torch.max(torch.tensor([box_height, box_width, box_depth]))
            radius_in = (max_edge) / (2 * np.tan(afov / 2))
            pose[2, -1] = box_center_world[2] + radius_in + zoom

            rgb, disp, acc, extras = render_single_pose(pose, hwf, K, args.chunk, render_kwargs_test, 0)
            x_axis_pixels = get_pixels(x_axis, pose.detach().cpu().numpy(), hwf, K, extras['depth_map_full_frame'])
            frame = rgb.detach().cpu().numpy()
            if x_axis_pixels is not None:
                x_axis_pixels = x_axis_pixels.detach().cpu().numpy()
                x_axis_pixels = np.clip(x_axis_pixels, 0, hwf[1] - 1)
                frame[x_axis_pixels[:, 1], x_axis_pixels[:, 0]] = np.array([1, 0, 0])
            y_axis_pixels = get_pixels(y_axis, pose.detach().cpu().numpy(), hwf, K, extras['depth_map_full_frame'])
            if y_axis_pixels is not None:
                y_axis_pixels = y_axis_pixels.detach().cpu().numpy()
                y_axis_pixels = np.clip(y_axis_pixels, 0, hwf[1] - 1)
                frame[y_axis_pixels[:, 1], y_axis_pixels[:, 0]] = np.array([0, 1, 0])
            z_axis_pixels = get_pixels(z_axis, pose.detach().cpu().numpy(), hwf, K, extras['depth_map_full_frame'])
            if z_axis_pixels is not None:
                z_axis_pixels = z_axis_pixels.detach().cpu().numpy()
                z_axis_pixels = np.clip(z_axis_pixels, 0, hwf[1] - 1)
                frame[z_axis_pixels[:, 1], z_axis_pixels[:, 0]] = np.array([0, 0, 1])
            box_pixels = get_pixels(box_range_world, pose.detach().cpu().numpy(), hwf, K, extras['depth_map_full_frame'])
            if box_pixels is not None:
                box_pixels = box_pixels.detach().cpu().numpy()
                box_pixels = np.clip(box_pixels, 0, hwf[1] - 1)
                frame[box_pixels[:, 1], box_pixels[:, 0]] = np.array([0.6, 0, 0.3])

            print(f"Full image:")
            figure(num=None, figsize=(30, 30), dpi=30, facecolor='w', edgecolor='k')
            plt.imshow(frame)
            plt.show()

            print(f"In box rgb map:")
            figure(num=None, figsize=(30, 30), dpi=30, facecolor='w', edgecolor='k')
            plt.imshow(extras['rgb_map_in_box'].cpu().detach().numpy(), cmap='gray')
            plt.show()

    if create_box:
        torch.save(render_kwargs_test['box_points'], box_path)