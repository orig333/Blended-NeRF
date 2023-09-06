import os
import imageio
import time
from tqdm import tqdm, trange
from optimization.augmentations import sample_background
from BlendedNeRF.run_BlendedNeRF_helpers import *
from loaders.load_llff import load_llff_data, render_path_spiral
from loaders.load_blender import load_blender_data, pose_spherical
from loaders.load_deepvoxels import load_dv_data
from loaders.load_LINEMOD import load_LINEMOD_data
from optimization.losses import ModuleLosses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False


def ndc2world(points_ndc, hwf):
    """
    convert points from ndc to world system
    """
    points_world = points_ndc.clone()
    points_world[:, 2] = 2 / (points_ndc[:, 2] - 1)
    points_world[:, 0] = (-points_ndc[:, 0] * points_world[:, 2] * hwf[0]) / 2 / hwf[2]
    points_world[:, 1] = (-points_ndc[:, 1] * points_world[:, 2] * hwf[1]) / 2 / hwf[2]
    return points_world


# Origin tracking
class EMA:
    """Keep track of the EMA of a single array."""

    def __init__(self, value, decay):
        self.value = value.to(device)
        self.decay = decay

    def update(self, new_value):
        self.value = (self.value * self.decay + torch.tensor(new_value) * (1 - self.decay))


def render_single_pose(c2w, hwf, K, chunk, render_kwargs, render_factor, batch_rays=None):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = int(H // render_factor)
        W = int(W // render_factor)
        focal = focal / render_factor
    if batch_rays is None:
        rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
    else:
        rgb, disp, acc, extras = render(H, W, K, chunk=chunk, rays=batch_rays, **render_kwargs)
    if 'rgb_map_out_box' in extras.keys():
        rgb = extras['rgb_map_out_box']
    elif 'rgb_map_in_box' in extras.keys():
        rgb = extras['rgb_map_in_box']
    return rgb, disp


def sample_img(i_train, images, poses, hwf, K, render_kwargs, render_factor, module_losses, args, i):
    if args.sample_pose:
        caption = args.description
        if args.dataset_type == 'blender' or args.is360Scene:
            theta = (-180. - 180.) * torch.rand(1) + 180
            phi = (-90. - 15.) * torch.rand(1) + 15.
            if args.use_dir_caption:
                if phi < -90 * (3 / 4) + 10 * (1 / 4):
                    caption = args.description + ", top down view"
                elif 0 <= theta <= 90:
                    caption = args.description + ", front view"
                elif 90 < theta <= 180 or -90 <= theta < 0:
                    caption = args.description + ", side view"
                else:
                    caption = args.description + ", back view"

            cube_points = render_kwargs['box_points']
            box_width = torch.norm(cube_points[0] - cube_points[1])
            box_height = torch.norm(cube_points[0] - cube_points[4])
            box_depth = torch.norm(cube_points[0] - cube_points[3])

            # copmute radius from box according to FOV
            afov = 2 * np.arctan(hwf[0] / (2 * hwf[2]))  # FOV angle
            max_edge = torch.max(torch.tensor([box_height, box_width, box_depth]))
            radius_in = max_edge / (2 * np.tan(afov / 2))

            radius_factor = (args.zoom_low - args.zoom_high) * torch.rand(1) + args.zoom_high
            render_kwargs['radius_render_in_box'] = radius_in * radius_factor
            pose_in = pose_spherical(theta, phi, radius_in)
            # move pose according to center of scene
            pose_in[:3, -1] = pose_in[:3, -1] + render_kwargs['scene_origin']

            rays_o_in, rays_d_in = get_rays(int(args.sample_scale), int(args.sample_scale), K, torch.Tensor(pose_in))  # (H, W, 3), (H, W, 3)
            batch_rays_in = torch.stack([rays_o_in, rays_d_in], 0)

        elif args.dataset_type == 'llff' and not args.is360Scene:
            c2w = np.array([[1, 0, 0, 0, hwf[0]], [0, 1, 0, 0, hwf[1]], [0, 0, 1, 0, hwf[2]]])
            up = np.array([0, 1, 0])
            box_points_ndc = render_kwargs['box_points']

            # transfer box coordinates to world coordinates
            box_points_world = ndc2world(box_points_ndc, hwf)
            render_kwargs['box_points_world'] = box_points_world

            # sample pose
            box_width = torch.norm(box_points_world[0] - box_points_world[1]).cpu()
            box_height = torch.norm(box_points_world[0] - box_points_world[4]).cpu()
            box_depth = torch.norm(box_points_world[0] - box_points_world[3]).cpu()
            zoom = (args.zoom_low - args.zoom_high) * torch.rand(1) + args.zoom_high
            module_losses.zoom = zoom
            x_factor = args.rads_x_factor * (torch.exp(torch.tensor(2) * zoom))
            y_factor = args.rads_y_factor * (torch.exp(torch.tensor(2) * zoom))
            rads = torch.tensor([box_width * x_factor, box_height * y_factor, 0.1]).cpu().detach().numpy()
            zdelta = 0  # not used in function
            zrate = 0.5  # rate if angle
            N_rots = 2
            N_views = 120
            z_shift = 0
            offset = np.array([0, 0.4 * box_height, z_shift])
            render_poses = render_path_spiral(c2w, up, rads, hwf[2], zdelta, zrate, N_rots, N_views, offset)
            render_poses = np.array(render_poses).astype(np.float64)
            pose_idx = np.random.choice(120, 1)
            pose_in = render_poses[pose_idx][0, :3, :4]
            # update caption
            caption = args.description
            if args.use_dir_caption:
                side_range = list(range(0, 11)) + list(range(17, 41)) + list(range(51, 70)) + list(range(80, 101)) + list(range(115, 120))
                if pose_idx in side_range:
                    caption = args.description + ", side view"
                else:
                    caption = args.description + ", front view"

            # move pose according to center of scene
            box_range = torch.tensor([[box_points_world[:, 0].min(), box_points_world[:, 1].min(), box_points_world[:, 2].min()],
                                      [box_points_world[:, 0].max(), box_points_world[:, 1].max(), box_points_world[:, 2].max()]])
            x_rnd_val = ((box_range[0, 0] - box_range[1, 0]) * torch.rand(1) + box_range[1, 0]).cpu().detach().numpy()[0]
            y_rnd_val = ((box_range[0, 1] - box_range[1, 1]) * torch.rand(1) + box_range[1, 1]).cpu().detach().numpy()[0]
            z_rnd_val = ((box_range[0, 2] - box_range[1, 2]) * torch.rand(1) + box_range[1, 2]).cpu().detach().numpy()[0]
            rnd_pos_in_box = np.array([x_rnd_val, y_rnd_val, z_rnd_val])
            use_scene_origin = np.random.binomial(1, args.p_center_scene_origin, 1)
            if use_scene_origin:
                shift = render_kwargs['scene_origin'].cpu().numpy()
            else:
                shift = rnd_pos_in_box
            pose_in[:3, -1] = pose_in[:3, -1] + shift

            # compute radius from center of scene in z axis
            afov = 2 * np.arctan(hwf[0] / (2 * hwf[2]))  # FOV angle
            max_edge = torch.max(torch.tensor([box_height, box_width, box_depth]))
            radius_in = max_edge / (2 * np.tan(afov / 2))
            render_kwargs['radius_render_in_box'] = radius_in

            # move camera pose in z axis according to radius
            max_edge_factor = 0.0
            pose_in[2, -1] = shift[2] + radius_in + zoom
            # get rays
            rays_o_in, rays_d_in = get_rays(int(args.sample_scale), int(args.sample_scale), K, torch.Tensor(pose_in))  # (H, W, 3), (H, W, 3)
            batch_rays_in = torch.stack([rays_o_in, rays_d_in], 0)
        else:
            return None
        module_losses.update_in_radius(radius_in)
        module_losses.update_caption(caption)
    else:
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]
        return target, pose

    return batch_rays_in, radius_in


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # For forward facing scenes
        # changed H,W to sample_scale
        if kwargs['render_in_box']:
            sample_scale_shape = rays_o.shape[0]
            rays_o, rays_d = ndc_rays(sample_scale_shape, sample_scale_shape, kwargs['hwf'][2], 1., rays_o, rays_d)
        else:
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        if k != "weights_sum_in_box":
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    if kwargs['render_full_frame']:
        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]
    else:
        ret_dict = {k: all_ret[k] for k in all_ret}
        return ret_dict


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())
    fixed_model = NeRF(D=args.netdepth, W=args.netwidth,
                       input_ch=input_ch, output_ch=output_ch, skips=skips,
                       input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
        fixed_model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                                input_ch=input_ch, output_ch=output_ch, skips=skips,
                                input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) == 0 and os.path.exists(args.base_weights):
        ckpts = [args.base_weights]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[args.ckpt_num]
        print('Reloading from', ckpt_path)
        if torch.cuda.is_available():
            ckpt = torch.load(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

        start = ckpt['global_step']

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path_fixed = ckpts[0]
            if torch.cuda.is_available():
                ckpt_fixed = torch.load(ckpt_path_fixed)
            else:
                ckpt_fixed = torch.load(ckpt_path_fixed, map_location=torch.device('cpu'))

            # freeze fixed_model weights
            fixed_model.load_state_dict(ckpt_fixed['network_fn_state_dict'])
            fixed_model.train(False)
            fixed_model.requires_grad_(False)
            if fixed_model_fine is not None:
                fixed_model_fine.load_state_dict(ckpt_fixed['network_fine_state_dict'])
                fixed_model_fine.train(False)
                fixed_model_fine.requires_grad_(False)

    model = nn.DataParallel(model)
    model_fine = nn.DataParallel(model_fine)
    fixed_model = nn.DataParallel(fixed_model)
    fixed_model_fine = nn.DataParallel(fixed_model_fine)
    if args.change_color:
        required_grads_names = ['feature_linear.weight', 'feature_linear.bias', 'views_linears.0.weight',
                                'views_linears.0.bias', 'rgb_linear.weight', 'rgb_linear.bias']
        grad_vars = []
        for name, param in model.module.named_parameters():
            if name not in required_grads_names:
                param.requires_grad = False
            else:
                grad_vars.append(param)

        if model_fine is not None:
            for name, param in model_fine.module.named_parameters():
                if name not in required_grads_names:
                    param.requires_grad = False
                else:
                    grad_vars.append(param)

    # # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }
    render_kwargs_train.update({'network_fine_fixed': fixed_model_fine, 'network_fn_fixed': fixed_model})

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)  # 1 - exp(-sigma(i)* delta(i))

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1).to(device)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3] # c(r) =  sum(Ti(1 - exp(-sigma(i)* delta(i)))ci)

    depth_map = torch.sum(weights * z_vals, -1).detach()
    disp_map = (1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))).detach()
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights.detach(), depth_map


def raw2outputsBlend(raw_fixed, raw_train, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, sum_in_act=False):
    """Transforms model's predictions to semantically meaningful values(with blending).
    Args:
        raw_fixed: [num_rays, num_samples along ray, 4]. Prediction from model.
        raw_train: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha_sum_in_act = lambda raw_f, raw_t, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw_f + raw_t) * dists)  # 1 - exp(-sigma(i)* delta(i))
    raw2alpha_sum_out_act = lambda raw_f, raw_t, dists, act_fn=F.relu: 1. - torch.exp(-(act_fn(raw_f) + act_fn(raw_t)) * dists)  # 1 - exp(-sigma(i)* delta(i))
    oneRaw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)  # 1 - exp(-sigma(i)* delta(i))
    if sum_in_act:
        raw2alpha = raw2alpha_sum_in_act
    else:
        raw2alpha = raw2alpha_sum_out_act

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1).to(device)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    alpha_fixed = oneRaw2alpha(raw_fixed[..., 3], dists)
    alpha_train = oneRaw2alpha(raw_train[..., 3], dists)
    alpha_fixed = torch.stack([alpha_fixed, alpha_fixed, alpha_fixed], dim=-1)
    alpha_train = torch.stack([alpha_train, alpha_train, alpha_train], dim=-1)

    rgbs_sum = (raw_fixed[..., :3] * alpha_fixed + raw_train[..., :3] * alpha_train) / (0.000001 + alpha_fixed + alpha_train)
    rgb = torch.sigmoid(rgbs_sum)  # [N_rays, N_samples, 3]
    if not sum_in_act:
        transmittance_train = torch.cumprod(torch.cat([torch.ones((alpha_train[..., 0].shape[0], 1)), 1. - alpha_train[..., 0] + 1e-10], -1), -1)[:, :-1]
        weights_train = alpha_train[..., 0] * transmittance_train
        acc_map_train = torch.sum(weights_train, -1)
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_train[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(rgb[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw_fixed[..., 3], raw_train[..., 3] + noise, dists)  # [N_rays, N_samples]

    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3] # c(r) =  sum(Ti(1 - exp(-sigma(i)* delta(i)))ci)

    depth_map = torch.sum(weights * z_vals, -1).detach()
    disp_map = (1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))).detach()
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    if not sum_in_act:
        return rgb_map, disp_map, acc_map, weights.detach(), depth_map, acc_map_train
    else:
        return rgb_map, disp_map, acc_map, weights.detach(), depth_map


def render_full_frame_fn(pts, viewdirs, box_points, network_query_fn, run_fn, run_fn_fixed, z_vals, rays_d, raw_noise_std, white_bkgd, pytest, blend,
                         pt_is_in_box, scene_origin, scene_origin_ndc, dist_blend_alpha, sum_in_activation, data_type, is360Scene, render_fixed,
                         use_dist_blend):
    with torch.no_grad():
        if render_fixed:
            raw = network_query_fn(pts, viewdirs, run_fn_fixed)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        elif blend:
            raw_fixed = network_query_fn(pts, viewdirs, run_fn_fixed)
            if pts[torch.any(pt_is_in_box, -1)].numel() > 0:
                raw_train = network_query_fn(pts[torch.any(pt_is_in_box, -1)], viewdirs[torch.any(pt_is_in_box, -1)], run_fn)
                # set density to zero for all points outside the box
                raw_in_train = raw_train.clone()
                raw_in_train[pt_is_in_box[torch.any(pt_is_in_box, -1)] == False, :] = 0
                raw_in = torch.zeros(pts.shape[0], pts.shape[1], raw_train.shape[2])
                raw_in[torch.any(pt_is_in_box, -1)] = raw_in_train
            else:
                raw_in = torch.zeros(pts.shape[0], pts.shape[1], raw_fixed.shape[2])
            if not sum_in_activation:
                rgb_map, disp_map, acc_map, weights, depth_map, acc_map_train = raw2outputsBlend(raw_fixed, raw_in, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                                                 pytest=pytest, sum_in_act=sum_in_activation)
            else:
                rgb_map, disp_map, acc_map, weights, depth_map = raw2outputsBlend(raw_fixed, raw_in, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest,
                                                                                  sum_in_act=sum_in_activation)
        else:
            raw_fixed = network_query_fn(pts, viewdirs, run_fn_fixed)
            if pts[torch.any(pt_is_in_box, -1)].numel() > 0:
                raw_train = network_query_fn(pts[torch.any(pt_is_in_box, -1)], viewdirs[torch.any(pt_is_in_box, -1)], run_fn)
                # set density to zero for all points outside the box
                raw_in_box_train = raw_train.clone()
                raw_in_box_train[pt_is_in_box[torch.any(pt_is_in_box, -1)] == False, :] = 0
                raw_in = torch.zeros(pts.shape[0], pts.shape[1], raw_train.shape[2])
                raw_in[torch.any(pt_is_in_box, -1)] = raw_in_box_train
                if use_dist_blend:
                    box_width = torch.norm(box_points[0] - box_points[1]).cpu()
                    box_height = torch.norm(box_points[0] - box_points[4]).cpu()
                    box_depth = torch.norm(box_points[0] - box_points[3]).cpu()
                    cube_diag_len = np.sqrt(box_height.numpy() ** 2 + box_width.numpy() ** 2 + box_depth.numpy() ** 2)
                    dist = torch.tensor(0.0)
                    if data_type == 'llff' and not is360Scene:
                        dist = torch.norm(pts[pt_is_in_box == True] - scene_origin_ndc, dim=-1)
                    elif data_type == 'blender' or is360Scene:
                        dist = torch.norm(pts[pt_is_in_box == True] - scene_origin, dim=-1)
                    dist_factor = (1 - torch.exp(-dist_blend_alpha * (dist / cube_diag_len))).unsqueeze(-1)
                    raw_in[pt_is_in_box == True] = dist_factor * raw_fixed[pt_is_in_box == True] + \
                                                   (1 - dist_factor) * raw_in[pt_is_in_box == True]
            else:
                raw_in = torch.zeros(pts.shape[0], pts.shape[1], raw_fixed.shape[2])

            raw_out_box_fixed = raw_fixed.clone()
            raw = torch.zeros_like(raw_out_box_fixed)
            raw[pt_is_in_box] = raw_in[pt_is_in_box]
            raw[pt_is_in_box == False] = raw_out_box_fixed[pt_is_in_box == False]
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        return rgb_map, disp_map, acc_map, weights, depth_map


def render_in_box_fn(pts, viewdirs, network_query_fn, run_fn, run_fn_fixed, z_vals, rays_d, raw_noise_std, white_bkgd, pytest, blend, pt_is_in_box,
                     sum_in_activation):
    acc_map_train = None
    if blend:
        # render points inside the box
        if pts[torch.any(pt_is_in_box, -1)].numel() > 0:
            raw_train = network_query_fn(pts[torch.any(pt_is_in_box, -1)], viewdirs[torch.any(pt_is_in_box, -1)], run_fn)
            raw_fixed = network_query_fn(pts[torch.any(pt_is_in_box, -1)], viewdirs[torch.any(pt_is_in_box, -1)], run_fn_fixed)
            # set density to zero for all points outside the box
            raw_in_box_fixed = raw_fixed.clone()
            raw_in_box_fixed[pt_is_in_box[torch.any(pt_is_in_box, -1)] == False, :] = 0
            raw_in_box_train = raw_train.clone()
            raw_in_box_train[pt_is_in_box[torch.any(pt_is_in_box, -1)] == False, :] = 0
            raw_fixed = torch.zeros(pts.shape[0], pts.shape[1], raw_train.shape[2])
            raw_fixed[torch.any(pt_is_in_box, -1)] = raw_in_box_fixed
            raw_train = torch.zeros(pts.shape[0], pts.shape[1], raw_train.shape[2])
            raw_train[torch.any(pt_is_in_box, -1)] = raw_in_box_train
        else:
            raw_fixed = torch.zeros(pts.shape[0], pts.shape[1], 4)
            raw_train = torch.zeros(pts.shape[0], pts.shape[1], 4)

        if not sum_in_activation:
            rgb_map_in_box, disp_map_in_box, acc_map_in_box, weights_in_box, depth_map_in_box, acc_map_train = raw2outputsBlend(raw_fixed, raw_train, z_vals,
                                                                                                                                rays_d,
                                                                                                                                raw_noise_std, white_bkgd,
                                                                                                                                pytest=pytest,
                                                                                                                                sum_in_act=sum_in_activation)
        else:
            rgb_map_in_box, disp_map_in_box, acc_map_in_box, weights_in_box, depth_map_in_box = raw2outputsBlend(raw_fixed, raw_train, z_vals, rays_d,
                                                                                                                 raw_noise_std, white_bkgd,
                                                                                                                 pytest=pytest,
                                                                                                                 sum_in_act=sum_in_activation)
    else:
        # render points inside the box
        if pts[torch.any(pt_is_in_box, -1)].numel() > 0:
            raw_train = network_query_fn(pts[torch.any(pt_is_in_box, -1)], viewdirs[torch.any(pt_is_in_box, -1)], run_fn)
            # set density to zero for all points outside the box
            raw_in_box = raw_train.clone()
            raw_in_box[pt_is_in_box[torch.any(pt_is_in_box, -1)] == False, :] = 0
            raw = torch.zeros(pts.shape[0], pts.shape[1], raw_train.shape[2])
            raw[torch.any(pt_is_in_box, -1)] = raw_in_box
        else:
            raw = torch.zeros(pts.shape[0], pts.shape[1], 4)

        rgb_map_in_box, disp_map_in_box, acc_map_in_box, weights_in_box, depth_map_in_box = raw2outputs(raw, z_vals, rays_d,
                                                                                                        raw_noise_std, white_bkgd,
                                                                                                        pytest=pytest)
    return rgb_map_in_box, disp_map_in_box, acc_map_in_box, weights_in_box, depth_map_in_box, acc_map_train


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                render_fixed=False,
                network_fn_fixed=None,
                network_fine_fixed=None,
                scene_origin=None,
                scene_origin_ndc=None,
                render_in_box=False,
                update_bounds=False,
                blend=False,
                sum_in_activation=False,
                radius_render_in_box=None,
                white_bkgd=False,
                raw_noise_std=0.,
                render_full_frame=False,
                pytest=False,
                box_points=None,
                box_points_world=None,
                data_type=None,
                hwf=None,
                sample_scale=None,
                use_dist_blend=False,
                dist_blend_alpha=0,
                is360Scene=False,
                change_color=False,
                retraw=False,
                verbose=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    cube_diag_len = 0
    if update_bounds:  # setting tighter bounds when rendering in box
        box_width = torch.norm(box_points[0] - box_points[1])
        box_height = torch.norm(box_points[0] - box_points[4])
        box_depth = torch.norm(box_points[0] - box_points[3])
        cube_diag_len = torch.sqrt(box_height ** 2 + box_width ** 2 + box_depth ** 2)
        if data_type == 'blender' or is360Scene:
            near = torch.maximum(radius_render_in_box - (cube_diag_len / 2), torch.tensor([0.0001])) * torch.ones_like(bounds[..., 0])
            far = (radius_render_in_box + (cube_diag_len / 2)) * torch.ones_like(bounds[..., 1])
        elif data_type == "llff" and not is360Scene:
            max_edge = torch.max(torch.tensor([box_height, box_width, box_depth]))
            afov = 2 * torch.atan(torch.tensor(sample_scale) / (2 * hwf[2]))  # FOV angle
            radius_in = max_edge / (2 * torch.tan(afov / 2))
            near = torch.maximum(radius_in - (cube_diag_len / 2), torch.zeros_like(radius_in))
            far = radius_in + (cube_diag_len * 1.5)
        else:
            print("Wrong data type!")
            return None

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    if box_points is not None:
        pt_is_in_box = (((box_points[:, 0].min() <= pts[..., 0]) & (box_points[:, 0].max() >= pts[..., 0])) &
                        ((box_points[:, 1].min() <= pts[..., 1]) & (box_points[:, 1].max() >= pts[..., 1])) &
                        ((box_points[:, 2].min() <= pts[..., 2]) & (box_points[:, 2].max() >= pts[..., 2]))).clone().detach()
    if render_full_frame:
        rgb_map, disp_map, acc_map, weights, depth_map = render_full_frame_fn(pts, viewdirs, box_points, network_query_fn, network_fn, network_fn_fixed, z_vals,
                                                                              rays_d, raw_noise_std, white_bkgd, pytest, blend, pt_is_in_box, scene_origin,
                                                                              scene_origin_ndc, dist_blend_alpha, sum_in_activation, data_type, is360Scene,
                                                                              render_fixed, use_dist_blend)
    if render_in_box:
        rgb_map_in_box, disp_map_in_box, acc_map_in_box, weights_in_box, depth_map_in_box, acc_map_train = render_in_box_fn(pts, viewdirs, network_query_fn,
                                                                                                                            network_fn, network_fn_fixed,
                                                                                                                            z_vals,
                                                                                                                            rays_d, raw_noise_std, white_bkgd,
                                                                                                                            pytest, blend, pt_is_in_box,
                                                                                                                            sum_in_activation)

    if N_importance > 0:
        if render_full_frame:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        if box_points is not None:
            if render_in_box:
                rgb_map_in_box_0, acc_map_in_box_0, disp_map_in_box_0, depth_map_in_box_0 = rgb_map_in_box, acc_map_in_box, disp_map_in_box, depth_map_in_box
                if blend and not sum_in_activation:
                    acc_map_train0 = acc_map_train

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        if render_in_box:
            z_samples = sample_pdf(z_vals_mid, weights_in_box[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        if render_full_frame:
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        if box_points is not None:
            pt_is_in_box = (((box_points[:, 0].min() <= pts[..., 0]) & (box_points[:, 0].max() >= pts[..., 0])) &
                            ((box_points[:, 1].min() <= pts[..., 1]) & (box_points[:, 1].max() >= pts[..., 1])) &
                            ((box_points[:, 2].min() <= pts[..., 2]) & (box_points[:, 2].max() >= pts[..., 2]))).clone().detach()
        if render_full_frame:
            run_fn_fixed = network_fn_fixed if network_fine_fixed is None else network_fine_fixed
            run_fn = network_fn if network_fine is None else network_fine
            rgb_map, disp_map, acc_map, weights, depth_map = render_full_frame_fn(pts, viewdirs, box_points, network_query_fn, run_fn, run_fn_fixed, z_vals,
                                                                                  rays_d,
                                                                                  raw_noise_std, white_bkgd, pytest, blend, pt_is_in_box, scene_origin,
                                                                                  scene_origin_ndc, dist_blend_alpha, sum_in_activation, data_type, is360Scene,
                                                                                  render_fixed, use_dist_blend)
        if render_in_box:
            run_fn = network_fn if network_fine is None else network_fine
            run_fn_fixed = network_fn_fixed if network_fine_fixed is None else network_fine_fixed
            rgb_map_in_box, disp_map_in_box, acc_map_in_box, weights_in_box, depth_map_in_box, acc_map_train = render_in_box_fn(pts, viewdirs, network_query_fn,
                                                                                                                                run_fn, run_fn_fixed, z_vals,
                                                                                                                                rays_d, raw_noise_std,
                                                                                                                                white_bkgd,
                                                                                                                                pytest, blend, pt_is_in_box,
                                                                                                                                sum_in_activation)

            # compute center of mass
            weights_dot_pts_in_box = (weights_in_box[:, :, None] * pts)
            weights_sum_in_box = weights_in_box.sum()[None]

    if render_full_frame:
        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        ret.update({'depth_map_full_frame': depth_map})
    else:
        ret = {}
    if box_points is not None:
        if render_in_box:
            ret.update({'rgb_map_in_box': rgb_map_in_box, 'transmittance_in_box': acc_map_in_box})
            if blend and not sum_in_activation:
                ret.update({'transmittance_in_box_train': acc_map_train})

    if N_importance > 0:
        if render_full_frame:
            ret['rgb0'] = rgb_map_0
        if box_points is not None:
            if render_in_box:
                ret['rgb_in_box0'] = rgb_map_in_box_0
                ret['transmittance_in_box0'] = acc_map_in_box_0
                ret['weights_dot_pts_in_box'] = weights_dot_pts_in_box
                ret['weights_sum_in_box'] = weights_sum_in_box
                ret['disp_map_in_box0'] = disp_map_in_box_0
                ret['depth_map_in_box0'] = depth_map_in_box_0
                ret['disp_map_in_box'] = disp_map_in_box
                ret['depth_map_in_box'] = depth_map_in_box
                if blend and not sum_in_activation:
                    ret.update({'transmittance_in_box_train0': acc_map_train0})
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--base_weights", type=str, default='',
                        help='original scene weights')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 50,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 1024,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--N_iters", type=int, default=50000, help='number of training iterations')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--is360Scene", action='store_true',
                        help='if the scene type is 360 or not')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=1000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=2000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=2000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_in_box_img", type=int, default=200,
                        help='frequency of in_box_frame saving')

    # Blending options
    parser.add_argument("--description", type=str, default="A green excavator",
                        help="the text that guides the editing/generation (Or: A photo of a XXX excavator)")
    parser.add_argument("--CLIP", action='store_true', help='training with CLIP')
    parser.add_argument("--BLIP", action='store_true', help='training with BLIP')
    parser.add_argument("--sample_pose", action='store_true', help='sample random pose each iteration')
    parser.add_argument("--sample_scale", type=int, default=60, help='sample scale for patch-based , should be a multiple of 8')
    parser.add_argument("--change_color", action='store_true', help='whether change color only')
    parser.add_argument("--box_points_path", type=str, default='box_points.pt',
                        help='3d cube coordinates for clip:\n'
                             '* Currently has to be a cube\n'
                             '* Front vertexes starts from the top left corner in the front plane\n'
                             '* Rear vertexes starts from the top left corner in the rear plane')
    parser.add_argument("--blend", action='store_true', help="perform blending between the fixed net and the trained clip net")
    parser.add_argument("--sum_in_activation", action='store_true', help="when blending whether preform summation inside or outside activation function")
    parser.add_argument("--max_trans", type=float, default=0.88, help='max transmittance')
    parser.add_argument("--trans_loss_weight", type=float, default=0.25, help='weight of trans loss')
    parser.add_argument("--trans_loss_alpha", type=float, default=0.0002, help='controls the slope in the annealing of the trans loss')
    parser.add_argument("--max_depth_var", type=float, default=0.5, help='max depth variance')
    parser.add_argument("--depth_loss_weight", type=float, default=6., help='weight of trans loss')
    parser.add_argument("--ckpt_num", type=int, default=-1, help="checkpoint number for loading weights")
    parser.add_argument("--depth_loss", action='store_true', help="use depth loss or not")
    parser.add_argument("--zoom_low", type=float, default=0.0, help="low value of zoom range")
    parser.add_argument("--zoom_high", type=float, default=1.0, help="high value of zoom range")
    parser.add_argument("--rads_x_factor", type=float, default=0.8, help="radius factor for the x axis in the camera spiral curve trajectory, for llff data")
    parser.add_argument("--rads_y_factor", type=float, default=0.3, help="radius factor for the y axis in the camera spiral curve trajectory, for llff data")
    parser.add_argument("--p_center_scene_origin", type=float, default=0.9,
                        help="probability of centring around origin or (1-p) of centring around random point in box")
    parser.add_argument("--use_dir_caption", action='store_true', help="if concat directional caption according to pose or not")
    parser.add_argument("--use_dist_blend", action='store_true', help="if use blending according to dist from scene center or not")
    parser.add_argument("--dist_blend_alpha", type=float, default=1.5, help="alpha value for dist blend")

    return parser


############## training clip ###################

def load_data(args):
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)



    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            # due to png values(4th channel is alpha value)
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return None, None, None, None, None, None, None, None, None, None

    return hwf, K, poses, images, i_train, i_val, i_test, near, far, render_poses


def set_train_and_test_args(args, render_kwargs_train, render_kwargs_test, hwf):
    render_kwargs_train['render_full_frame'] = False
    render_kwargs_test['render_full_frame'] = True
    render_kwargs_train['render_fixed'] = False
    render_kwargs_test['render_fixed'] = False
    render_kwargs_train['blend'] = args.blend
    render_kwargs_test['blend'] = args.blend
    render_kwargs_train['is360Scene'] = args.is360Scene
    render_kwargs_test['is360Scene'] = args.is360Scene
    render_kwargs_train['use_dist_blend'] = args.use_dist_blend
    render_kwargs_test['use_dist_blend'] = args.use_dist_blend
    render_kwargs_train['dist_blend_alpha'] = args.dist_blend_alpha
    render_kwargs_test['dist_blend_alpha'] = args.dist_blend_alpha
    render_kwargs_train['sample_scale'] = args.sample_scale
    render_kwargs_test['sample_scale'] = args.sample_scale
    render_kwargs_train['change_color'] = args.change_color
    render_kwargs_test['change_color'] = args.change_color

    render_kwargs_train['data_type'] = args.dataset_type
    render_kwargs_train['sum_in_activation'] = args.sum_in_activation

    # load box coordinates
    render_kwargs_train['box_points'] = torch.load(args.box_points_path)
    render_kwargs_test['box_points'] = render_kwargs_train['box_points']

    # scale focal length according to sample scale size
    f_sample_scale = hwf[2] * (args.sample_scale / hwf[1])
    hwf_sample_scale = [args.sample_scale, args.sample_scale, f_sample_scale]
    render_kwargs_train['hwf'] = hwf_sample_scale
    render_kwargs_test['hwf'] = hwf


def init_scene_origin(args,render_kwargs_train):
    hwf_sample_scale = render_kwargs_train['hwf']
    box_points = render_kwargs_train['box_points']
    ema_scene_origin_world = None
    ema_scene_origin_ndc = None

    # compute center of box
    x_center = (box_points[:, 0].min() + box_points[:, 0].max()) / 2
    y_center = (box_points[:, 1].min() + box_points[:, 1].max()) / 2
    z_center = (box_points[:, 2].min() + box_points[:, 2].max()) / 2

    if args.dataset_type == 'llff' and not args.is360Scene:
        box_center_ndc = torch.stack([x_center, y_center, z_center])
        ema_scene_origin_ndc = EMA(box_center_ndc, decay=0.9995)
        render_kwargs_train['scene_origin_ndc'] = ema_scene_origin_ndc.value

        box_points_world = ndc2world(box_points, [args.sample_scale, args.sample_scale, hwf_sample_scale[2]])
        x_center = (box_points_world[:, 0].min() + box_points_world[:, 0].max()) / 2
        y_center = (box_points_world[:, 1].min() + box_points_world[:, 1].max()) / 2
        z_center = (box_points_world[:, 2].min() + box_points_world[:, 2].max()) / 2
        box_center_world = torch.stack([x_center, y_center, z_center])
        ema_scene_origin_world = EMA(box_center_world, decay=0.9995)
    else:
        box_center = torch.stack([x_center, y_center, z_center])
        ema_scene_origin_world = EMA(box_center, decay=0.9995)

    render_kwargs_train['scene_origin'] = ema_scene_origin_world.value
    return ema_scene_origin_world, ema_scene_origin_ndc


def train_blended_nerf():
    parser = config_parser()
    args = parser.parse_args()

    hwf, K, poses, images, i_train, i_val, i_test, near, far, render_poses = load_data(args)
    if hwf is None:
        return None

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    set_train_and_test_args(args, render_kwargs_train, render_kwargs_test, hwf)

    # set K according to sample scale
    K_sample_scale = np.array([
        [render_kwargs_train['hwf'][2], 0, 0.5 * args.sample_scale],
        [0, render_kwargs_train['hwf'][2], 0.5 * args.sample_scale],
        [0, 0, 1]
    ])

    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move test data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # init module losses
    module_losses = ModuleLosses(args.description, start, hwf=render_kwargs_train['hwf'], alpha=args.trans_loss_alpha, max_trans=args.max_trans,
                                 trans_loss_lambda=args.trans_loss_weight, use_blip=args.BLIP, use_depth_loss=args.depth_loss, max_depth_var=args.max_depth_var,
                                 depth_loss_lambda=args.depth_loss_weight)

    # init scene origin
    ema_scene_origin_world, ema_scene_origin_ndc = init_scene_origin(args, render_kwargs_train)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            render_kwargs_test['render_in_box'] = False
            render_kwargs_test['update_bounds'] = False
            render_kwargs_test['scene_origin'] = ema_scene_origin_world.value
            if args.dataset_type == 'llff' and not args.is360Scene:
                render_kwargs_test['scene_origin_ndc'] = ema_scene_origin_ndc.value
            render_kwargs_test['N_samples'] = 128
            render_kwargs_test['N_importance'] = 256

            rgbs, _ = render_path(render_poses, render_kwargs_test['hwf'], K, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return

    N_iters = 200000 + args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # start training
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        # Sample random image/pose
        batch_rays_in, radius_in_box = sample_img(i_train, images, render_poses, render_kwargs_train['hwf'], K_sample_scale, render_kwargs_train, 0,
                                                  module_losses, args, i)
        # render in box
        render_kwargs_train['render_in_box'] = True
        render_kwargs_train['update_bounds'] = True
        extras = render(args.sample_scale, args.sample_scale, K_sample_scale, chunk=args.chunk, rays=batch_rays_in,
                        verbose=i < 10, retraw=False,
                        **render_kwargs_train)

        #  sample background
        rgb_img_in_box = sample_background(extras['rgb_map_in_box'].view(args.sample_scale, args.sample_scale, -1), extras['transmittance_in_box'], args)
        if 'rgb_in_box0' in extras:
            rgb0_img_in_box = sample_background(extras['rgb_in_box0'].view(args.sample_scale, args.sample_scale, -1),
                                                extras['transmittance_in_box0'], args)
            rgb0_img_in_box = rgb0_img_in_box.permute(2, 0, 1).unsqueeze(0)

        # get transmittance mean
        if args.blend and not args.sum_in_activation:
            transmittance_mean = 1 - extras['transmittance_in_box_train'].mean()
            if 'transmittance_in_box_train0' in extras:
                transmittance_mean0 = 1 - extras['transmittance_in_box_train0'].mean()
        else:
            transmittance_mean = 1 - extras['transmittance_in_box'].mean()
            if 'transmittance_in_box0' in extras:
                transmittance_mean0 = 1 - extras['transmittance_in_box0'].mean()

        # get depth map variance
        if args.depth_loss:
            depth_var = extras['depth_map_in_box'].var()
            if 'depth_map_in_box0' in extras:
                depth_var0 = extras['depth_map_in_box0'].var()

            disp_var = extras['disp_map_in_box'][torch.logical_not(extras['disp_map_in_box'].isnan())].var()
            if 'disp_map_in_box0' in extras:
                disp_var0 = extras['disp_map_in_box0'][torch.logical_not(extras['disp_map_in_box0'].isnan())].var()

        # update scene origin by center of mass
        if extras['weights_sum_in_box'].sum() != 0:
            origin = extras['weights_dot_pts_in_box'] / extras['weights_sum_in_box'].sum()
            origin = torch.tensor([origin[..., 0].sum(), origin[..., 1].sum(), origin[..., 2].sum()])
            if args.dataset_type == 'llff' and not args.is360Scene:
                hwf_sample_scale = [args.sample_scale, args.sample_scale, render_kwargs_train['hwf'][2]]
                origin_world = ndc2world(origin.view(1, -1), hwf_sample_scale)
                origin_world = origin_world[0]
                ema_scene_origin_world.update(origin_world)
                ema_scene_origin_ndc.update(origin)
                render_kwargs_train['scene_origin_ndc'] = ema_scene_origin_ndc.value
            else:
                ema_scene_origin_world.update(origin)
            render_kwargs_train['scene_origin'] = ema_scene_origin_world.value

        if i % args.i_in_box_img == 0:
            os.makedirs(f"{basedir}//{expname}/plots/in_box", exist_ok=True)
            imageio.imsave(f"{basedir}//{expname}/plots/in_box/rgb_img_in_box_{i}.png", rgb_img_in_box.cpu().detach().numpy() * 255)

        rgb_img_in_box = rgb_img_in_box.permute(2, 0, 1).unsqueeze(0)

        # rendering loss
        optimizer.zero_grad()
        if args.depth_loss:
            loss = module_losses(rgb_img_in_box, trans_mean=transmittance_mean, iter=i, rgb0_img_in_box=rgb0_img_in_box, trans_mean0=transmittance_mean0,
                                 depth_var=depth_var, depth_var0=depth_var0)
        else:
            loss = module_losses(rgb_img_in_box, trans_mean=transmittance_mean, iter=i, rgb0_img_in_box=rgb0_img_in_box, trans_mean0=transmittance_mean0)

        del rgb_img_in_box, transmittance_mean, rgb0_img_in_box, transmittance_mean0, batch_rays_in, extras
        torch.cuda.empty_cache()

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** ((i - start) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].module.state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > start:
            # Turn on testing mode
            with torch.no_grad():
                render_kwargs_test['render_in_box'] = False
                render_kwargs_test['update_bounds'] = False
                render_kwargs_test['scene_origin'] = ema_scene_origin_world.value
                if args.dataset_type == 'llff' and not args.is360Scene:
                    render_kwargs_test['scene_origin_ndc'] = ema_scene_origin_ndc.value
                render_kwargs_test['N_samples'] = 128
                render_kwargs_test['N_importance'] = 256
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > start:  # 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_kwargs_test['render_in_box'] = False
                render_kwargs_test['update_bounds'] = False
                render_kwargs_test['scene_origin'] = ema_scene_origin_world.value
                if args.dataset_type == 'llff' and not args.is360Scene:
                    render_kwargs_test['scene_origin_ndc'] = ema_scene_origin_ndc.value
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            if args.BLIP:
                tqdm.write(
                    f"[TRAIN] Iter: {i} Loss: {loss.item()}    CLoss: {module_losses.l_clip.item()}    BLoss: {module_losses.l_blip.item()}  TLoss: {module_losses.l_trans.item()}")
            else:
                if args.depth_loss:
                    tqdm.write(
                        f"[TRAIN] Iter: {i} Loss: {loss.item()}    CLoss: {module_losses.l_clip.item()}   TLoss: {module_losses.l_trans.item()}   DLoss: {module_losses.l_depth.item()}")
                else:
                    tqdm.write(
                        f"[TRAIN] Iter: {i} Loss: {loss.item()}    CLoss: {module_losses.l_clip.item()}        TLoss: {module_losses.l_trans.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """
        global_step += 1
