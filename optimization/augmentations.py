import numpy as np
import torch
from math import sqrt, exp

# Constants from lucid/optvis/param/color.py
color_correlation_svd_sqrt = np.asarray(
    [[0.26, 0.09, 0.02],
     [0.27, 0.00, -0.05],
     [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]


def constrain_l_inf(x):
  # NOTE(jainajay): does not use custom grad unlike Lucid
  return x / torch.maximum(torch.tensor(1.0), torch.abs(x))

def rfft2d_freqs(h, w):
  """Computes 2D spectrum frequencies."""
  fy = np.fft.fftfreq(h)[:, None]
  # when we have an odd input dimension we need to keep one additional
  # frequency and later cut off 1 pixel
  fx = np.fft.fftfreq(w)[:w // 2 + 1 + w % 2]
  return np.sqrt(fx * fx + fy * fy)

def rand_fft_image(shape, sd=None, decay_power=1):
  """Generate a random background."""
  b, h, w, ch = shape
  sd = 0.01 if sd is None else sd

  imgs = []
  for _ in range(b):
    freqs = rfft2d_freqs(h, w)
    fh, fw = freqs.shape
    spectrum_var = sd * np.random.normal(0.0, 1.0, [2, ch, fh, fw])
    spectrum = spectrum_var[0] + 1j*spectrum_var[1]
    spectrum_scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w))**decay_power

    # Scale the spectrum by the square-root of the number of pixels
    # to get a unitary transformation. This allows to use similar
    # learning rates to pixel-wise optimisation.
    spectrum_scale *= np.sqrt(w * h)

    scaled_spectrum = spectrum * spectrum_scale
    img = np.fft.irfft2(scaled_spectrum)

    # in case of odd input dimension we cut off the additional pixel
    # we get from irfft2d length computation
    img = img[:ch, :h, :w]
    img = np.transpose(img, [1, 2, 0])

    imgs.append(img)
  return np.stack(imgs) / 4.0

def _linear_correlate_color(t):
  """Multiply input by sqrt of empirical (ImageNet) color correlation matrix.
  If you interpret t's innermost dimension as describing colors in a
  decorrelated version of the color space (which is a very natural way to
  describe colors -- see discussion in Feature Visualization article) the way
  to map back to normal colors is multiply the square root of your color
  correlations.
  Args:
    t: input whitened color array, with trailing dimension 3.
  Returns:
    t_correlated: RGB color array.
  """
  assert t.shape[-1] == 3
  t_flat = np.reshape(t, [-1, 3])
  color_correlation_normalized = (
      color_correlation_svd_sqrt / max_norm_svd_sqrt)
  t_flat = np.matmul(t_flat, color_correlation_normalized.T)
  t_correlated = np.reshape(t_flat, t.shape)
  return t_correlated

def to_valid_rgb(t, decorrelated=False, sigmoid=True):
  """Transform inner dimension of t to valid rgb colors.
  In practice this consists of two parts:
  (1) If requested, transform the colors from a decorrelated color space to RGB.
  (2) Constrain the color channels to be in [0,1], either using a sigmoid
      function or clipping.
  Args:
    t: Input tensor, trailing dimension will be interpreted as colors and
      transformed/constrained.
    decorrelated: If True, the input tensor's colors are interpreted as coming
      from a whitened space.
    sigmoid: If True, the colors are constrained elementwise using sigmoid. If
      False, colors are constrained by clipping infinity norm.
  Returns:
    t with the innermost dimension transformed.
  """
  if decorrelated:
    t = _linear_correlate_color(t)
  if decorrelated and not sigmoid:
    t += color_mean

  if sigmoid:
    return torch.sigmoid(torch.tensor(t))

  return constrain_l_inf(2 * t - 1) / 2 + 0.5


def image_sample(shape, decorrelated=True, sd=None, decay_power=1.):
  raw_spatial = rand_fft_image(shape, sd=sd, decay_power=decay_power)
  return to_valid_rgb(raw_spatial, decorrelated=decorrelated)[0]


def checkerboard(h, w=None, channels=3, tiles=4):
  """Create a shape (w,h,1) array tiled with a checkerboard pattern."""
  color1, color2 = np.random.uniform(low=0.0,high=1.0,size=(2, 3))
  w = w or h
  sq = h // tiles
  canvas = np.full((tiles, sq, tiles, sq, 3), color1, dtype=np.float32)
  canvas[::2, :, 1::2, :, :] = color2
  canvas[1::2, :, ::2, :, :] = color2
  canvas = canvas.reshape((sq * tiles, sq * tiles, 3))
  return canvas


def alpha_blend_background(im, background, alpha=None):
    if torch.isnan(im).any():
        return None
    if int(im.max()) > 1:
        foreground = im[:,:,:3].to(torch.float32) / 255.
    else:
        foreground = im[:, :, :3].to(torch.float32)

    background = background.to(torch.float32)
    if alpha is None:
        alpha = im[:,:,3].to(torch.float32) / 255.
    outImage = foreground*alpha[...,None] + background*(1-alpha[...,None])
    # foreground[:,:,0] = np.multiply(foreground[:,:,0],alpha)
    # foreground[:,:,1] = np.multiply(foreground[:,:,1],alpha)
    # foreground[:,:,2] = np.multiply(foreground[:,:,2],alpha)
    #
    # background[:,:,0] = np.multiply(background[:,:,0],1. - alpha)
    # background[:,:,1] = np.multiply(background[:,:,1],1. - alpha)
    # background[:,:,2] = np.multiply(background[:,:,2],1. - alpha)

    #outImage = (foreground + background)
    # plt.imshow(outImage,interpolation='none')
    # plt.show()
    return outImage


def random_weights_freeze(module_coarse,module_fine):
    module_coarse.requires_grad_(True)
    module_fine.requires_grad_(True)
    freeze = torch.bernoulli(torch.full((len(module_coarse.state_dict().keys()),),0.6))
    for i, (para_c, para_f) in enumerate(zip(module_coarse.parameters(), module_fine.parameters())):
        if freeze[i]:
            para_c.requires_grad = False
            para_f.requires_grad = False

def sample_background(frame, transmittance,args):
    """

    :param frame:
    :param transmittance:
    :return:
    """
    alpha_mask = transmittance.view(args.sample_scale, args.sample_scale)
    h,w = frame.shape[0], frame.shape[1]
    bgs = ['uniform_noise', 'checkerboard' ,'random_fft']
    bg_idx = np.random.randint(0,3)
    bg = bgs[bg_idx]
    if bg == 'uniform_noise':
        noise_bg = np.random.uniform(size=(h,w,3))
        return alpha_blend_background(frame, torch.tensor(noise_bg),alpha_mask)
    elif bg == 'checkerboard':
        tiles_power = np.random.randint(2,4)
        tiles = 2**tiles_power
        checker_bg = checkerboard(h,w,3, tiles=tiles)
        return alpha_blend_background(frame, torch.tensor(checker_bg),alpha_mask)
    elif bg == 'random_fft':
        fft_bg = image_sample([1, h, w, 3], sd=0.2, decay_power=1.5)
        return alpha_blend_background(frame, fft_bg,alpha_mask)


############## find 3d locations ###################


def get_pixel_location(points,pose,K):
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
    return pixels

def get_select_inds(N_samples, iterations, random_scale=True, random_shift=True):
    ####   creating image patch from rays  ####
    # select random locations to sample rays from the original rays from pixel in the image
    # given the grid: select_inds we perform bilinear sampling from the input rays_o and get sample_scale rays
    # that are generate by interpolating the original rays in the location given by select_inds
    # see: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    N_samples_sqrt = int(sqrt(N_samples))
    w, h = torch.meshgrid([torch.linspace(-1, 1, N_samples_sqrt),
        torch.linspace(-1, 1, N_samples_sqrt)])
    h = h.unsqueeze(2)
    w = w.unsqueeze(2)

    scale_anneal = 0.0025
    min_scale = 0.5
    max_scale = 1.0
    if scale_anneal > 0:
        k_iter = iterations // 1000 * 3
        min_scale = max(min_scale, max_scale * exp(-k_iter * scale_anneal))
        min_scale = min(0.9, min_scale)
    else:
        min_scale = 0.25

    scale = 1
    if random_scale:
        scale = torch.Tensor(1).uniform_(min_scale, max_scale)
        h = h * scale
        w = w * scale

    if random_shift:
        max_offset = 1 - scale.item()
        h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2
        w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2

        h += h_offset
        w += w_offset

    return torch.cat([h, w], dim=2)


def sample_rays(rays_o, rays_d,iteration,pose, K,args,render_kwargs_train,target=None):
    if args.crop_by_box:
        H, W = target.size()[0],target.size()[1]
        box_pixels = get_pixel_location(render_kwargs_train['box_points'], pose, K)
        min_row, max_row = torch.max(box_pixels[:, 1].min(), torch.tensor(0)), torch.min(box_pixels[:, 1].max(), torch.tensor(H - 1))
        min_col, max_col = torch.max(box_pixels[:, 0].min(), torch.tensor(0)), torch.min(box_pixels[:, 0].max(), torch.tensor(W - 1))
        cropped_target = target[min_row:max_row, min_col:max_col]
        H_cropped = box_pixels[:, 1].max() - box_pixels[:, 1].min()
        W_cropped = box_pixels[:, 0].max() - box_pixels[:, 0].min()
        rays_o_cropped = rays_o[min_row:max_row, min_col:max_col, :]
        rays_d_cropped = rays_d[min_row:max_row, min_col:max_col, :]

    if render_kwargs_train['render_out_box']:
        select_inds = get_select_inds(int(args.sample_scale_out)**2, iteration)
    else:
        select_inds = get_select_inds(args.sample_scale * args.sample_scale, iteration)
    if args.crop_by_box:
        rays_o = torch.nn.functional.grid_sample(rays_o_cropped.permute(2, 0, 1).unsqueeze(0),
                                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
        rays_d = torch.nn.functional.grid_sample(rays_d_cropped.permute(2, 0, 1).unsqueeze(0),
                                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
    else:
        rays_o = torch.nn.functional.grid_sample(rays_o.permute(2, 0, 1).unsqueeze(0),
                                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
        rays_d = torch.nn.functional.grid_sample(rays_d.permute(2, 0, 1).unsqueeze(0),
                                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]

    rays_o = rays_o.permute(1, 2, 0).view(-1, 3)
    rays_d = rays_d.permute(1, 2, 0).view(-1, 3)

    if target is not None:
        if args.crop_by_box:
            target_s = torch.nn.functional.grid_sample(cropped_target.permute(2, 0, 1).unsqueeze(0),
                                                       select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
        else:
            target_s = torch.nn.functional.grid_sample(target.permute(2, 0, 1).unsqueeze(0),
                                                       select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]

        target_s = target_s.permute(1, 2, 0).view(-1, 3)


        if args.crop_by_box:
            return rays_o, rays_d, target_s,H_cropped,W_cropped

        return rays_o, rays_d, target_s
    else:
        return rays_o, rays_d






# h,w = 800, 800
# im = cv2.imread('data\\nerf_synthetic\\lego\\train\\r_6.png',cv2.IMREAD_UNCHANGED)
#
# white_bg = np.ones((h, w, 3), dtype=np.float32)
# out_img = alpha_blend_background(im, white_bg)
# cv2.imwrite('plots\\backgrounds\\white_background.png',out_img*255)
#
# checker_bg = checkerboard(h, tiles=8)
# out_img = alpha_blend_background(im, checker_bg)
# cv2.imwrite('plots\\backgrounds\\checkerboard_background.png',out_img*255)
#
# noise_bg = np.random.uniform(size=(h, w, 3))
# out_img = alpha_blend_background(im, np.array(noise_bg))
# cv2.imwrite('plots\\backgrounds\\uniform_noise_background.png',out_img*255)
#
# t_bg = image_sample([1, h, w, 3], sd=0.2, decay_power=1.5)
# out_img = alpha_blend_background(im, np.array(t_bg))
# cv2.imwrite('plots\\backgrounds\\fft_background.png',out_img*255)


# if __name__ == '__main__':
#
#     im = cv2.imread('data\\nerf_synthetic\\lego\\train\\r_6.png',cv2.IMREAD_UNCHANGED)
#     out = sample_background(im)
#     cv2.imwrite('plots\\backgrounds\\random_background.png',out*255)