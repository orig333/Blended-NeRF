import clip
import numpy as np
import torch
import torchvision.transforms as T
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        else:
            self.model, self.preprocess = clip.load("RN50", device="cpu")
        # self.upsample = torch.nn.Upsample(scale_factor=7)
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text=None, ref_image=None):
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        if text is not None:
            similarity = 1 - self.model(image, text)[0] / 100  # cos is big when angle is small, thus 1 - cos
        elif ref_image is not None:
            ref_image = torch.nn.functional.upsample_bilinear(ref_image, (224, 224))
            ref_image_features = self.model.encode_image(ref_image)
            image_features = self.model.encode_image(image)
            ref_image_features = ref_image_features / ref_image_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ ref_image_features.T)
            similarity = 1 - similarity[0] / 100

        return similarity


class BLIPLoss(torch.nn.Module):
    def __init__(self, loss_type='itm'):
        super().__init__()
        self.loss_type = loss_type
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

    def forward(self, image, caption=None, ref_image=None):
        if caption is not None:
            # Preprocess image and text inputs
            transform = T.ToPILImage()
            image = transform(image[0])  # input image is in shape [1,3,W,H]
            img = self.vis_processors["eval"](image).unsqueeze(0).to(device)
            txt = self.text_processors["eval"](caption)
            if self.loss_type == 'itm':
                # Compute image-text matching (ITM) score
                itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                return itm_scores[:, 0]  # we return the probability that they are not similar and we would like to minimize it
            elif self.loss_type == 'itc':
                itc_score = self.model({"image": img, "text_input": txt}, match_head='itc')
                return 1 - itc_score


def tokenize_text(text):
    return torch.cat([clip.tokenize(text)])


class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)


class TransLoss(torch.nn.Module):
    def __init__(self, start_iter, alpha, max_trans, focal):
        super().__init__()
        self.alpha = alpha
        self.max_trans = torch.tensor(max_trans)
        self.start_iter = start_iter
        self.base_trans = torch.tensor(0.4)
        self.focal = focal
        self.trans_factor = 1

    def forward(self, trans_mean, iteration):
        annealing = 1 - torch.exp(torch.tensor(-(iteration - 200000 + 1) * self.alpha))
        if (iteration - self.start_iter) <= 2000:
            loss = -annealing * self.trans_factor * (torch.minimum(torch.minimum(trans_mean, self.base_trans), self.max_trans))
        else:
            loss = -annealing * self.trans_factor * (torch.minimum(trans_mean, self.max_trans))
        return loss

    def update_trans_factor(self, radius):
        self.trans_factor = torch.minimum(torch.tensor(1.), (self.focal ** 2 / ((radius / 4.) ** 2)) / 1.2)


def pts_center(pts, weights):
    total_weight = torch.sum(weights)
    origin_weights = weights[Ellipsis, None] / total_weight
    origin = origin_weights * pts
    origin = torch.tensor([
        torch.sum(origin[Ellipsis, 0]),
        torch.sum(origin[Ellipsis, 1]),
        torch.sum(origin[Ellipsis, 2]),
    ])  # 3-dim
    return origin, total_weight


class ModuleLosses(torch.nn.Module):
    def __init__(self, caption, start_iter, hwf, alpha=0.0001, max_trans=0.88, trans_loss_lambda=0.25, use_blip=False, use_depth_loss=False,
                 max_depth_var=0.5, depth_loss_lambda=6):
        super().__init__()
        self.use_blip = use_blip
        self.clip_loss = CLIPLoss()
        if self.use_blip:
            self.blip_loss = BLIPLoss()
        self.trans_loss = TransLoss(start_iter, alpha, max_trans, hwf[2])
        self.start_iter = start_iter
        self.caption = caption
        self.alpha = alpha
        self.text_inputs = tokenize_text(caption).to(device)
        self.l_mse = 0
        self.psnr = 0
        self.l_clip = 0
        self.l_blip = 0
        self.l_trans = 0
        self.max_depth_var = max_depth_var
        self.depth_loss_lambda = depth_loss_lambda
        self.l_depth = 0
        self.trans_loss_lambda = trans_loss_lambda
        self.in_radius = 0
        self.target_in = np.empty(0)
        self.l_depth = 0
        self.use_depth_loss = use_depth_loss
        self.zoom = 0

    def update_in_radius(self, radius):
        self.in_radius = radius
        self.trans_loss.update_trans_factor(self.in_radius)

    def update_caption(self, caption):
        self.caption = caption
        self.text_inputs = tokenize_text(caption).to(device)

    def mse2psnr(self, x):
        return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def forward(self, rgb_img_in_box=None, trans_mean=None, iter=None, rgb0_img_in_box=None,
                trans_mean0=None, depth_mean0=None, depth_mean=None, depth_var=None, depth_var0=None):

        # CLIP loss and BLIP loss
        if rgb_img_in_box is not None:
            self.l_clip = self.clip_loss(rgb_img_in_box, self.text_inputs)
            if self.use_blip:
                self.l_blip = self.blip_loss(rgb_img_in_box, self.caption)
            if rgb0_img_in_box is not None:
                self.l_clip = (self.l_clip + self.clip_loss(rgb0_img_in_box, self.text_inputs)) / 2.
                if self.use_blip:
                    self.l_blip = (self.l_blip + self.blip_loss(rgb0_img_in_box, self.caption)) / 2.

        # Transmittance loss
        if trans_mean is not None and iter is not None:
            self.l_trans = self.trans_loss(trans_mean, iter)
            if trans_mean0 is not None:
                self.l_trans = self.l_trans + self.trans_loss(trans_mean0, iter)

        # depth_loss
        if self.use_depth_loss:
            if depth_mean is not None and depth_var is not None:
                self.l_depth = -torch.min(depth_var, torch.tensor(self.max_depth_var))
                if depth_mean0 is not None and depth_var0 is not None:
                    self.l_depth = self.l_depth - torch.min(depth_var0, torch.tensor(self.max_depth_var))
            self.l_depth = self.l_depth * torch.exp(self.zoom)
        else:
            self.l_depth = 0

        if self.use_blip:
            txt_img_similarity_loss = ((self.l_clip + self.l_blip) / 2.)
        else:
            txt_img_similarity_loss = self.l_clip

        return txt_img_similarity_loss + self.trans_loss_lambda * self.l_trans + self.depth_loss_lambda * self.l_depth
