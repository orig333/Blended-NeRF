# Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields <br /> [ICCV 2023 AI3DCC]

<a href="https://www.vision.huji.ac.il/blended-nerf/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2306.12760"><img src="https://img.shields.io/badge/arXiv-2111.14818-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch->=1.13.1-Red?logo=pytorch"></a>

<img src="docs/full_flow.png" width="1200px">

> <a href="https://www.vision.huji.ac.il/blended-nerf/">**Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields**</a>
> 
>  Ori Gordon, Omri Avrahami, Dani Lischinski
> 
> Abstract: Editing a local region or a specific object in a 3D scene
represented by a NeRF or consistently blending a new realistic object into the scene is challenging, mainly due to
the implicit nature of the scene representation. We present
Blended-NeRF, a robust and flexible framework for editing a
specific region of interest in an existing NeRF scene, based
on text prompts, along with a 3D ROI box. Our method
leverages a pretrained language-image model to steer the
synthesis towards a user-provided text prompt, along with
a 3D MLP model initialized on an existing NeRF scene
to generate the object and blend it into a specified region
in the original scene. We allow local editing by localizing a 3D ROI box in the input scene, and blend the content synthesized inside the ROI with the existing scene using
a novel volumetric blending technique. To obtain natural
looking and view-consistent results, we leverage existing
and new geometric priors and 3D augmentations for improving the visual fidelity of the final result. We test our
framework both qualitatively and quantitatively on a variety of real 3D scenes and text prompts, demonstrating realistic multi-view consistent results with much flexibility and
diversity compared to the baselines. Finally, we show the
applicability of our framework for several 3D editing applications, including adding new objects to a scene, removing/replacing/altering existing objects, and texture conversion.

# Getting Started
## Installation
1. Create virtual environment:
```bash
$ conda create --name blended-nerf python=3.9
$ conda activate blended-nerf
```
2. Clone repository and install requirements:
```bash
git clone https://github.com/orig333/Blended-NeRF.git
cd Blended-NeRF
pip install -r requirements.txt
```
3. Download scenes data:
```bash
bash download_data.sh nerf_synthetic
bash download_data.sh nerf_llff
bash download_data.sh nerf_real_360
```
# Usage
## Training
1. Create a config.txt file and place it in configs directory.
2. Place existing scene weights in base_weights directory.
3. Place box points .pt file in box_points directory.

Start training:
```bash
python main.py --config ./configs/config.txt
```
We provide a few configs files along with their box_points .pt files and base scene weights.

## Localizing 3D box
In notebooks directory we provide a simple framework for localizing a 3d box in existing NeRF scene.
There are 3 notebooks for blender, llff and llff 360 scenes.
Given a config file you can look around the scene from different angles and distances
and localize a 3d box which can be than used for training.

# Acknowledgments
This code borrows from [NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch), [CLIP](https://github.com/openai/CLIP) and
[BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

# Citation
If you use this code for your research, please cite the following:
```
@article{gordon2023blended,
  title={Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields},
  author={Gordon, Ori and Avrahami, Omri and Lischinski, Dani},
  journal={arXiv preprint arXiv:2306.12760},
  year={2023}
}
```

