import warnings

warnings.filterwarnings("ignore")

import json
import os
import random

import numpy as np
import torch
from PIL import Image
import math
from tqdm import tqdm
from scene.utils import Camera
from typing import NamedTuple
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
import torch.nn.functional as F
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.pose_utils import smooth_camera_poses
from time_alignment.affine_clock import normalized_time_coordinate

class CameraInfoDual(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    rgb_image: np.array
    thermal_image: np.array
    FovY: np.array
    FovX: np.array
    TFovY: np.array
    TFovX: np.array
    width: int
    height: int
    image_width_thermal: int
    image_height_thermal: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float
    rgb_path: str
    thermal_path: str
    rgb_name: str
    thermal_name: str
    thermal_source_frame: int
    thermal_source_name: str
    thermal_frame_shift: int
    mask: np.array


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float
    mask: np.array


class Load_hyper_data(Dataset):
    # from 4DGaussians (https://github.com/hustvl/4DGaussians)
    def __init__(self, 
                 datadir, 
                 ratio=1.0,
                 use_bg_points=False,
                 split="train",
                 startime=0,
                 duration=None
                 ):
        
        from .utils import Camera
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)


        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        self.startime = startime
        self.duration = len(self.all_img)//2 if duration == None else duration

        self.all_img = self.all_img[self.startime*2 : (self.startime+self.duration)*2]
        self.val_id = self.val_id[self.startime : self.startime+self.duration]

        self.split = split
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                            (i%4 == 0)])
            self.i_test = self.i_train+2
            self.i_test = self.i_test[:-1,]
        else:
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)

        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]

        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]
        self.selected_time = set(self.all_time)
        self.ratio = ratio
        self.max_time = max(self.all_time)
        self.min_time = min(self.all_time)
        self.i_video = [i for i in range(len(self.all_img))]
        self.i_video.sort()
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')

            self.all_cam_params.append(camera)
        self.all_img_origin = self.all_img
        self.all_depth = [f'{datadir}/depth/{int(1/ratio)}x/{i}.npy' for i in self.all_img]

        self.all_img = [f'{datadir}/images/{int(1/ratio)}x/{i}.png' for i in self.all_img]

        self.h, self.w = self.all_cam_params[0].image_shape
        self.map = {}
        self.image_one = Image.open(self.all_img[0])
        self.image_one_torch = PILtoTorch(self.image_one,None).to(torch.float32)
        if os.path.exists(os.path.join(datadir,"covisible")):
            self.image_mask = [f'{datadir}/covisible/{int(2)}x/val/{i}.png' for i in self.all_img_origin]
        else:
            self.image_mask = None
        self.generate_video_path()

    def generate_video_path(self):
        self.select_video_cams = [item for i, item in enumerate(self.all_cam_params) if i % 1 == 0 ]
        self.video_path, self.video_time = smooth_camera_poses(self.select_video_cams,10)
        self.video_path = self.video_path[:500]
        self.video_time = self.video_time[:500]

        
    def __getitem__(self, index):
        if self.split == "train":
            return self.load_raw(self.i_train[index])
        elif self.split == "test":
            return self.load_raw(self.i_test[index])
        elif self.split == "video":
            return self.load_video(index)
        
    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        elif self.split == "video":
            return len(self.video_path)
            
    def load_video(self, idx):
        startime = self.startime
        duration = self.duration
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]

        w = self.image_one.size[0]
        h = self.image_one.size[1]

        time = self.video_time[idx]

        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        cxr = ((camera.principal_point[0])/ self.w - 0.5)
        cyr = ((camera.principal_point[1])/ self.h - 0.5)

        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=self.image_one, 
                              image_path=image_path, image_name=image_name, width=w, 
                              height=h, near=self.near, far=self.far, timestamp=(time-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr,
                              mask=None)
        self.map[idx] = caminfo
        return caminfo
    
    def load_raw(self, idx):
        startime = self.startime
        duration = self.duration
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        image = []
        img = Image.open(self.all_img[idx])
        image = img.copy()
        img.close()
        w = image.size[0]
        h = image.size[1]

        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R

        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        cxr = ((camera.principal_point[0])/ self.w - 0.5)
        cyr = ((camera.principal_point[1])/ self.h - 0.5)
        
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]
        if self.image_mask is not None and self.split == "test":
            mask = Image.open(self.image_mask[idx])
            mask = PILtoTorch(mask,None)
            mask = mask.to(torch.float32)[0:1,:,:]

            mask = F.interpolate(mask.unsqueeze(0), size=[self.h, self.w], mode='bilinear', align_corners=False).squeeze(0)
        else:
            mask = None
        
        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, 
                              image_path=image_path, image_name=image_name, width=w, 
                              height=h, near=self.near, far=self.far, timestamp=(time-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr,
                              mask=mask)
        self.map[idx] = caminfo
        return caminfo

        
def format_hyper_data(data_class, split, near=None, far=None, startime=0, duration=None):
    if split == "train":
        data_idx = data_class.i_train
    elif split == "test":
        data_idx = data_class.i_test

    cam_infos = []
    for uid, index in tqdm(enumerate(data_idx)):
        camera = data_class.all_cam_params[index]
        image = Image.open(data_class.all_img[index])

        time = data_class.all_time[index]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, data_class.h)
        FovX = focal2fov(camera.focal_length, data_class.w)
        cxr = ((camera.principal_point[0])/ camera.image_size[0] - 0.5)
        cyr = ((camera.principal_point[1])/ camera.image_size[1] - 0.5)
        
        image_path = "/".join(data_class.all_img[index].split("/")[:-1])
        image_name = data_class.all_img[index].split("/")[-1]
        
        if data_class.image_mask is not None and data_class.split == "test":
            mask = Image.open(data_class.image_mask[index])
            mask = PILtoTorch(mask,None)
            
            mask = mask.to(torch.float32)[0:1,:,:]
            
        
        else:
            mask = None
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, 
                              image_path=image_path, image_name=image_name, width=int(data_class.w), 
                              height=int(data_class.h), near=data_class.near, far=data_class.far, timestamp=(time-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr,
                              mask=mask)

        cam_infos.append(cam_info)
    return cam_infos


def _camera_side(name):
    base = os.path.basename(str(name)).lower()
    if "left" in base:
        return "left"
    if "right" in base:
        return "right"
    raise ValueError(f"Camera side missing from id: {name}")


def _thermal_source_index(rgb_data, thermal_data, index, split, frame_shift):
    """Return a same-side, same-split Thermal source index or the identity boundary index."""
    if frame_shift == 0:
        return index

    source_index = index + 2 * frame_shift
    if source_index < 0 or source_index >= len(thermal_data.all_img):
        return index

    target_rgb_id = rgb_data.all_img_origin[index]
    source_thermal_id = thermal_data.all_img_origin[source_index]
    target_frame = int(rgb_data.all_time[index])
    source_frame = int(thermal_data.all_time[source_index])
    if _camera_side(target_rgb_id) != _camera_side(source_thermal_id):
        raise AssertionError(
            f"Temporal shift changed camera side: {target_rgb_id} -> {source_thermal_id}")
    if source_frame != target_frame + frame_shift:
        raise AssertionError(
            f"Temporal shift mismatch: target={target_frame}, source={source_frame}, "
            f"shift={frame_shift}")

    split_ids = set(thermal_data.train_id if split == "train" else thermal_data.val_id)
    if source_thermal_id not in split_ids:
        raise AssertionError(
            f"Temporal shift crossed the native {split} split: {source_thermal_id}")
    return source_index


def format_hyper_dual_data(rgb_data, thermal_data, split, near=None, far=None,
                           startime=0, duration=None, rgb_only_teacher=False):
    if split == "train":
        data_idx = rgb_data.i_train
    elif split == "test":
        data_idx = rgb_data.i_test

    train_frame_shift = int(os.environ.get("ED3DGS_THERMAL_FRAME_SHIFT", "0"))
    eval_frame_shift = int(os.environ.get(
        "ED3DGS_EVAL_THERMAL_FRAME_SHIFT", "0"))
    requested_frame_shift = (
        train_frame_shift if split == "train" else eval_frame_shift)
    endpoint_drift = float(os.environ.get(
        "ED3DGS_THERMAL_ENDPOINT_DRIFT_V34", "0"))
    affine_clock = os.environ.get("ED3DGS_V34_AFFINE_CLOCK", "0") == "1"
    if endpoint_drift != 0.0 and (not affine_clock or split != "train"):
        raise ValueError(
            "Endpoint drift is allowed only for the v34 affine train arm")
    if abs(endpoint_drift) >= 4.0:
        raise ValueError("Endpoint drift must be strictly inside four frames")
    if rgb_only_teacher and requested_frame_shift != 0:
        raise ValueError("RGB-only teacher requires ED3DGS_THERMAL_FRAME_SHIFT=0")
    frame_shift = requested_frame_shift
    cam_infos = []
    shifted_count = 0
    blended_count = 0
    for uid, index in tqdm(enumerate(data_idx)):
        rgb_cam = rgb_data.all_cam_params[index]
        # thermal_cam = thermal_data.all_cam_params[index]

        rgb_img = Image.open(rgb_data.all_img[index]).copy()
        if rgb_only_teacher:
            thermal_source_index = None
            thermal_img = None
        else:
            if endpoint_drift == 0.0:
                thermal_source_index = _thermal_source_index(
                    rgb_data, thermal_data, index, split, frame_shift)
                thermal_img = Image.open(
                    thermal_data.all_img[thermal_source_index]).copy()
            else:
                target_frame = int(rgb_data.all_time[index])
                coordinate = normalized_time_coordinate(
                    target_frame - startime, duration)
                query_shift = frame_shift + endpoint_drift * coordinate
                lower_shift = int(math.floor(query_shift / 2.0) * 2)
                upper_shift = lower_shift + 2
                lower_index = _thermal_source_index(
                    rgb_data, thermal_data, index, split, lower_shift)
                upper_index = _thermal_source_index(
                    rgb_data, thermal_data, index, split, upper_shift)
                if lower_index == index or upper_index == index:
                    thermal_source_index = index
                    thermal_img = Image.open(
                        thermal_data.all_img[index]).copy()
                else:
                    alpha = (query_shift - lower_shift) / 2.0
                    lower_image = Image.open(
                        thermal_data.all_img[lower_index]).copy()
                    upper_image = Image.open(
                        thermal_data.all_img[upper_index]).copy()
                    thermal_img = Image.blend(
                        lower_image, upper_image, float(alpha))
                    thermal_source_index = _thermal_source_index(
                        rgb_data, thermal_data, index, split, frame_shift)
                    blended_count += 1
            shifted_count += int(thermal_source_index != index)

        time = rgb_data.all_time[index]

        # RGB
        R_rgb = rgb_cam.orientation.T
        T_rgb = - rgb_cam.position @ R_rgb

        FovY = focal2fov(rgb_cam.focal_length, rgb_data.h)
        FovX = focal2fov(rgb_cam.focal_length, rgb_data.w)
        cxr = ((rgb_cam.principal_point[0])/ rgb_cam.image_size[0] - 0.5)
        cyr = ((rgb_cam.principal_point[1])/ rgb_cam.image_size[1] - 0.5)

        rgb_path = "/".join(rgb_data.all_img[index].split("/")[:-1])
        rgb_name = rgb_data.all_img[index].split("/")[-1]
        if rgb_only_teacher:
            thermal_path = None
            thermal_name = None
            thermal_source_frame = None
            thermal_source_name = None
        else:
            thermal_path = "/".join(thermal_data.all_img[thermal_source_index].split("/")[:-1])
            thermal_name = thermal_data.all_img[thermal_source_index].split("/")[-1]
            thermal_source_frame = int(thermal_data.all_time[thermal_source_index])
            thermal_source_name = thermal_data.all_img_origin[thermal_source_index]

        cam_info = CameraInfoDual(
            uid=uid,
            R=R_rgb,
            T=T_rgb,
            rgb_image=rgb_img,
            thermal_image=thermal_img,
            FovY=FovY,
            FovX=FovX,
            TFovY=FovY,
            TFovX=FovX,
            width=int(rgb_data.w),
            height=int(rgb_data.h),
            image_width_thermal = int(rgb_data.w),
            image_height_thermal = int(rgb_data.h),
            near=near,
            far=far,
            timestamp=(time - startime) / duration,
            pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr,
            rgb_path=rgb_path,
            thermal_path=thermal_path,
            rgb_name=rgb_name,
            thermal_name=thermal_name,
            thermal_source_frame=thermal_source_frame,
            thermal_source_name=thermal_source_name,
            thermal_frame_shift=(0 if rgb_only_teacher else thermal_source_frame - int(time)),
            mask=None
        )

        cam_infos.append(cam_info)
    if rgb_only_teacher:
        print(f"[RGBTeacher] Thermal {split} images NOT OPENED count={len(cam_infos)}")
    else:
        print(
            f"[TemporalCorruption] split={split} requested_shift={requested_frame_shift} "
            f"effective_shift={frame_shift} "
            f"endpoint_drift={endpoint_drift:+.6f} "
            f"affine_blended={blended_count}/{len(cam_infos)} "
            f"shifted={shifted_count}/{len(cam_infos)} boundary_identity={len(cam_infos) - shifted_count}"
        )
    return cam_infos
