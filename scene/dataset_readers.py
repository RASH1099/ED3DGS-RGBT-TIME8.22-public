#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data, format_hyper_dual_data
import copy
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import glob
import natsort
import torch
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    thermal: np.array
    image_path: str
    image_name: str
    thermal_path: str
    thermal_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    

# def getNerfppNorm(cam_info):
#     def get_center_and_diag(cam_centers):
#         cam_centers = np.hstack(cam_centers)
#         avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
#         center = avg_cam_center
#         dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
#         diagonal = np.max(dist)
#         return center.flatten(), diagonal

#     cam_centers = []

#     for cam in cam_info:
#         W2C = getWorld2View2(cam.R, cam.T)
#         C2W = np.linalg.inv(W2C)
#         cam_centers.append(C2W[:3, 3:4])

#     center, diagonal = get_center_and_diag(cam_centers)
#     radius = diagonal * 1.1

#     translate = -center

#     return {"translate": translate, "radius": radius}


def getNerfppNorm(cam_info):
    """
    计算场景的中心和平移归一化参数（NeRF++规范）。
    兼容单模态 CameraInfo 和双模态 CameraInfoDual。
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        # 支持双模态
        if hasattr(cam, "rgb") and isinstance(cam.rgb, dict):
            extrinsics = cam.rgb["extrinsics"]
        else:
            # 兼容旧格式
            extrinsics = getattr(cam, "extrinsics", None)
            if extrinsics is None:
                R = getattr(cam, "R", None)
                T = getattr(cam, "T", None)
                if R is not None and T is not None:
                    extrinsics = np.concatenate([R, T.reshape(3, 1)], axis=1)
                else:
                    raise AttributeError(f"Camera object {cam} has no extrinsics / R / T attribute.")

        # W2C -> C2W 转换
        W2C = getWorld2View2_from_extrinsics(extrinsics)
        C2W = np.linalg.inv(W2C)

        cam_centers.append(C2W[:3, 3:4])

    # 聚合中心与半径
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}


def getWorld2View2_from_extrinsics(extrinsics):
    """兼容从 3x4 外参矩阵构造 4x4 齐次矩阵"""
    W2C = np.eye(4)
    W2C[:3, :4] = extrinsics
    return W2C


def readColmapCamerasDynerf(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=300):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics): 
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x / 2, height / 2)
            FovX = focal2fov(focal_length_x / 2, width / 2)

            t_focal_x = intr.params[0]
            TFovY = focal2fov(t_focal_x, intr.height)
            TFovX = focal2fov(t_focal_x, intr.width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1] 
            FovY = focal2fov(focal_length_y / 2, height / 2)
            FovX = focal2fov(focal_length_x / 2, width / 2)

            t_focal_x = intr.params[0]
            t_focal_y = intr.params[1]
            TFovY = focal2fov(t_focal_y, intr.height)
            TFovX = focal2fov(t_focal_x, intr.width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        height = intr.height / 2
        width = intr.width / 2

        for j in range(startime, startime+int(duration)):
            image_path = os.path.join(images_folder,f"images/{extr.name[:-4]}", "%04d.png" % j)
            thermal_path = os.path.join(images_folder,f"thermal/{extr.name[:-4]}", "%04d.png" % j)
            image_name = os.path.join(f"{extr.name[:-4]}", image_path.split('/')[-1])
            thermal_name = os.path.join(f"{extr.name[:-4]}", thermal_path.split('/')[-1])

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            assert os.path.exists(thermal_path), "Thermal {} does not exist!".format(thermal_path)
            if j == startime:
                image = Image.open(image_path)
                image = image.resize((int(width), int(height)), Image.LANCZOS)
                thermal = Image.open(thermal_path)
                thermal = thermal.resize((int(width), int(height)), Image.LANCZOS)
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,TFovY=TFovY,TFovX=TFovX,  image=image, thermal=thermal, image_path=image_path, image_name=image_name, thermal_path=thermal_path, thermal_name=thermal_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1,cxr=0.0, cyr=0.0)
            else:
                image = None
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,TFovY=TFovY,TFovX=TFovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, thermal=thermal, thermal_path=thermal_path, thermal_name=thermal_name, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasTechnicolorTestonly(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics): 
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)

            t_focal_x = intr.params[0]
            TFovY = focal2fov(t_focal_x, intr.height)
            TFovX = focal2fov(t_focal_x, intr.width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

            t_focal_x = intr.params[0]
            t_focal_y = intr.params[1]
            TFovY = focal2fov(t_focal_y, intr.height)
            TFovX = focal2fov(t_focal_x, intr.width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder,f"images/{extr.name[:-4]}", "%04d.png" % j)
            thermal_path = os.path.join(images_folder,f"thermal/{extr.name[:-4]}", "%04d.png" % j)
            image_name = os.path.join(f"{extr.name[:-4]}", image_path.split('/')[-1])
            thermal_name = os.path.join(f"{extr.name[:-4]}", thermal_path.split('/')[-1])
        
            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            assert os.path.exists(thermal_path), "Thermal {} does not exist!".format(thermal_path)
            
            if image_name == "cam10":
                image = Image.open(image_path)
                thermal = Image.open(thermal_path)
            else:
                image = None
                thermal = None

            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, TFovY=TFovY,TFovX=TFovX, image=image, thermal=thermal, image_path=image_path, image_name=image_name, thermal_path=thermal_path, thermal_name=thermal_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,TFovY=TFovY,TFovX=TFovX,  image=image, thermal=thermal, image_path=image_path, image_name=image_name, thermal_path=thermal_path, thermal_name=thermal_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasTechnicolor(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics): 
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)


            t_focal_x = intr.params[0]
            TFovY = focal2fov(t_focal_x, intr.height)
            TFovX = focal2fov(t_focal_x, intr.width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

            t_focal_x = intr.params[0]
            t_focal_y = intr.params[1]
            TFovY = focal2fov(t_focal_y, intr.height)
            TFovX = focal2fov(t_focal_x, intr.width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder,f"images/{extr.name[:-4]}", "%04d.png" % j)
            thermal_path = os.path.join(images_folder,f"thermal/{extr.name[:-4]}", "%04d.png" % j)
            image_name = os.path.join(f"{extr.name[:-4]}", image_path.split('/')[-1])
            thermal_name = os.path.join(f"{extr.name[:-4]}", thermal_path.split('/')[-1])

            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 
    
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            assert os.path.exists(thermal_path), "Thermal {} does not exist!".format(thermal_path)
            image = Image.open(image_path)
            thermal = Image.open(thermal_path)

            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, TFovY=TFovY,TFovX=TFovX, image=image, thermal=thermal, image_path=image_path, image_name=image_name, thermal_path=thermal_path, thermal_name=thermal_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, TFovY=TFovY,TFovX=TFovX, image=image, thermal=thermal, image_path=image_path, image_name=image_name,
                thermal_name=thermal_name, thermal_path=thermal_path,
                width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def normalize(v):
    return v / np.linalg.norm(v)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), #('t','f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfoDynerf(path, images, eval, duration=300, testonly=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    near = 0.01
    far = 100

    cam_infos_unsorted = readColmapCamerasDynerf(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path, near=near, far=far, duration=duration)    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
     
    train_cam_infos = [_ for _ in cam_infos if "cam00" not in _.image_name]
    test_cam_infos = [_ for _ in cam_infos if "cam00" in _.image_name]

    uniquecheck = []
    for cam_info in test_cam_infos:
        if cam_info.image_name[:5] not in uniquecheck:
            uniquecheck.append(cam_info.image_name[:5])
    assert len(uniquecheck) == 1 
    
    sanitycheck = []
    for cam_info in train_cam_infos:
        if  cam_info.image_name[:5] not in sanitycheck:
            sanitycheck.append( cam_info.image_name[:5])
    for testname in uniquecheck:
        assert testname not in sanitycheck

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3D_downsample.ply")
    
    if not testonly:
        try:
            pcd = fetchPly(ply_path)
        except Exception as e:
            print("error:", e)
            pcd = None
    else:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapSceneInfoTechnicolor(path, images, eval, duration=None, testonly=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    near = 0.01
    far = 100

    if testonly:
        cam_infos_unsorted = readColmapCamerasTechnicolorTestonly(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path, near=near, far=far, duration=duration)
    else:
        cam_infos_unsorted = readColmapCamerasTechnicolor(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path, near=near, far=far, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
     
    train_cam_infos = [_ for _ in cam_infos if "cam10" not in _.image_name]
    test_cam_infos = [_ for _ in cam_infos if "cam10" in _.image_name]

    uniquecheck = []
    for cam_info in test_cam_infos:
        if cam_info.image_name[:5] not in uniquecheck:
            uniquecheck.append(cam_info.image_name[:5])
    assert len(uniquecheck) == 1 
    
    sanitycheck = []
    for cam_info in train_cam_infos:
        if  cam_info.image_name[:5] not in sanitycheck:
            sanitycheck.append( cam_info.image_name[:5])
    for testname in uniquecheck:
        assert testname not in sanitycheck

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3D_downsample.ply")
    if not testonly:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readHyperDataInfos(datadir, use_bg_points, eval, startime=0, duration=None,
                       load_test_cameras=True, rgb_only_teacher=False):
    rgb_dir = os.path.join(datadir, "rgb")
    thermal_dir = os.path.join(datadir, "thermal")

    # train_cam_infos = Load_hyper_data(datadir, 0.5, use_bg_points, split ="train", startime=startime, duration=duration)
    # test_cam_infos = Load_hyper_data(datadir, 0.5, use_bg_points, split="test", startime=startime, duration=duration)

    rgb_data = Load_hyper_data(rgb_dir, 1, use_bg_points, split="train", startime=startime, duration=duration)
    thermal_data = None
    if not rgb_only_teacher:
        thermal_data = Load_hyper_data(
            thermal_dir, 1, use_bg_points, split="train",
            startime=startime, duration=duration)
    test_rgb = None
    test_thermal = None
    if load_test_cameras:
        test_rgb = Load_hyper_data(
            rgb_dir, 1, use_bg_points, split="test",
            startime=startime, duration=duration)
        if not rgb_only_teacher:
            test_thermal = Load_hyper_data(
                thermal_dir, 1, use_bg_points, split="test",
                startime=startime, duration=duration)
    print("load finished")

    # train_cam = format_hyper_data(train_cam_infos,"train", 
    #                               near=train_cam_infos.near, far=train_cam_infos.far,
    #                               startime=train_cam_infos.startime, duration=train_cam_infos.duration)
    
    train_cam = format_hyper_dual_data(rgb_data, thermal_data, "train", 
                                       near=rgb_data.near, far=rgb_data.far,
                                       startime=rgb_data.startime, duration=rgb_data.duration,
                                       rgb_only_teacher=rgb_only_teacher)
    test_cam = []
    if load_test_cameras:
        test_cam = format_hyper_dual_data(
            test_rgb, test_thermal, "test",
            near=test_rgb.near, far=test_rgb.far,
            startime=test_rgb.startime, duration=test_rgb.duration,
            rgb_only_teacher=rgb_only_teacher)

    print("format finished")

    nerf_normalization = getNerfppNorm(train_cam)
    video_cam_infos = copy.deepcopy(test_cam) if load_test_cameras else []

    # ply_path = os.path.join(datadir, "points3D_downsample.ply")
    ply_path = os.path.join(rgb_dir, "points3D.ply")
    bin_path = os.path.join(rgb_dir, "points3D.bin")
    txt_path = os.path.join(rgb_dir, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)     
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)
    pcd = pcd._replace(points=xyz)

    # scene_info = SceneInfo(point_cloud=pcd,
    #                        train_cameras=train_cam_infos,
    #                        test_cameras=test_cam_infos,
    #                        video_cameras=video_cam_infos,
    #                        nerf_normalization=nerf_normalization,
    #                        ply_path=ply_path,
    #                        )
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam,
                           test_cameras=test_cam,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Technicolor": readColmapSceneInfoTechnicolor,
    "Nerfies": readHyperDataInfos,
    "Dynerf": readColmapSceneInfoDynerf,
}
