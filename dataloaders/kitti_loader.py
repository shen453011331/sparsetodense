import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
from dataloaders.pose_estimator import get_pose_pnp
import pandas as pd

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']


def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = K[
        0,
        2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = K[
        1,
        2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K


def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb
            or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            args.data_folder,
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            args.data_folder,
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([args.data_folder] + ['data_rgb'] + ps[-6:-4] +
                            ps[-2:-1] + ['data'] + ps[-1:])
            return pnew
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                ps = p.split('/')
                pnew = '/'.join(ps[:-7] +  
                    ['data_rgb']+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                return pnew
        elif args.val == "select":
            transform = no_transform
            glob_d = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            )
            def get_rgb_paths(p):
                return p.replace("groundtruth_depth","image")
    elif split == "test_completion":
        transform = no_transform
        glob_d = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/image/*.png")
    elif split == "test_prediction":
        transform = no_transform
        glob_d = None
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d)) 
        paths_gt = sorted(glob.glob(glob_gt)) 
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:  
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


oheight, owidth = 352, 1216


def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth


def train_transform(rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
        if sparse_s1 is not None and sparse_s2 is not None:
            sparse_s1 = transform_geometric(sparse_s1)
            sparse_s2 = transform_geometric(sparse_s2)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
        if rgb_s1 is not None and rgb_s2 is not None:
            rgb_s1 = transform_rgb(rgb_s1)
            rgb_s2 = transform_rgb(rgb_s2)
    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2


def val_transform(rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2, args):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if rgb_s1 is not None:
        rgb_s1 = transform(rgb_s1)
    if rgb_s2 is not None:
        rgb_s2 = transform(rgb_s2)
    if sparse_s1 is not None:
        sparse_s1 = transform(sparse_s1)
    if sparse_s1 is not None:
        sparse_s1 = transform(sparse_s1)
    return rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2


def no_transform(rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2, args):
    return rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img


def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 3
    candidates = [
        i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]
    while True:
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number + random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(
            path_rgb_tgt)

    return rgb_read(path_near), path_near


def get_rgb_neighbor(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    path_s1 = get_nearby_filename(path, number - 1)
    path_s2 = get_nearby_filename(path, number + 1)
    if os.path.exists(path_s1) and  os.path.exists(path_s2):
        return rgb_read(path_s1), rgb_read(path_s2), path_s1, path_s2
    else:
        print('no 2 neigbor image')
        return None, None, None, None


def get_sparse_neighbor(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    path_s1 = get_nearby_filename(path, number - 1)
    path_s2 = get_nearby_filename(path, number + 1)
    if os.path.exists(path_s1) and  os.path.exists(path_s2):
        return depth_read(path_s1), depth_read(path_s2), path_s1, path_s2
    else:
        print('no 2 neigbor sparse')
        return None, None, None, None


def get_pnp_pose():
    filename = 'dataloaders/pnp_pose.txt'
    df = pd.read_csv(filename, sep=' ', index_col=False, header=None)
    df.columns = ['rgb_path', 'rgb_near_path',
                  'r_vec0', 'r_vec1', 'r_vec2',
                  't_vec0', 't_vec1', 't_vec2', 'success']
    return df


def get_pose(df_temp, rgb, rgb_near, sparse, K, threshold_translation,
             sparse_near):
    if len(df_temp) != 0:
        # print('use_pnp#{}'.format(self.pnp_count))
        # self.pnp_count = self.pnp_count + 1
        r_vec_1 = df_temp.head(1)[['r_vec0', 'r_vec1', 'r_vec2']].values.reshape(3, 1)
        t_vec_1 = df_temp.head(1)[['t_vec0', 't_vec1', 't_vec2']].values.reshape(3, 1)
        success = df_temp.head(1)['success'].values[0]
    else:
        success, r_vec_1, t_vec_1 = get_pose_pnp(rgb, rgb_near, sparse, K)
        # discard if translation is too small
        success = success and LA.norm(t_vec_1) > threshold_translation
    if success:
        r_mat_1, _ = cv2.Rodrigues(r_vec_1)
    else:
        # return the same image and no motion when PnP fails
        rgb_near = rgb
        sparse_near = sparse
        t_vec_1 = np.zeros((3, 1))
        r_mat_1 = np.eye(3)
    return rgb_near, t_vec_1, r_mat_1, sparse_near


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1
        self.path_s1_seq = np.ones(100000).tolist()
        self.path_s2_seq = np.ones(100000).tolist()
        self.sparse_path_s1_seq = np.ones(100000).tolist()
        self.sparse_path_s2_seq = np.ones(100000).tolist()
        pnp_df = get_pnp_pose()
        self.pnp_df = pnp_df
        self.pnp_count = 0

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        sparse = depth_read(self.paths['d'][index]) if \
            (self.paths['d'][index] is not None and self.args.use_d) else None
        target = depth_read(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        if self.split == 'train' and self.args.use_pose:
            rgb_near, path_near = get_rgb_near(self.paths['rgb'][index], self.args)
            rgb_s1, rgb_s2, path_s1, path_s2 = get_rgb_neighbor(self.paths['rgb'][index], self.args)
            sparse_s1, sparse_s2, _, _ = \
                get_sparse_neighbor(self.paths['d'][index], self.args)
            if sparse_s1 is None or sparse_s2 is None:
                sparse_s1 = sparse
                rgb_s1 = rgb
                path_s1 = self.paths['rgb'][index]
                sparse_s2 = sparse
                rgb_s2 = rgb
                path_s2 = self.paths['rgb'][index]
                print('recover sparse')
        else:
            rgb_near = None
            rgb_s1, rgb_s2 = None, None
        if self.split == 'train' and self.args.use_pose:
            self.path_s1_seq[index] = path_s1
            self.path_s2_seq[index] = path_s2
        return rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2

    def __getitem__(self, index):
        rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2 = self.__getraw__(index)
        rgb, sparse, target, rgb_s1, rgb_s2, sparse_s1, sparse_s2 = self.transform(rgb, sparse, target,
                                                       rgb_s1, rgb_s2, sparse_s1, sparse_s2, self.args)
        rgb_path = self.paths['rgb'][index]
        r_mat_1, t_vec_1 = None, None
        r_mat_2, t_vec_2 = None, None
        if self.split == 'train' and self.args.use_pose:
            rgb_s1_path = self.path_s1_seq[index]
            rgb_s2_path = self.path_s2_seq[index]
            df_temp = self.pnp_df[self.pnp_df['rgb_path'] == rgb_path]
            df_temp_s1 = df_temp[df_temp['rgb_near_path'] == rgb_s1_path]
            df_temp_s2 = df_temp[df_temp['rgb_near_path'] == rgb_s2_path]
            rgb_s1, t_vec_1, r_mat_1, sparse_s1 = get_pose(df_temp_s1, rgb, rgb_s1,
                                                sparse, self.K,
                                                self.threshold_translation, sparse_s1)
            rgb_s2, t_vec_2, r_mat_2, sparse_s2 = get_pose(df_temp_s1, rgb, rgb_s1,
                                                sparse, self.K,
                                                self.threshold_translation, sparse_s2)

        rgb, gray = handle_gray(rgb, self.args)
        candidates = {"rgb":rgb, "d":sparse, "gt":target,
                      "g":gray, "r_mat_1":r_mat_1, "t_vec_1":t_vec_1,
                      "rgb_s1":rgb_s1, 'd_s1':sparse_s1,
                      "r_mat_2":r_mat_2, "t_vec_2":t_vec_2,
                      "rgb_s2":rgb_s2, 'd_s2':sparse_s2,}

        # candidates = {"rgb":rgb, "d":sparse, "gt":target, \
        #     "g":gray, "r_mat":r_mat, "t_vec":t_vec, "rgb_near":rgb_near}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])
