import argparse

import cv2
import numpy as np
import torch
import random
import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration

        print(self.file_names[self.idx])
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)

        bordersize = 111
        if img.shape[0] > img.shape[1]:
            top_pad = bordersize
            lef_pad = (bordersize * 2 + img.shape[0] - img.shape[1]) // 2
        else:
            top_pad = (bordersize * 2 + img.shape[1] - img.shape[0]) // 2
            lef_pad = bordersize

        padding_warped_image = cv2.copyMakeBorder(
                    img,
                    top=top_pad,
                    bottom=top_pad,
                    left=lef_pad,
                    right=lef_pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                
        if padding_warped_image.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return padding_warped_image


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height
    print("HW: ", height, width)

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)

    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))    
    
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    
    # print(len(heatmaps))
    # for i in range(8):
    #     heatmap = heatmaps[:, :, i]
    #     cv2.imshow("Heat map", heatmap)
    #     cv2.waitKey(0)
    
    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    
    # print(len(pafs))
    # for paf in pafs:
    #     print(paf.shape)
    #     cv2.imshow("paf map", paf)
    #     cv2.waitKey(0)
    return heatmaps, pafs, scale, pad

import time

def run_demo(net, image_provider, height_size, cpu, track_ids):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    print("Number of keypoints", num_keypoints)
    previous_poses = []
    for img in image_provider:
        start_time = time.time()
        orig_img = img.copy()
        scale = 368 / orig_img.shape[0]
        orig_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        img = orig_img.copy()
        start_time = time.time()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)
            
        print(time.time() - start_time)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        print("Time = {}".format(time.time() - start_time))
        print("All keypoints", all_keypoints)
        print("Total kps", len(all_keypoints))
        try:
            cv2.line(img, (int(all_keypoints[0][0]), int(all_keypoints[0][1])), (int(all_keypoints[1][0]), int(all_keypoints[1][1])), Pose.color, 2)
            cv2.line(img, (int(all_keypoints[1][0]), int(all_keypoints[1][1])), (int(all_keypoints[2][0]), int(all_keypoints[2][1])), Pose.color, 2)
            cv2.line(img, (int(all_keypoints[2][0]), int(all_keypoints[2][1])), (int(all_keypoints[3][0]), int(all_keypoints[3][1])), Pose.color, 2)
            cv2.line(img, (int(all_keypoints[3][0]), int(all_keypoints[3][1])), (int(all_keypoints[0][0]), int(all_keypoints[0][1])), Pose.color, 2)

            # for i, kp in enumerate(all_keypoints):
            #     if i + 1 == len(all_keypoints):
            #         break
            #     cv2.line(img, (int(kp[i][0]), int(all_keypoints[i][1])), (int(all_keypoints[i+1][0]), int(all_keypoints[i+1][1])), Pose.color, 2)
            cv2.imwrite("visualize/Img_{}.png".format(str(random.randint(0, 1000))), img)
            continue
        except Exception:
            continue
        
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][5])
            current_poses.append(pose)
            pose.draw(img)

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        if track_ids == True:
            propagate_ids(previous_poses, current_poses)
            previous_poses = current_poses
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        # cv2.waitKey(0)
        # if key == 27:  # esc
        #     return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default='weights\checkpoint_iter_3200.pth', help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=368, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default=[r'D:\Coding\DocHomographyGenerator\res\output\train_1\Image_39.png'], help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track-ids', default=True, help='track poses ids')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')
    list_file = []
    # dir_test = r'D:\\Coding\\DocHomographyGenerator\\res\\output\\train_2\\'
    # dir_test = r"D:\Coding\DocHomographyGenerator\res\output\train_2\\"
    dir_test = r"C:\Users\ADMIN\Downloads\test_data\test_data"
    for file in os.listdir(dir_test):
        if file.endswith("JPG") or file.endswith("jpg") or file.endswith("png"):
            list_file.append(os.path.join(dir_test, file))
            args.images = list_file

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    
    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)

    run_demo(net, frame_provider, args.height_size, False, args.track_ids)
