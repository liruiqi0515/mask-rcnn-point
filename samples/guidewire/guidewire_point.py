#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python guidewire_point.py train --dataset=/home/li/GuideWire/guidewire_2020 --weights=coco

    # Resume training a model that you had trained earlier
    python guidewire_point.py train --dataset=/home/li/GuideWire/guidewire_2020 --weights=last

    # Train a new model starting from ImageNet weights
    python guidewire_point.py train --dataset=/home/li/GuideWire/guidewire_2020 --weights=imagenet
    
    # Train a new model starting from random
    python guidewire_point.py train --dataset=/home/li/GuideWire/guidewire_2020 --weights=None

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import datetime
import numpy as np
import skimage.draw
import cv2, shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model_point as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs_point_test", 'res50_FF_hour256_8_0.0005_b2_k11')  # change_point
if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)
sourceDir = './guidewire_point.py'
# targetDir = os.path.join(DEFAULT_LOGS_DIR, 'guidewire_point.py')
# shutil.copy(sourceDir, targetDir)
TRAIN_LAYER = 'all'
gpu_num = '3'  # change_point
EPOCH = 80
TXT_DIR = '/home/li/GuideWire/guidewire_2020/txt'
IMAGE_DIR = '/home/li/GuideWire/guidewire_2020'
MASK_DIR = '/home/li/GuideWire/guidewire_2020/seg'
# guidewire_mode = 'same'           #change_point
guidewire_mode = 'two'


############################################################
#  Configurations
############################################################


class GuidewireConfig(Config):
    """Configuration for training on the guidewire dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    # TRAIN_ROIS_PER_IMAGE = 50
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    max_contrast_delta = 0.1
    max_bright_delta = 0.1
    NAME = "guidewire"
    GPU_COUNT = 1  # change_point
    LEARNING_RATE = 0.0005  # change_point
    USE_MASK = False  # change_point
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (64, 64)
    USE_P2 = True  # change_point
    BRANCH = 'hour'  # change_point in 'old'
    MASK_CHANNEL = 256  # change_point
    DOUBLE = False  # change_point
    MASK_POOL_SIZE = 32  # change_point
    MASK_SHAPE = [64, 64]  # change_pointl
    MASK_LOSS_TYPE = 'normal'  # ['cls', '1channel', 'normal']
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + guidewire
    # DSA iamge can not contain more than 5 guidewires
    DETECTION_MAX_INSTANCES = 5
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500
    TRAIN_BN_BACKBONE = False  # change_point (False or None)   Not trainable, but training in BN,
    TRAIN_BN_NEW = False  # change_point (False or None)
    # confidence threshold of detections
    DETECTION_MIN_CONFIDENCE = 0.9
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "class_loss": 1.,
        "bbox_loss": 1.,
        "heatmap_loss": 8.,
        "mask_loss": 4.  # change_point
    }
    if guidewire_mode == 'same':
        NUM_KEYPOINT = 1
        SAME_MODE = True
    else:
        NUM_KEYPOINT = 2
        SAME_MODE = False
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    SIGMA = 11  # change_point
    PRE_TRAIN = True  # change_point
    BACKBONE = "resnet50"  # change_point
    # BACKBONE = "resnet101"
    # BACKBONE = "mobilenetv2"

    VALIDATION_STEPS = 50
    INFERENCE_NUM = 200


############################################################
#  Dataset
############################################################

class GuidewireDataset(utils.Dataset):

    def load_guidewire(self, dataset_dir, subset, txt_dir=TXT_DIR):
        """Load a subset of the guidewire dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("guidewire", 1, "guidewire")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        image_list = os.listdir(dataset_dir)

        # Add images
        for a in image_list:
            txt_path = os.path.join(txt_dir, a[:-4] + '.txt')
            with open(txt_path, 'r')as f:
                string = f.readlines()
                point = []
                for s in string:
                    point_item = [int(i) for i in s[:-1].split(' ')]
                    point.append(point_item)
                #################################print

            image_path = os.path.join(dataset_dir, a)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            # add a image information (as a dict) into a list: self.image_info
            self.add_image(
                "guidewire",
                image_id=a,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                point=point)

    def makeGaussian(self, height, width, sigma=3, center=None):
        """ make一个高斯核，是生成heatmap的一个部分
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        x0 = center[0]
        y0 = center[1]
        output = np.exp(-np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / (sigma ** 2)) * 255
        return output.astype(np.uint8)

    def load_mask(self, image_id, mask_dir=MASK_DIR):
        """Generate instance masks for an image.
        image_id: image_name
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "guidewire":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask_now = cv2.imread(os.path.join(mask_dir, image_info["id"][:-4] + '.png'), 0)
        mask = np.zeros([info["height"], info["width"], len(info["point"])], dtype=np.uint8)
        point = np.zeros([len(info["point"]), 2, 2], dtype=np.int32)
        bbox = np.zeros([len(info["point"]), 4], dtype=np.int32)
        # print(info["point"])
        for i, p in enumerate(info["point"]):
            bbox[i] = np.array([p[2], p[0], p[3], p[1]])
            point[i, 0] = np.array([p[4], p[5]], dtype=np.int32)
            point[i, 1] = np.array([p[6], p[7]], dtype=np.int32)
            mask_temp = np.zeros([info["height"], info["width"]], dtype=np.uint8)
            mask_temp[p[2]:p[3], p[0]:p[1]] = np.ones([p[3] - p[2], p[1] - p[0]], dtype=np.uint8)
            mask_temp = mask_temp * mask_now
            mask[:, :, i] = mask_temp
        # print('load_mask:', mask.max())  # 255
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32), bbox.astype(np.int32), point.astype(
            np.int32)

    def load_point(self, image_id):
        """Generate instance keypoints, classes and boxes for an image.
        image_id: image_name
       Returns:
        points: A int32 array of shape [instance count, num_keypoints, 2]
        class_ids: a 1D array [instance count] of class IDs of the instance masks.
        bbox: bounding box of the instance. shape: [instance count, 4] not normalized
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "guidewire":
            return super(self.__class__, self).load_point(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        num_point = 2
        point = np.zeros([len(info["point"]), num_point, 2], dtype=np.int32)
        bbox = np.zeros([len(info["point"]), 4], dtype=np.int32)
        # print(info["point"])
        for i, p in enumerate(info["point"]):
            bbox[i] = np.array([p[2], p[0], p[3], p[1]])
            point[i, 0] = np.array([p[4], p[5]], dtype=np.int32)
            point[i, 1] = np.array([p[6], p[7]], dtype=np.int32)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return point.astype(np.int32), np.ones([point.shape[-2]], dtype=np.int32), bbox.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "guidewire":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = GuidewireDataset()
    dataset_train.load_guidewire(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GuidewireDataset()
    dataset_val.load_guidewire(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=EPOCH,
                layers=TRAIN_LAYER)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect guidewires.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar=IMAGE_DIR,
                        help='Directory of the guidewire dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GuidewireConfig()
    else:
        class InferenceConfig(GuidewireConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()
    not_last = True
    if args.weights.lower() == "last":
        not_last = False
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs, not_last=not_last)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs, not_last=not_last)

    # Select weights file to load. only resnet has coco pre-trained model
    if args.weights.lower() == "coco":
        # Download weights file
        if not os.path.exists(COCO_WEIGHTS_PATH):
            utils.download_trained_weights(COCO_WEIGHTS_PATH)
        print('load_weight_coco')
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "mrcnn_mask_deconv",
            "mrcnn_mask_bn4", "mrcnn_mask_conv4",
            "mrcnn_mask_bn3", "mrcnn_mask_conv3",
            "mrcnn_mask_bn2", "mrcnn_mask_conv2",
            "mrcnn_mask_bn1", "mrcnn_mask_conv1"])
    elif args.weights.lower() == "last":
        # Find last trained weights
        print('load_weight_last')
        model.load_weights(model.find_last(), by_name=True)
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        print('load_weight_imagenet')
        # model.load_weights_include(COCO_WEIGHTS_PATH, by_name=True, include=[
        #     "mrcnn_class_conv1", "mrcnn_class_bn1",
        #     "mrcnn_class_conv2", "mrcnn_class_bn2"])
    else:
        pass

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
