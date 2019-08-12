#! /usr/bin/env python
# coding=utf-8

import os
from core import imageutil
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
import config as cfg
import fileutils
from collections import defaultdict


def read_annotations(dataset_path: str, filename: str):
    """
    read_instances
    :param dataset_path
    :param filename:
    :return:
    """

    # read instances from given filename
    if isinstance(filename, str) and isinstance(dataset_path, str):
        filename = [filename]
        dataset_path = [dataset_path]
    elif not isinstance(filename, list) and isinstance(dataset_path, list):
        raise ValueError("filename and dataset_path both must either be str or list")

    if len(filename) is not len(dataset_path):
        raise ValueError("Same number of filename and dataset_path must be passed")

    instances = list()
    seen_labels = defaultdict(int)

    for idx, annotation_filename in enumerate(filename):

        annotations = fileutils.download_file_as_string(os.path.join(dataset_path[idx], annotation_filename)).split(
            "\\n")

        for line in annotations:
            # initialize an instance for each line
            line = line.strip()

            items = line.split("\\t")
            instance = dict()

            # read image information
            instance["filename"] = os.path.join(dataset_path[idx], items[0].strip())

            # read objects for image
            instance['object'] = list()
            for item in items[1:]:
                # initialize object dict
                item = item.strip()
                obj = dict()
                attributes = item.split(",")
                obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"] = [int(float(e)) for e in attributes[:-1]]
                obj["name"] = attributes[-1]
                # skip object if not name not mentioned in training labels
                if obj['name'] not in cfg.LABELS:
                    print("Image has unseen label {}, not included in cfg.LABELS = {}".format(obj["name"],
                                                                                              str(cfg.LABELS)))
                    continue

                # add object to instance
                instance['object'] += [obj]

                # log number of seen labels
                seen_labels[obj['name']] += 1

            if len(instance['object']) > 0:
                instances += [instance]

    return instances, seen_labels


class Dataset(object):
    """implement Dataset here"""
    def __init__(self):

        self.annot_dataset, self.annot_path = cfg.TRAIN_ANNOTATION_DATASET, cfg.TRAIN_ANNOTATION_PATH
        self.batch_size = cfg.TRAIN_BATCH_SIZE
        self.data_aug = True

        self.train_input_sizes = cfg.TRAIN_INPUT_SIZE
        self.strides = np.array(cfg.NETWORK_STRIDES)
        self.classes = utils.read_class_names(cfg.TRAIN_CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = utils.get_anchors(cfg.TRAIN_ANCHORS)
        self.anchor_per_scale = cfg.TRAIN_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = cfg.MAX_BBOX_PER_SCALE

        self.annotations, self.seen_labels = read_annotations(self.annot_dataset, self.annot_path)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_image(self, image_name):
        # load from gs
        try:
            image = fileutils.load_rgb_image(image_name)
            # TODO add image resizer
            image = image_resize(image, width=cfg.MAX_INPUT_SIZE)
        except Exception as e:
            raise e
        return image

    def parse_annotation(self, annotation: dict):

        image_path = annotation['filename']
        image = self.load_image(image_path)
        bboxes = np.array([list([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]) for obj in annotation['object']])

        if self.data_aug:
            image, bboxes = imageutil.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = imageutil.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = imageutil.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preprocess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def __len__(self):
        return self.num_batchs

    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
            label_sbbox, label_mbbox, label_lbbox = label
            sbboxes, mbboxes, lbboxes = bboxes_xywh
            return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes





