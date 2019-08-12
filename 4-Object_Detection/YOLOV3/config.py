# coding:utf-8
import os

# REMOTE ENVIRONMENT
# GOOGLE_CLOUD_SERVICE_ACCOUNT_KEY_PATH = "/home/jovyan/radius-retail.json"
GOOGLE_CLOUD_SERVICE_ACCOUNT_KEY_PATH = "/tf/radius-retail.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_SERVICE_ACCOUNT_KEY_PATH
GS_BUCKET_NAME = "radius-retail-data-science-model"
S3_BUCKET_NAME = "radiusiot-videos"

# NETWORK INIT
MODEL_NAME = "keras-yolo"
LABELS = ['person', 'car', 'licenseplate', 'motorbike', 'bicycle', 'bus', 'truck']
GPUS = "1"
# Ratio between network input's size and network output's size, 32 for YOLOv3
DOWNSAMPLE = 32

# NETWORK FLOW
MAX_BBOX_PER_SCALE = 150



MIN_INPUT_SIZE = 224
MAX_INPUT_SIZE = 480

LEARNING_RATE = 1e-5
GRID_SCALES = [1, 1, 1]
OBJECT_SCALE = 5
NO_OBJECT_SCALE = 1
XYWH_SCALE = 1
CLASS_SCALE = 1
OBJECTNESS_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45



TRAIN_CACHE_NAME = "keras_yolo_train_cache.pkl"
TRAIN_TIMES = 1
TRAIN_EPOCHS = 100
WARMUP_EPOCHS = 3
# WARM-UP BATCHES = WARM-UP EPOCHS * (BATCHES_IN_AN_EPOCH) (1 EPOCH = 1 TRAINING PERIOD)

# NETWORK TEST
TEST_ANNOTATION_PATH = 'dataset/voc/test_annotation.txt'
TEST_CACHE_NAME = "keras_yolo_test_cache.pkl"
TEST_EPOCHS = 1

# LOCAL ENVIRONMENT
LOG_DIR = "logs"
WEIGHTS_DIR = 'weights'
TENSORBOARD_DIR = "tb_logs"

""""""""
# NETWORK TRAIN
TFRECORD_DATASET = "gs://{}/{}".format(GS_BUCKET_NAME, "dataset/beh1/beh1_images.tfrecords")
TRAIN_ANNOTATION_PATH = 'beh1_annotation.txt'
TRAIN_BATCH_SIZE = 8
TRAIN_INPUT_SIZE = [416]
TRAIN_CLASSES = "beh.names"
TRAIN_ANCHORS = "anchors.txt"
TRAIN_ANCHOR_PER_SCALE = 3

# NETWORK CONF
NETWORK_STRIDES = [8, 16, 32]
