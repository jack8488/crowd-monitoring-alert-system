import datetime

# -------------------- Video Source Settings --------------------
VIDEO_CONFIG = {
    "VIDEO_CAP": "video7.mp4",
    "IS_CAM": False,
    "CAM_APPROX_FPS": 3,
    "HIGH_CAM": False,
    "START_TIME": datetime.datetime(2020, 11, 5, 0, 0, 0, 0)
}

# -------------------- YOLOv4-Tiny Configuration --------------------
YOLO_CONFIG = {
    "WEIGHTS_PATH": "YOLOv4-tiny/yolov4-tiny.weights",
    "CONFIG_PATH": "YOLOv4-tiny/yolov4-tiny.cfg"
}

# -------------------- General Display Settings --------------------
SHOW_PROCESSING_OUTPUT = True
SHOW_DETECT = True
SHOW_VIOLATION_COUNT = False
SHOW_TRACKING_ID = True

# -------------------- Data Recording --------------------
DATA_RECORD = True
DATA_RECORD_RATE = 5
ABNORMAL_ACTIVITY_FILE = "processed_data/abnormal_activity.csv"

# -------------------- Restricted Entry --------------------
RE_CHECK = False
RE_START_TIME = datetime.time(0, 0, 0)
RE_END_TIME = datetime.time(23, 0, 0)

# -------------------- Social Distance Detection --------------------
SD_CHECK = True
SOCIAL_DISTANCE = 20
GROUP_THRESHOLD = 5

# -------------------- Abnormal Activity Detection --------------------
ABNORMAL_CHECK = True
ABNORMAL_MIN_PEOPLE = 5
ABNORMAL_MAX_PEOPLE = 7
ABNORMAL_ENERGY = 800
ABNORMAL_THRESH = 0.3

# -------------------- Detection Thresholds --------------------
MIN_CONF = 0.35
NMS_THRESH = 0.45

# -------------------- Frame Resize --------------------
FRAME_SIZE = 1080

# -------------------- Tracking Settings --------------------
TRACK_MAX_AGE = 7