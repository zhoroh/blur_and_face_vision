from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
UPLOAD_VIDEO = 'Upload Video'

SOURCES_LIST = [IMAGE, VIDEO, UPLOAD_VIDEO]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'cars.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'people_1.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'people_car.mp4'
VIDEOS_DICT = {
    'cars': VIDEO_1_PATH,
    'people_1': VIDEO_2_PATH,
    'people_car': VIDEO_3_PATH,
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'


CLASS_NAMES= {'person': 0,
 'bicycle': 1,
 'car': 2,
 'motorcycle': 3,
 'airplane': 4,
 'bus': 5,
 'train': 6,
 'truck': 7,
 'boat': 8,
 'traffic light': 9,
 'fire hydrant': 10,
 'stop sign': 11,
 'parking meter': 12,
 'bench': 13,
 'bird': 14,
 'cat': 15,
 'dog': 16,
 'horse': 17,
 'sheep': 18,
 'cow': 19,
 'elephant': 20,
 'bear': 21,
 'zebra': 22,
 'giraffe': 23,
 'backpack': 24,
 'umbrella': 25,
 'handbag': 26,
 'tie': 27,
 'suitcase': 28,
 'frisbee': 29,
 'skis': 30,
 'snowboard': 31,
 'sports ball': 32,
 'kite': 33,
 'baseball bat': 34,
 'baseball glove': 35,
 'skateboard': 36,
 'surfboard': 37,
 'tennis racket': 38,
 'bottle': 39,
 'wine glass': 40,
 'cup': 41,
 'fork': 42,
 'knife': 43,
 'spoon': 44,
 'bowl': 45,
 'banana': 46,
 'apple': 47,
 'sandwich': 48,
 'orange': 49,
 'broccoli': 50,
 'carrot': 51,
 'hot dog': 52,
 'pizza': 53,
 'donut': 54,
 'cake': 55,
 'chair': 56,
 'couch': 57,
 'potted plant': 58,
 'bed': 59,
 'dining table': 60,
 'toilet': 61,
 'tv': 62,
 'laptop': 63,
 'mouse': 64,
 'remote': 65,
 'keyboard': 66,
 'cell phone': 67,
 'microwave': 68,
 'oven': 69,
 'toaster': 70,
 'sink': 71,
 'refrigerator': 72,
 'book': 73,
 'clock': 74,
 'vase': 75,
 'scissors': 76,
 'teddy bear': 77,
 'hair drier': 78,
 'toothbrush': 79}

