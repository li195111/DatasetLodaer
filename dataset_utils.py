import os
import sys
import json
import traceback
from collections import OrderedDict

import cv2
import numpy as np 

class Config(object):
    
    def __init__(self):
        super(Config, self).__init__()
        
    @classmethod
    def from_json(cls, filepath):
        cfg = Config()
        with open(filepath, 'r', encoding= 'utf-8') as f:
            config = json.loads(f.read())
        for NAME in config:
            cfg.__dict__[NAME] = config[NAME]
        return cfg

class DatasetLoader:
    def __init__(self, dataset_path, dataset_path_list=None):
        self.DATASET_PATH = dataset_path
        self.DATASET_PATH_LIST = dataset_path_list
        self.NAME_DICT = OrderedDict()
        self.ALL_FILE_NAMES = None
        self.NUM_IMAGES = 0

    def create_generator(self, batch_size, dsize= (640,360), shuffle= True):
        num_iters = self.NUM_IMAGES // batch_size
        if shuffle:
            np.random.shuffle(self.IMAGE_NAMES)
        for i in range(num_iters):
            batch_name = self.IMAGE_NAMES[i*batch_size:(i+1)*batch_size]
            batch_images = []
            batch_points = []
            for name in batch_name:
                img = self.read_image_file(self.NAME_DICT[name]["image"])
                img_shape = img.shape
                if len(img_shape) == 3:
                    H, W, C = img_shape
                else:
                    H, W = img_shape
                pts = self.read_points_file(self.NAME_DICT[name]["points"])
                pts[:,0] *= dsize[0] / W
                pts[:,1] *= dsize[1] / H
                img = cv2.resize(img, dsize, interpolation= cv2.INTER_LANCZOS4)
                batch_images.append(img)
                batch_points.append(pts)
            yield [np.array(batch_images), np.array(batch_points)]
        
    def read_points_file(self, pts_path):
        with open(pts_path, 'r', encoding= 'utf-8') as f:
            points_string = f.read().split('\n')[3:-1]
            if '}' in points_string:
                points_string.remove('}')
            points = [tuple([float(value_string) for value_string in point_string.split()]) for point_string in points_string]
        points = np.array(points, dtype= np.float)
        return points
    
    def read_image_file(self, img_path):
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            error_class = e.__class__.__name__  #取得錯誤類型
            detail = e.args[0]                  #取得詳細內容
            cl, exc, tb = sys.exc_info()        #取得Call Stack
            details = '\n'.join([f"File \"{s[0]}\", line {s[1]} in {s[2]}" for s in traceback.extract_tb(tb)])
            errMsg = f"\n[{error_class}] {detail}"
            print (details, errMsg)
        return img
class DatasetLoader_68(DatasetLoader):
    def __init__(self, dataset_path_list):
        super(DatasetLoader_68, self).__init__(None, dataset_path_list)
        for path in self.DATASET_PATH_LIST:
            if "300W" in path:
                dataloader = DatasetLoader_300W_68(path)
            if "AFW" in path:
                dataloader = DatasetLoader_AFW_68(path)
            if "Helen" in path:
                dataloader = DatasetLoader_Helen_68(path)
            if "LFPW" in path:
                dataloader = DatasetLoader_LFPW_68(path)
            for NAME in dataloader.NAME_DICT:
                self.NAME_DICT[NAME] = dataloader.NAME_DICT[NAME]
        self.IMAGE_NAMES = np.array(list(self.NAME_DICT.keys()))
        self.NUM_IMAGES = len(self.IMAGE_NAMES)

class DatasetLoader_LFPW_68(DatasetLoader):
    def __init__(self, dataset_path):
        super(DatasetLoader_LFPW_68, self).__init__(dataset_path)
        self.TRAIN_DATASET_PATH = os.path.join(self.DATASET_PATH, "Training")
        self.TEST_DATASET_PATH = os.path.join(self.DATASET_PATH,"Testing")
        self.TRAIN_FILE_NAMES = os.listdir(self.TRAIN_DATASET_PATH)
        self.TEST_FILE_NAMES = os.listdir(self.TEST_DATASET_PATH)
        self.ALL_FILE_NAMES = self.TRAIN_FILE_NAMES + self.TEST_FILE_NAMES
        for f in self.ALL_FILE_NAMES:
            NAME = f.split('.')[0]
            try:
                self.NAME_DICT[NAME]
            except KeyError:
                self.NAME_DICT[NAME] = {}
            PATH = self.TRAIN_DATASET_PATH if f in self.TRAIN_FILE_NAMES else self.TEST_DATASET_PATH
            if f.endswith('png'):
                TYPE = "image"
            if f.endswith('pts'):
                TYPE = "points" 
            self.NAME_DICT[NAME][TYPE] = os.path.join(PATH, f)
        self.IMAGE_NAMES = np.array(list(self.NAME_DICT.keys()))
        self.NUM_IMAGES = len(self.IMAGE_NAMES)

class DatasetLoader_Helen_68(DatasetLoader):
    def __init__(self, dataset_path):
        super(DatasetLoader_Helen_68, self).__init__(dataset_path)
        self.TRAIN_DATASET_PATH = os.path.join(self.DATASET_PATH, "Training")
        self.TEST_DATASET_PATH = os.path.join(self.DATASET_PATH,"Testing")
        self.TRAIN_FILE_NAMES = os.listdir(self.TRAIN_DATASET_PATH)
        self.TEST_FILE_NAMES = os.listdir(self.TEST_DATASET_PATH)
        self.ALL_FILE_NAMES = self.TRAIN_FILE_NAMES + self.TEST_FILE_NAMES
        for f in self.ALL_FILE_NAMES:
            NAME = f.split('.')[0]
            try:
                self.NAME_DICT[NAME]
            except KeyError:
                self.NAME_DICT[NAME] = {}
            PATH = self.TRAIN_DATASET_PATH if f in self.TRAIN_FILE_NAMES else self.TEST_DATASET_PATH
            if f.endswith('jpg'):
                TYPE = "image"
            if f.endswith('pts'):
                TYPE = "points" 
            self.NAME_DICT[NAME][TYPE] = os.path.join(PATH, f)
        self.IMAGE_NAMES = np.array(list(self.NAME_DICT.keys()))
        self.NUM_IMAGES = len(self.IMAGE_NAMES)

class DatasetLoader_AFW_68(DatasetLoader):
    def __init__(self, dataset_path):
        super(DatasetLoader_AFW_68, self).__init__(dataset_path)
        self.ALL_FILE_NAMES = os.listdir(self.DATASET_PATH)
        for f in self.ALL_FILE_NAMES:
            NAME = f.split('.')[0]
            try:
                self.NAME_DICT[NAME]
            except KeyError:
                self.NAME_DICT[NAME] = {}
            PATH = self.DATASET_PATH
            if f.endswith('jpg'):
                TYPE = "image"
            if f.endswith('pts'):
                TYPE = "points" 
            self.NAME_DICT[NAME][TYPE] = os.path.join(PATH, f)
        self.IMAGE_NAMES = np.array(list(self.NAME_DICT.keys()))
        self.NUM_IMAGES = len(self.IMAGE_NAMES)
        
class DatasetLoader_300W_68(DatasetLoader):
    '''
    Images: shape (68, H, W)
    Points: shape (68, 2) 
    '''
    def __init__(self, dataset_path):
        super(DatasetLoader_300W_68, self).__init__(dataset_path)
        self.INDOOR_NAME = "01_Indoor"
        self.OUTDOOR_NAME = "02_Outdoor"
        self.INDOOR_PATH = os.path.join(self.DATASET_PATH, self.INDOOR_NAME)
        self.OUTDOOR_PATH = os.path.join(self.DATASET_PATH, self.OUTDOOR_NAME)
        self.INDOOR_FILE_NAMES = os.listdir(self.INDOOR_PATH)
        self.OUTDOOR_FILE_NAMES = os.listdir(self.OUTDOOR_PATH)
        self.ALL_FILE_NAMES = self.INDOOR_FILE_NAMES + self.OUTDOOR_FILE_NAMES
        for f in self.ALL_FILE_NAMES:
            NAME = f.split('.')[0]
            try:
                self.NAME_DICT[NAME]
            except KeyError:
                self.NAME_DICT[NAME] = {}
            PATH = self.INDOOR_PATH if f in self.INDOOR_FILE_NAMES else self.OUTDOOR_PATH
            if f.endswith('png'):
                TYPE = "image"
            if f.endswith('pts'):
                TYPE = "points" 
            self.NAME_DICT[NAME][TYPE] = os.path.join(PATH, f)
        self.IMAGE_NAMES = np.array(list(self.NAME_DICT.keys()))
        self.NUM_IMAGES = len(self.IMAGE_NAMES)

class DatasetLoader_CelebA(DatasetLoader):
    '''
    - In-The-Wild Images (Img/img_celeba.7z)
    202,599 original web face images. See In-The-Wild Images section below for more info.

    - Align&Cropped Images (Img/img_align_celeba.zip & Img/img_align_celeba_png.7z)
        202,599 align&cropped face images. See Align&Cropped Images section below for more info.
    原始相片
    - Bounding Box Annotations (Anno/list_bbox_celeba.txt)
        bounding box labels. See BBOX LABELS section below for more info.
    錨盒標記
    - Landmarks Annotations (Anno/list_landmarks_celeba.txt & Anno/list_landmarks_align_celeba.txt)
        5 landmark location labels. See LANDMARK LABELS section below for more info.
    特徵點標記
    - Attributes Annotations (Anno/list_attr_celeba.txt)
        40 binary attribute labels. See ATTRIBUTE LABELS section below for more info.
    屬性標記
    - Identity Annotations (available upon request)
        10,177 identity labels. See IDENTITY LABELS section below for more info.
    身分標記
    - Evaluation Partitions (Eval/list_eval_partition.txt)
        image ids for training, validation and testing set respectively. See EVALUATION PARTITIONS section below for more info.
    '''
    def __init__(self, dataset_path):
        super(DatasetLoader_CelebA, self).__init__(dataset_path)
        self.ANNO_PATH = os.path.join(self.DATASET_PATH,"Anno")
        self.ANNO_IDENTITY_FILE_PATH = os.path.join(self.ANNO_PATH,"identity_CelebA.txt")
        self.ANNO_LIST_ATTR_FILE_PATH = os.path.join(self.ANNO_PATH,"list_attr_celeba.txt")
        self.ANNO_LIST_BBOX_FILE_PATH = os.path.join(self.ANNO_PATH,"list_bbox_celeba.txt")
        self.ANNO_LIST_LANDMARK_ALIGN_FILE_PATH = os.path.join(self.ANNO_PATH,"list_landmarks_align_celeba.txt")
        self.ANNO_LIST_LANDMARK_FILE_PATH = os.path.join(self.ANNO_PATH,"list_landmarks_celeba.txt")
        self.EVAL_PATH = os.path.join(self.DATASET_PATH,"Eval")
        self.IMGS_PATH = os.path.join(self.DATASET_PATH,"Img")
        self.IMG_ALIGN_CELEBA_PATH = os.path.join(self.IMGS_PATH,"img_align_celeba")
        self.IMG_CELEBA_PATH = os.path.join(self.IMGS_PATH,"img_celeba")
        self.ALL_FILE_NAMES = os.listdir(self.IMG_CELEBA_PATH)
        self.ANNO_LIST = {'BBOX':self.ANNO_LIST_BBOX_FILE_PATH,
                          'ID':self.ANNO_IDENTITY_FILE_PATH,
                          'ATTRS':self.ANNO_LIST_ATTR_FILE_PATH,
                          'LANDMARKS':self.ANNO_LIST_LANDMARK_FILE_PATH,
                          'LANDMARKS_ALIGN':self.ANNO_LIST_LANDMARK_ALIGN_FILE_PATH}
        for ANNO in self.ANNO_LIST:
            self.read_file(self.ANNO_LIST[ANNO], ANNO)
        self.IMAGE_NAMES = np.array(list(self.NAME_DICT.keys()))
        self.NUM_IMAGES = len(self.ALL_FILE_NAMES)
            
    def create_generator(self, batch_size, dsize= (640,360), shuffle= True):
        num_iters = self.NUM_IMAGES // batch_size
        if shuffle:
            np.random.shuffle(self.IMAGE_NAMES)
        for i in range(num_iters):
            batch_name = self.IMAGE_NAMES[i*batch_size:(i+1)*batch_size]
            batch_images = []
            batch_bboxes = []
            batch_attrs = []
            batch_pts = []
            batch_align_pts = []
            for name in batch_name:
                IMG = self.read_image_file(self.NAME_DICT[name]["image"])
                H, W, C = IMG.shape if len(IMG.shape) == 3 else list(IMG.shape) + [1]
                ID = self.NAME_DICT[name]["ID"]
                BBOX = self.NAME_DICT[name]["BBOX"]
                BBOX[0] *= dsize[0] / W
                BBOX[1] *= dsize[1] / H
                BBOX[2] *= dsize[0] / W
                BBOX[3] *= dsize[1] / H
                ATTRS = self.NAME_DICT[name]["ATTRS"]
                PTS = self.NAME_DICT[name]["LANDMARKS"]
                PTS[:,0] *= dsize[0] / W
                PTS[:,1] *= dsize[1] / H
                ALIGN_PTS = self.NAME_DICT[name]["LANDMARKS_ALIGN"]
                ALIGN_PTS[:,0] *= dsize[0] / W
                ALIGN_PTS[:,1] *= dsize[1] / H
                IMG = cv2.resize(IMG, dsize, interpolation= cv2.INTER_LANCZOS4)
                batch_images.append(IMG)
                batch_bboxes.append(BBOX)
                batch_attrs.append(ATTRS)
                batch_pts.append(PTS)
                batch_align_pts.append(ALIGN_PTS)
            yield [np.array(batch_images), np.array(batch_bboxes), np.array(batch_attrs), np.array(batch_pts), np.array(batch_align_pts)]
        
    def read_file(self, filepath, annotation):
        with open(filepath, 'r', encoding= 'utf-8') as f:
            if filepath == self.ANNO_IDENTITY_FILE_PATH:
                start = 0
            else:
                start = 2
            datas = f.read().split('\n')[start:-1]
            for data in datas:
                name = data.split()[0]
                try:
                    self.NAME_DICT[name]
                except KeyError:
                    self.NAME_DICT[name] = {}
                details = data.split()[1:]
                details = np.array([float(v) for v in details])
                if annotation == "LANDMARKS" or annotation == "LANDMARKS_ALIGN":
                    details = details.reshape([-1,2])
                self.NAME_DICT[name]["image"] = os.path.join(self.IMG_CELEBA_PATH,name)
                self.NAME_DICT[name]["align_image"] = os.path.join(self.IMG_ALIGN_CELEBA_PATH,name)
                self.NAME_DICT[name][annotation] = details
                
class DatasetLoader_WIDER(DatasetLoader):
    '''
    Attached the mappings between attribute names and label values.
    blur:
        clear->0
        normal blur->1
        heavy blur->2
    expression:
        typical expression->0
        exaggerate expression->1
    illumination:
        normal illumination->0
        extreme illumination->1
    occlusion:
        no occlusion->0
        partial occlusion->1
        heavy occlusion->2
    pose:
        typical pose->0
        atypical pose->1
    invalid:
        false->0(valid image)
        true->1(invalid image)
    The format of txt ground truth.
    File name
    Number of bounding box
    x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
    '''
    def __init__(self, dataset_path, mode):
        super(DatasetLoader_WIDER, self).__init__(dataset_path)
        self.MODE = mode
        self.TRAIN_IMAGE_DIR_PATH = os.path.join(self.DATASET_PATH,"WIDER_train","images")
        self.TEST_IMAGE_PATH = os.path.join(self.DATASET_PATH,"WIDER_test","images")
        self.VALID_IMAGE_PATH = os.path.join(self.DATASET_PATH,"WIDER_val","images")
        self.WIDER_FACE_SPLIT_PATH = os.path.join(self.DATASET_PATH,"wider_face_split")
        self.WIDER_FACE_TRAIN_BBOX_GT_FILE_PATH = os.path.join(self.WIDER_FACE_SPLIT_PATH,"wider_face_train_bbx_gt.txt")
        self.WIDER_FACE_TEST_FILE_LIST_FILE_PATH = os.path.join(self.WIDER_FACE_SPLIT_PATH,"wider_face_test_filelist.txt")
        self.WIDER_FACE_VALID_BBOX_GT_FILE_PATH = os.path.join(self.WIDER_FACE_SPLIT_PATH,"wider_face_val_bbx_gt.txt")
        self.WIDER_FACE_IMAGE_LIST = {'TRAIN':self.TRAIN_IMAGE_DIR_PATH,
                                      'TEST':self.TEST_IMAGE_PATH,
                                      'VALID':self.VALID_IMAGE_PATH}
        self.WIDER_FACE_GT_LIST = {'TRAIN':self.WIDER_FACE_TRAIN_BBOX_GT_FILE_PATH,
                                   'TEST':self.WIDER_FACE_TEST_FILE_LIST_FILE_PATH,
                                   'VALID':self.WIDER_FACE_VALID_BBOX_GT_FILE_PATH}
        
        self.read_file(self.MODE)
        
        self.IMAGE_NAMES = np.array(list(self.NAME_DICT[self.MODE].keys()))
        self.NUM_IMAGES = len(self.IMAGE_NAMES)
        
    def create_generator(self, batch_size, dsize= (640,360), shuffle= True, attrs= False):
        num_iters = self.NUM_IMAGES // batch_size
        if shuffle:
            np.random.shuffle(self.IMAGE_NAMES)
        for i in range(num_iters):
            batch_name = self.IMAGE_NAMES[i*batch_size:(i+1)*batch_size]
            batch_images = []
            batch_bboxes = []
            for name in batch_name:
                img = self.read_image_file(self.NAME_DICT[self.MODE][name]["image"])
                H, W, C = img.shape if len(img.shape) == 3 else list(img.shape) + [1]
                BBOX = self.NAME_DICT[self.MODE][name]["BBOX"]
                if not attrs:
                    BBOX = BBOX[:,:4]
                BBOX[:,0] *= dsize[0] / W
                BBOX[:,1] *= dsize[1] / H
                BBOX[:,2] *= dsize[0] / W
                BBOX[:,3] *= dsize[1] / H
                img = cv2.resize(img, dsize, interpolation= cv2.INTER_LANCZOS4)
                batch_images.append(img)
                batch_bboxes.append(bboxes)
            yield [np.array(batch_images), np.array(batch_bboxes)]
        
    def read_file(self, mode= 'TRAIN'):
        self.NAME_DICT[mode] = {}
        if mode == 'TEST':
            with open(self.WIDER_FACE_GT_LIST[mode],'r',encoding='utf-8') as f:
                datas = f.read().split('\n')[:-1]
                for name in datas:
                    dirname, filename = name.split('/')
                    image_dirname = self.WIDER_FACE_IMAGE_LIST[mode]
                    image_path = os.path.join(image_dirname,dirname,filename)
                    self.NAME_DICT[mode][filename] = {"image":image_path}
        else:
            with open(self.WIDER_FACE_GT_LIST[mode], 'r', encoding= 'utf-8') as f:
                datas = f.read().split('\n')[:-1]
                i = 0
                while i < len(datas)-1:
                    filename_id = i
                    num_bboxes_id = i + 1
                    last_bbox_id = num_bboxes_id + int(datas[num_bboxes_id]) if int(datas[num_bboxes_id]) != 0 else num_bboxes_id + 1
                    bboxes = []
                    for j in range(int(datas[num_bboxes_id])):
                        bbox_id = num_bboxes_id + (j+1)
                        bboxes.append([float(v) for v in datas[bbox_id].split()])
                    dirname, filename = datas[filename_id].split('/')
                    image_dirname = self.WIDER_FACE_IMAGE_LIST[mode]
                    image_path = os.path.join(image_dirname,dirname,filename)
                    self.NAME_DICT[mode][filename] = {"image":image_path,"BBOX":np.array(bboxes)}
                    i = last_bbox_id + 1

class LocalDatasetLoader:
    def __init__(self):
        self.NAME_DICT = OrderedDict()
        self.ALL_FILE_NAMES = None
        self.IMAGE_NAMES = []
        self.NUM_IMAGES = 0

    def create_generator(self, batch_size, dsize= (640,360), shuffle= True):
        num_iters = self.NUM_IMAGES // batch_size
        if shuffle:
            np.random.shuffle(self.IMAGE_NAMES)
        for i in range(num_iters):
            batch_name = self.IMAGE_NAMES[i*batch_size:(i+1)*batch_size]
            batch_images = []
            batch_classes = []
            for PATH in batch_name:
                CLASS_NAME = os.path.basename(os.path.dirname(PATH))
                CLASS_ID = self.CLASS_NAMES[CLASS_NAME]
                img = self.read_image_file(PATH)
                img_shape = img.shape
                H, W, C = img_shape
                img = cv2.resize(img, dsize, interpolation= cv2.INTER_LANCZOS4)
                batch_images.append(img)
                batch_classes.append(np.int64(CLASS_ID))
            yield [np.array(batch_images), np.array(batch_classes)]
           
    def read_image_file(self, img_path):
        img = cv2.imread(img_path)
        TYPE = "GRAY" if len(img.shape) == 2 else "BGR"
        if TYPE == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

class LocalDataLoader(LocalDatasetLoader):
    def __init__(self, config, classes):
        super(LocalDataLoader, self).__init__()
        self.cfg = config
        self.KNOWN_DATASET = self.cfg.KNOWN
        self.UNKNOW_DATASET = self.cfg.UNKNOWN
        self.CLASS_NAMES = classes
        self.N_CLASSES = len(self.CLASS_NAMES)
        for CLASS_NAME in self.CLASS_NAMES:
            if CLASS_NAME != "unknown":
                CLASS_PATH = os.path.join(self.KNOWN_DATASET,CLASS_NAME)
                NAMES = os.listdir(CLASS_PATH)
                PATHS = [os.path.join(CLASS_PATH, NAME) for NAME in NAMES]
                self.NAME_DICT[CLASS_NAME] = PATHS
        self.UNKNOW_NAMES = os.listdir(self.UNKNOW_DATASET)
        self.NAME_DICT["unknown"] = [os.path.join(self.UNKNOW_DATASET,NAME) for NAME in self.UNKNOW_NAMES]
        for NAME in self.NAME_DICT:
            NAME_PATHS = self.NAME_DICT[NAME]
            self.IMAGE_NAMES += NAME_PATHS
        self.NUM_IMAGES = len(self.IMAGE_NAMES)

if __name__ == "__main__":
    CONFIG_NAME = "config.json"
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),CONFIG_NAME)
    
    cfg = Config.from_json(CONFIG_PATH)

    dataloader = DatasetLoader_CelebA(cfg.CELEBA_DATASET_PATH)
    loader = dataloader.create_generator(batch_size= 32, dsize= (640,360), shuffle= False)
    
    images, bboxes, attrs, pts, align_pts = loader.__next__()
    print (images.shape, bboxes.shape, attrs.shape, pts.shape, align_pts.shape)
