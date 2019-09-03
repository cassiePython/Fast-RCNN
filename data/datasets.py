from data.dataset_factory import DatasetBase
from utils.util import resize_image
from utils.util import image_proposal
from utils.util import IOU
from utils.util import view_bar
from sklearn.externals import joblib
import codecs
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

class AlexnetDataset(DatasetBase):

    def __init__(self, opt):
        super(AlexnetDataset, self).__init__(opt)
        self._name = 'AlexnetDataset'

        self.datas = {}
        self.labels = {}
        self.boxes = {}
        self.imgs = []

        self._batch_size = self._opt.batch_size
        self._img_width = self._opt.image_width
        self._img_height = self._opt.image_height
        self._batch_size = self._opt.batch_size
        self._threshold = self._opt.Roi_threshold

        self._annotation_dir = self._opt.annotation_dir
        self._img_dir = self._opt.image_dir
        self._save_dir = self._opt.generate_save_dir

        self.imgs = []

        self._classes = self._opt.classes.split(",")
        self._class_to_ind = dict(zip(self._classes, range(1, len(self._classes) + 1)))

        self.cursor = 0
        
        # read dataset
        self._load_dataset()

    def _load_dataset(self):
        file_path = os.path.join(self._root, self._opt.train_list)
        with codecs.open(file_path, 'r', 'utf-8') as fr:
            lines = fr.readlines()
            for ind, line in enumerate(lines):
                context = line.strip()
                image_path = os.path.join(self._root, context)

                image_idx = context.split('/')[1]
                label, ground_truth = self._get_annotation_by_id(image_idx) #ground_truth: [x,y,w,h]
                
                self.imgs.append(image_idx)
                self.labels[image_idx] = label
                self.boxes[image_idx] = ground_truth

                save_path = os.path.join(self._save_dir, image_idx.split(".")[0] + "_roi_data.npy")

                if os.path.exists(save_path):
                    self._load_from_numpy(image_idx)
                    continue

                datas = []
                vertices, _ = image_proposal(image_path)
                for proposal_vertice in vertices:
                    iou_val = IOU(ground_truth, proposal_vertice) #ground_truth=[x1,y1,w,h]
                    if iou_val < self._threshold:
                        Roi_label = 0
                    else:
                        Roi_label = label

                    px = float(proposal_vertice[0]) + float(proposal_vertice[4] / 2.0)
                    py = float(proposal_vertice[1]) + float(proposal_vertice[5] / 2.0)
                    ph = float(proposal_vertice[5])
                    pw = float(proposal_vertice[4])

                    gx = float(ground_truth[0])
                    gy = float(ground_truth[1])
                    gw = float(ground_truth[2])
                    gh = float(ground_truth[3]) 

                    box_label = np.zeros(5)
                    box_label[1:5] = [(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)]
                    box_label[0] = Roi_label

                    datas.append([proposal_vertice, box_label]) #proposal_vertice=[x1,y1,x2,y2,w,h]

                self.datas[image_idx] = datas
                
                joblib.dump(datas, save_path)
                    #self.datas.append(data), 'roi_data.npy')
                view_bar("Process image of %s" % image_path, ind + 1, len(lines))                       

    def _get_annotation_by_id(self, image_idx):
        image_annotation_path = os.path.join(self._root, self._annotation_dir, image_idx.split(".")[0] + '.xml')
        tree = ET.parse(image_annotation_path)
        objs = tree.findall('object')
        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            class_name = obj.find('name').text.lower().strip()
            cls_ind = self._class_to_ind[obj.find('name').text.lower().strip()]
            box = [x1, y1, x2 - x1, y2 - y1]
            #print (cls_ind, box)
        return cls_ind, box

    def _load_from_numpy(self, image_idx):
        box_numpy_path = os.path.join(self._save_dir, image_idx.split(".")[0] + "_roi_data.npy")
        self.datas[image_idx] = joblib.load(box_numpy_path)
        
    def get_batch(self):
        images = np.zeros((self._batch_size, 3, self._img_height, self._img_width))
        labels = np.zeros((self._batch_size, 1))
        
        #R=128 n=2(batch_size)  128/2=64 Rois
        Rois = []
        Verts = []
        Roi_labels = []
        count = 0
        while( count < self._batch_size):
            image_idx = self.imgs[self.cursor]
            img_path = os.path.join(self._root, self._img_dir, image_idx)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_image(img, self._img_width, self._img_height)
            img = np.asarray(img, dtype='float32')
            img = np.transpose(img, [2, 0, 1])
            #print (img.shape)
            images[count] = img
            labels[count] = self.labels[image_idx]

            data = self.datas[image_idx]
            data = sorted(data, key=lambda x:x[1].tolist(), reverse=True)
            data = data[:64]

            boxes = []
            roi_labels = []
            verts = []
            for item in data:
                tmp = item[1].tolist()
                Roi_labels.append(tmp[0])
                if tmp[0] > 0:
                    tmp[0] = 1
                boxes.append(tmp)
                vert = item[0][:4]
                for i in range(len(vert)):
                    vert[i] = (vert[i] / 16.0)
                vert = [count] + vert
                verts.append(vert)
                        
            #boxes = [item[1].tolist() for item in data]
            Rois += boxes
            Roi_labels += roi_labels

            #verts = [[count]+item[0][:4]/16.0 for item in data] #[ind_in_batch, x1,y1,x2,y2]
            
            Verts += verts

            count += 1
            self.cursor += 1
            if self.cursor >= len(self.imgs):
                self.cursor = 0
                np.random.shuffle(self.imgs)

            #print(img_path, vert)

        Rois = np.array(Rois)
        Verts = np.array(Verts)
        Roi_labels = np.array(Roi_labels)
        return images, labels, Rois, Roi_labels, Verts
  
    def __len__(self):
        return len(self.imgs)
