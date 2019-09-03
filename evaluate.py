from __future__ import division
from models.model_factory import ModelsFactory
from options.test_options import TestOptions
from utils.util import image_proposal
from utils.util import show_rect
from utils.util import resize_image
import torch
import numpy as np
import cv2
import torch.nn.functional as F

class Test:
    def __init__(self):
        self._opt = TestOptions().parse()
        self._img_path = self._opt.img_path
        self._img_width = self._opt.image_width
        self._img_height = self._opt.image_height

        self._model = ModelsFactory.get_by_name('AlexModel', self._opt, is_train=False)
        self._classes = self._opt.classes.split(",")
        self._class_to_ind = dict(zip(range(1, len(self._classes) + 1), self._classes))
        self._class_to_ind[0] = 'None'

        self.test()

    def test(self):
        img = cv2.imread(self._img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_image(img, self._img_width, self._img_height)
        img = np.asarray(img, dtype='float32')
        img = np.transpose(img, [2, 0, 1])
        img = img.reshape((1,)+img.shape) #(1,3,1200,1600)
        
        vertices, rects = image_proposal(self._img_path) #rects: x,y,w,h
        show_rect(self._img_path, rects, ' ')

        verts = []
        for item in vertices:
            item = item[:4]
            for i in range(len(item)):
                item[i] = item[i] / 16.0
            verts.append([0.]+item)
        
        cls, bbox = self._model._forward_test(img, verts)
        #cls = F.softmax(cls,dim = 1)
        cls = torch.argmax(cls, dim=1).data.cpu().numpy()
        bbox = bbox.data.cpu().numpy()

        tmp = sorted(zip(rects,bbox,cls), key=lambda x:x[1][0], reverse=True)
        rects, bbox, cls = zip(*tmp)

        #print (self._class_to_ind)

        index = 0
        cls = self._class_to_ind[cls[0]]
        
        show_rect(self._img_path, [rects[index]], cls)

        px, py, pw, ph = rects[index]
        old_center_x, old_center_y = px + pw / 2.0, py + ph / 2.0

        x_ping, y_ping, w_suo, h_suo = bbox[index][1:]
        new_center_x = x_ping * pw + old_center_x
        new_center_y = y_ping * ph + old_center_y
        new_w = pw * np.exp(w_suo)
        new_h = ph * np.exp(h_suo)
        new_verts = [new_center_x, new_center_y, new_w, new_h]

        show_rect(self._img_path, [new_verts], cls)

        
if __name__ == "__main__":
    Test()
        

            
            
            
        
