import torch.nn as nn
from .network_factory import NetworkBase
from sklearn import svm
import torch
import torchvision.ops as ops

class AlexNet(NetworkBase):

    def __init__(self, classes_num):
        super(AlexNet, self).__init__()
        self._name = 'AlexNet'
        self._classes_num = classes_num

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.roi_pool = ops.RoIPool(output_size=(6, 6), spatial_scale=1)

        self.fcs = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.cls = nn.Linear(4096, self._classes_num+1)
        self.bbox = nn.Linear(4096, 5)
       
            
    def forward(self, inp, verts):
        x = self.features(inp)
       
        #verts = torch.tensor(verts, dtype=torch.float) #[ind_in_batch, x1,y1,x2,y2]
        
        x = self.roi_pool(x, verts)
        x = x.view(x.size(0), -1)

        x = self.fcs(x)

        cls = self.cls(x)
        bbox = self.bbox(x)

        return cls, bbox
        

  
        
    
