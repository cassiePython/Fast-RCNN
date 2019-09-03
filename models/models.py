from .model_factory import BaseModel
from networks.network_factory import NetworksFactory
from torch.autograd import Variable
from collections import OrderedDict
import torch
import os
from sklearn.externals import joblib
from loss.losses import Regloss_MSE
from loss.losses import Regloss_SmoothL1

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexModel(BaseModel):
    def __init__(self, opt, is_train):
        super(AlexModel, self).__init__(opt, is_train)
        self._name = 'AlexModel'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # use pre-trained AlexNet
        if self._is_train and not self._opt.load_epoch > 0:
            self._init_weights()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        if not self._is_train:
            self.set_eval()

        # init loss
        self._init_losses()

    def _init_create_networks(self):
        network_type = 'AlexNet'
        self.network = self._create_branch(network_type)
        if len(self._gpu_ids) > 1:
            self.network = torch.nn.DataParallel(self.network, device_ids=self._gpu_ids)
        if torch.cuda.is_available():
            self.network.cuda()

    def _init_train_vars(self):
        self._current_lr = self._opt.learning_rate
        self._decay_rate = self._opt.decay_rate
        # initialize optimizers
        self._optimizer = torch.optim.SGD(self.network.parameters(),
                                         lr=self._current_lr,
                                         momentum=self._decay_rate)

    def _init_weights(self):
        state_dict=load_state_dict_from_url(model_urls['alexnet'], progress=True)
        current_state = self.network.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('features'):
                    current_state[key] = state_dict[key]

    def load(self):
        load_epoch = self._opt.load_epoch
        self._load_network(self.network, 'AlexNet', load_epoch, self._opt.alex_dir)

    def save(self, label):
        # save networks
        self._save_network(self.network, 'AlexNet', label, self._opt.alex_dir)

    def _create_branch(self, branch_name):
        return NetworksFactory.get_by_name(branch_name, self._opt.classes_num)

    def _init_losses(self):
        # define loss function
        self._cross_entropy = torch.nn.CrossEntropyLoss()
        self._reg_loss = Regloss_SmoothL1()#Regloss_MSE()
            
        if torch.cuda.is_available():
            self._cross_entropy = self._cross_entropy.cuda()
            self._reg_loss = self._reg_loss.cuda()

    def set_input(self, inputs, bboxs, labels, verts):
        self.batch_inputs = self._Tensor(inputs)
        self.verts_inputs = self._FloatTensor(verts) #[ind_in_batch, x1,y1,x2,y2]
        self.labels = Variable(self._LongTensor(labels))
        self.bboxs = Variable(self._FloatTensor(bboxs))

    def optimize_parameters(self):
        if self._is_train:

            loss = self._forward()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _forward(self):
        cls, bbox = self.network(self.batch_inputs, self.verts_inputs)
        self._Reg_loss = self._reg_loss(bbox, self.bboxs)
        self._CE_loss = self._cross_entropy(cls, self.labels)
        
        self._total_loss = self._Reg_loss + self._CE_loss
        return self._total_loss

    def _forward_test(self, inputs, verts):
        img_input = self._Tensor(inputs)
        vert_input = self._FloatTensor(verts) #[x1,y1,x2,y2]
        #print (vert_input)
        #rois = torch.tensor([[0, 0, 0, 9, 9],[0, 0, 0, 9, 9],
        #                     [0, 0, 0, 9, 9],[0, 0, 0, 9, 9]], dtype=torch.float)
        cls, bbox = self.network(img_input, vert_input)
        return cls, bbox
              
    def get_current_errors(self):
        loss_dict = OrderedDict([('loss_entropy', self._CE_loss.data),
                                 ('loss_reg', self._Reg_loss.data),
                                 ('loss_total', self._total_loss.data)])
        return loss_dict

    def set_train(self):
        self.network.train()
        self._is_train = True

    def set_eval(self):
        self.network.eval()
        self._is_train = False

