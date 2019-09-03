from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
       
        self._parser.add_argument('--img_path', type=str, default='./sample_dataset/Images/000032.jpg', help='Test Image') 
        
        self.is_train = False
