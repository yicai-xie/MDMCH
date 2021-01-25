import torch
from torch import nn
from models.basic_module import BasicModule
from config import opt

from torch.nn import functional as F

class ImgModule(BasicModule):
    def __init__(self, bit, pretrain_model=None):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.features1_3 = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1

            nn.ReLU(inplace=True),
            # 2 norm1
            #nn.BatchNorm2d(64, affine=True),
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            # 4
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 5 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 6 relu2

            nn.ReLU(inplace=True),
            # 7 norm2
            #nn.BatchNorm2d(256, affine=True),
            nn.LocalResponseNorm(size=2, k=2),
            # 8 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 9 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 10 relu3

            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(256, affine=True),
        )
        self.features4 = nn.Sequential(
            # 11 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 12 relu4

            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(256, affine=True),
        )
        self.features5 = nn.Sequential(
            # 13 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 14 relu5

            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(256, affine=True),
            # 15 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.features6_7 = nn.Sequential(
            # 16 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 17 relu6

            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(4096, affine=True),
            # 18 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 19 relu7

            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(4096, affine=True),

        )

        self.BatchNorm2d_256 = nn.BatchNorm2d(256, affine=True)
        self.BatchNorm2d_4096 = nn.BatchNorm2d(4096, affine=True)

        # fc8
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self.classifier.weight.data = torch.randn(bit, 4096) * 0.01
        self.classifier.bias.data = torch.randn(bit) * 0.01
        self.mean = torch.zeros(3, 224, 224)
        if pretrain_model:
            self._init(pretrain_model)
            #self._init()

    def _init(self, data):
        weights = data['layers'][0]
        # 加载预训练好的网络imagenet-vgg-f.mat 到这个分成了多个层组的新网络中
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        k_temp = 0
        for k, v in self.features1_3.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))
        k_temp = k + 1

        for k, v in self.features4.named_children():

            k = int(k) + k_temp
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))
        k_temp = k + 1

        for k, v in self.features5.named_children():
            k = int(k) + k_temp
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))
        k_temp = k + 1
        for k, v in self.features6_7.named_children():
            k = int(k) + k_temp
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))



    """
    def _init(self):
        VGG_F_items = self.state_dict().items()
        # 加载预训练好的网络image_model_CNN-F_pretrained.pth, 由imagenet-vgg-f.mat转存得来的
        CNN_F_pretrained = torch.load('./checkpoints/image_model_CNN-F_pretrained.pth')
        pretrain_model = {}
        j = 0
        
        for k, v in self.state_dict().__iter__():
            v = CNN_F_pretrained[j][1]
            k = VGG_F_items[j][0]
            pretrain_model[k] = v
            j += 1

        print('load the weight form CNN_F_pretrained')
        model_dict = self.state_dict()
        # 1. 把不属于新(本) 网络 (self) 所需要的层剔除
        pretrain_dict = {k: v for k, v in pretrain_model.items() if k in model_dict}
        # 2. 把参数存入以及存在的model_dict
        model_dict.update(pretrain_dict)
        # 3. 加载更新后的model_dict
        self.load_state_dict(model_dict)
        print('copy the weight from pretrained CNN_F_pretrained sucessfully')


    
    def _init(self, data):
        weights = data['layers'][0]
        # 加载预训练好的网络imagenet-vgg-f.mat
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

        #BasicModule.save(self, name=None)
    """

    def forward(self, x):
        if x.is_cuda:
            x = x - self.mean.cuda()
        else:
            x = x - self.mean
        """
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        return x
        """
        x1_3 = self.features1_3(x)
        #x1_3 = self.BatchNorm2d_256(x1_3)
        x4 =self.features4(x1_3)
        #x4 = self.BatchNorm2d_256(x4)
        x5 = self.features5(x4)
        #x5 = self.BatchNorm2d_256(x5)
        x = self.features6_7(x5)
        #x = self.BatchNorm2d_4096(x)
        x = x.squeeze()
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    """
    path = '/data5/guwen/data/imagenet-vgg-f.mat'
    import scipy.io as scio
    data = scio.loadmat(path)
    print(data['normalization'][0][0])
    """
    #pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    model = ImgModule(opt.bit)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters)


