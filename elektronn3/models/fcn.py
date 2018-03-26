import torch.nn as nn
import torch.nn.functional as F
# from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/fcn.py
# FCN32s
#In every layer few steps have been commented because of Memory constraints (please uncomment them acc to the resources)

class fcn32s(nn.Module):

    def __init__(self, n_classes=2, learned_billinear=False, red_fac=16):
        super(fcn32s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 64 // red_fac, 3, padding=100),
            nn.ReLU(inplace=True),
            # nn.Conv3d(64 // red_fac, 64 // red_fac, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64 // red_fac, 128 // red_fac, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(128 // red_fac, 128 // red_fac, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv3d(128 // red_fac, 256 // red_fac, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(256 // red_fac, 256 // red_fac, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(256 // red_fac, 256 // red_fac, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv3d(256 // red_fac, 512 // red_fac, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(512 // red_fac, 512 // red_fac, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(512 // red_fac, 512 // red_fac, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv3d(512 // red_fac, 512 // red_fac, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(512 // red_fac, 512 // red_fac, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(512 // red_fac, 512 // red_fac, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv3d(512 // red_fac, 4096 // red_fac, 7),
            nn.ReLU(inplace=True),
            nn.Dropout3d(),
            # nn.Conv3d(4096 // red_fac, 4096 // red_fac, 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout3d(),
            nn.Conv3d(4096 // red_fac, self.n_classes, 1),)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose3d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None


    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)

        score = self.classifier(out)

        out = F.upsample(score, x.size()[2:], mode='trilinear')

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv3d) and isinstance(l2, nn.Conv3d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            # print type(l1), dir(l1),
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

class fcn16s(nn.Module):

    def __init__(self, n_classes=2, learned_billinear=False):
        super(fcn16s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            # nn.Conv3d(64, 64, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(128, 128, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(256, 256, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(256, 256, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv3d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv3d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout3d(),
            # nn.Conv3d(4096, 4096, 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout3d(),
            nn.Conv3d(4096, self.n_classes, 1),)

        self.score_pool4 = nn.Conv3d(512, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose3d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        score = F.upsample(score, score_pool4.size()[2:], mode='trilinear')
        score += score_pool4
        out = F.upsample(score, x.size()[2:], mode='trilinear')

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv3d) and isinstance(l2, nn.Conv3d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

# FCN 8s
class fcn8s(nn.Module):

    def __init__(self, n_classes=2, learned_billinear=False):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            # nn.Conv3d(64, 64, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(128, 128, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(256, 256, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(256, 256, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv3d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv3d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout3d(),
            # nn.Conv3d(4096, 4096, 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout3d(),
            nn.Conv3d(4096, self.n_classes, 1),)

        self.score_pool4 = nn.Conv3d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv3d(256, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose3d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        score = F.upsample(score, score_pool4.size()[2:], mode='trilinear')
        score += score_pool4
        score = F.upsample(score, score_pool3.size()[2:], mode='trilinear')
        score += score_pool3
        out = F.upsample(score, x.size()[2:], mode='trilinear')

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv3d) and isinstance(l2, nn.Conv3d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]