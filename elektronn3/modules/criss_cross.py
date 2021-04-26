"""This code is copied from https://github.com/speedinghzl/CCNet, the code
associated with the "CCNet: Criss-Cross Attention for SemanticSegmentation" by Zilong Huang,
Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi,
Wenyu Liu and Thomas S. Huang (2018)
(https://arxiv.org/pdf/1811.11721.pdf).
The code is taken from the PurePython branch"""


def INF3D(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W * D, 1, 1)


class CrissCrossAttention3D(nn.Module):
    """ Criss-Cross Attention Module 3D version, inspired by the 2d version"""

    def __init__(self, in_dim, verbose=True):
        super(CrissCrossAttention3D, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=4)
        self.INF = INF3D
        self.gamma = nn.Parameter(torch.zeros(1))
        self.verbose = verbose

    def forward(self, x):
        m_batchsize, _, height, width, depth = x.size()
        proj_query = self.query_conv(x)
        # bchw > bwch, b*w*d-c-h > b*w*d-h-c
        proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth, -1,
                                                                           height).permute(0, 2, 1)
        # bchw > bhcw, b*h*d-c-w > b*h*d-w-c
        proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1,
                                                                           width).permute(0, 2, 1)
        # bchwd > bwch, b*h*w-c-d > b*h*w-d-c
        proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1,
                                                                           depth).permute(0, 2, 1)

        if self.verbose: print_tensor('q', proj_query)
        if self.verbose: print_tensor('qh', proj_query_H)
        if self.verbose: print_tensor('qw', proj_query_W)
        if self.verbose: print_tensor('qd', proj_query_D)

        proj_key = self.key_conv(x)

        # bchw > bwch, b*w*d-c-h
        proj_key_H = proj_key.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth, -1, height)
        # bchw > bhcw, b*h*d-c-w
        proj_key_W = proj_key.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1, width)
        proj_key_D = proj_key.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1, depth)

        if self.verbose: print_tensor('k', proj_key)
        if self.verbose: print_tensor('kh', proj_key_H)
        if self.verbose: print_tensor('kw', proj_key_W)
        if self.verbose: print_tensor('kd', proj_key_D)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth, -1, height)
        proj_value_W = proj_value.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1, width)
        proj_value_D = proj_value.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1, depth)

        # batch matrix-matrix
        inf_holder = self.INF(m_batchsize, height, width, depth)  # > bw-h-h
        if self.verbose: print_tensor('inf', inf_holder)
        energy_H = torch.bmm(proj_query_H, proj_key_H) + inf_holder  # bwd-h-c, bwd-c-h > bwd-h-h
        energy_H = energy_H.view(m_batchsize, width, depth, height, height).permute(0, 3, 1, 2, 4)  # bhwdh
        if self.verbose: print_tensor('eh', energy_H)

        #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, depth, width, width).permute(0, 1, 3,
                                                                                                              2, 4)  #
        if self.verbose: print_tensor('ew', energy_W)

        energy_D = torch.bmm(proj_query_D, proj_key_D).view(m_batchsize, height, width, depth, depth)
        if self.verbose: print_tensor('ew', energy_W)

        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 4))  # bhwd*(h+w+d)
        if self.verbose: print_tensor('eall', concate)
        # bhw(H+W) > bhwH, bwhH;
        att_H = concate[:, :, :, :, 0:height].permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * width * depth,
                                                                                       height, height)
        att_W = concate[:, :, :, :, height:height + width].permute(0, 1, 4, 2, 3).contiguous().view(
            m_batchsize * height * depth, width, width)
        att_D = concate[:, :, :, :, height + width:].contiguous().view(m_batchsize * height * width, depth, depth)

        if self.verbose: print_tensor('atth', att_H); print_tensor('attw', att_W);print_tensor('attd', att_D)

        # p-c-h, p-h-h > p-c-h
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, depth, -1, height).permute(0,3,4,1,2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, depth, -1, width).permute(0,3,1,4,2)
        out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize, height, width, -1, depth).permute(0,3,1,2,4)

        if self.verbose: print_tensor('outh', out_H); print_tensor('outw', out_W), print_tensor('outd', out_D)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W + out_D) + x



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out



class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, recurrence):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4

        self.conva = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout3d(0.1),
            nn.Conv3d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.recurrence = recurrence

    def forward(self, x, recurrence=self.recurrence):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output