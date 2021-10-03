import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    '''卷积 + relu + 卷积 用来预测光流的更新值'''
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))  

class ConvGRU(nn.Module): 
    '''卷积GRU'''
    def __init__(self, hidden_dim=128, input_dim=192+128):  # feature 192, context 128 flow 128
        super(ConvGRU, self).__init__()
        # 输入为上下文特征，flow, 最新的hidden state, 输出为update的hidden state, 光流的残差值
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)   
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        '''
        h: hidden state feature
        x: flow + correlation + context feature的叠加
        '''
        hx = torch.cat([h, x], dim=1) # 拼接hidden state feature, + correlation, context, flow

        z = torch.sigmoid(self.convz(hx)) # 门控卷积
        r = torch.sigmoid(self.convr(hx)) # 用来衡量前后hidden state的相关性
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1))) # 候选的hidden state

        h = (1-z) * h + z * q #根据权重来更新hidden state
        return h  # 返回hidden state

class SepConvGRU(nn.Module): 
    '''可分离卷积GRU  两个1x5 5x1 代替3✖3'''
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        # print('input_dim: ', input_dim)
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        '''
        h: hidden state feature
        x: flow + correlation + context feature的叠加
        '''
        # horizontal
        # print('using sep conv gru')
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2   # 相关性层级(2r+1)2
        # correlation特征使用两个卷积层处理
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0) 
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        # 两个卷积层用来处理光流特征
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        # 相关性特征
        cor = F.relu(self.convc1(corr))  
        cor = F.relu(self.convc2(cor))
        # 光流特征
        flo = F.relu(self.convf1(flow))  
        flo = F.relu(self.convf2(flo))
        # 叠加特征
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module): 
    '''小模型更新块'''
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()   
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)   
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128) 

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)  
        net = self.gru(net, inp)  # 返回hidden state
        delta_flow = self.flow_head(net) # 返回光流变化值  fk+1 = fk+ delta_fk+1

        return net, None, delta_flow # 返回gru单元， 光流残差值

class BasicUpdateBlock(nn.Module):
    '''基础模型更新块 采用可分离卷积GRU'''
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args) # 相关性+光流特征
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim) 
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)  
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



