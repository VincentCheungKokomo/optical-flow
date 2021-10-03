import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor import BasicEncoder, BasicEncoderQuarter
from .update import BasicUpdateBlock, BasicUpdateBlockQuarter
from .utils.utils import bilinear_sampler, coords_grid, coords_grid_y_first,\
    upflow4, compute_interpolation_weights
from .knn import knn_faiss_raw

autocast = torch.cuda.amp.autocast


def compute_sparse_corr(fmap1, fmap2, k=32):
    """
    计算k个相关性
    input  1 256 64 120
    Compute a cost volume containing the k-largest hypotheses for each pixel.
    Output: corr_mink
    """

    B, C, H1, W1 = fmap1.shape # 1 256 64 120
    H2, W2 = fmap2.shape[2:]  # 64 120
    N = H1 * W1 # 7680

    fmap1, fmap2 = fmap1.view(B, C, -1), fmap2.view(B, C, -1) # img2col
    # print('fmap1.shape: ', fmap1.shape) # 1, 256, 7680
    # print('fmap2.shape: ', fmap2.shape) # 1, 256, 7680
    
    with torch.no_grad():
        _, indices = knn_faiss_raw(fmap1, fmap2, k)  # [B, k, H1*W1]  1, 8, 7680 获取8个最相似向量的索引
        # print('indices.shape : ' , indices.shape) 
        indices_coord = indices.unsqueeze(1).expand(-1, 2, -1, -1)  # [B, 2, k, H1*W1] 1, 2, 8, 7680 构建索引对应的坐标
        # print(indices_coord[0][:][0][0])
        # print('indices_coord.shape : ' , indices_coord.shape) 
        coords0 = coords_grid_y_first(B, H2, W2).view(B, 2, 1, -1).expand(-1, -1, k, -1).to(fmap1.device)  # [B, 2, k, H1*W1] 
        coords1 = coords0.gather(3, indices_coord)  # [B, 2, k, H1*W1]
        coords1 = coords1 - coords0
        # print(coords1.shape)  # 1, 2, 8, 7680

        # Append batch index 添加索引
        batch_index = torch.arange(B).view(B, 1, 1, 1).expand(-1, -1, k, N).type_as(coords1) # 1, 1, 8, 7680 
        # print('batch_index.shape :', batch_index.shape)

    # Gather by indices from map2 and compute correlation volume 从map2收集k个最大相关性对应向量，并计算对应相似性
    fmap2 = fmap2.gather(2, indices.view(B, 1, -1).expand(-1, C, -1)).view(B, C, k, N)  # 1, 256, 8, 7680
    # print('fmap2 gathered: ', fmap2.shape)
    corr_sp = torch.einsum('bcn,bckn->bkn', fmap1, fmap2).contiguous() / torch.sqrt(torch.tensor(C).float())  # [B, k, H1*W1] 1, 8, 7680
    # print('corr_sp', corr_sp.shape) 
    return corr_sp, coords0, coords1, batch_index  # coords: [B, 2, k, H1*W1]


class FlowHead(nn.Module):
    def __init__(self, input_dim=256, batch_norm=True):
        super().__init__()
        if batch_norm:
            self.flowpredictor = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, 3, padding=1)
            )
        else:
            self.flowpredictor = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, 3, padding=1)
            )

    def forward(self, x):
        return self.flowpredictor(x)


class SparseNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # feature network, context network, and update block
        self.fnet = BasicEncoderQuarter(output_dim=256, norm_fn='instance', dropout=False)
        self.cnet = BasicEncoderQuarter(output_dim=256, norm_fn='batch', dropout=False)

        # correlation volume encoder
        self.update_block = BasicUpdateBlockQuarter(self.args, hidden_dim=128, input_dim=405)
        # self.update_block = BasicUpdateBlockQuarter(self.args, hidden_dim=128)


    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow_quarter(self, flow, mask):
        """ Upsample flow field [H/4, W/4, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4*H, 4*W)

    def forward(self, image1, image2, iters, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature and context network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        B, _, H1, W1 = fmap1.shape

        # GRU
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        # Generate sparse cost volume for GRU
        # corr_val, coords0_cv, coords1_cv, batch_index_cv = compute_sparse_corr(fmap1, fmap2, k=self.args.num_k)
        corr_val, coords0_cv, coords1_cv, batch_index_cv = compute_sparse_corr(fmap1, fmap2, k=4)
        delta_flow = torch.zeros_like(coords0)

        flow_predictions = []

        search_range = 4
        corr_val = corr_val.repeat(1, 4, 1)

        for itr in range(iters):
            with torch.no_grad():

                # need to switch order of delta_flow, also note the minus sign
                coords1_cv = coords1_cv - delta_flow[:, [1, 0], :, :].view(B, 2, 1, -1)  # [B, 2, k, H1*W1]

                mask_pyramid = []
                weights_pyramid = []
                coords_sparse_pyramid = []

                # Create multi-scale displacements
                for i in range(5):
                    coords1_sp = coords1_cv * 0.5**i
                    weights, coords1_sp = compute_interpolation_weights(coords1_sp)
                    mask = (coords1_sp[:, 0].abs() <= search_range) & (coords1_sp[:, 1].abs() <= search_range)
                    batch_ind = batch_index_cv.permute(0, 2, 3, 1).repeat(1, 4, 1, 1)[mask]
                    coords0_sp = coords0_cv.permute(0, 2, 3, 1).repeat(1, 4, 1, 1)[mask]
                    coords1_sp = coords1_sp.permute(0, 2, 3, 1)[mask]

                    coords1_sp = coords1_sp + search_range
                    coords_sp = torch.cat([batch_ind, coords0_sp, coords1_sp], dim=1)
                    coords_sparse_pyramid.append(coords_sp)

                    mask_pyramid.append(mask)
                    weights_pyramid.append(weights)

            corr_val_pyramid = []
            for mask, weights in zip(mask_pyramid, weights_pyramid):
                corr_masked = (weights * corr_val)[mask].unsqueeze(1)
                corr_val_pyramid.append(corr_masked)

            sparse_tensor_pyramid = [torch.sparse.FloatTensor(coords_sp.t().long(), corr_resample, torch.Size([B, H1, W1, 9, 9, 1])).coalesce()
                                     for coords_sp, corr_resample in zip(coords_sparse_pyramid, corr_val_pyramid)]

            corr = torch.cat([sp.to_dense().view(B, H1, W1, -1) for sp in sparse_tensor_pyramid], dim=3).permute(0, 3, 1, 2)

            coords1 = coords1.detach()

            flow = coords1 - coords0

            # GRU Update
            with autocast(enabled=self.args.mixed_precision):

                # 4D net map to 2D dense vector
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow4(coords1 - coords0)
            else:
                flow_up = self.upsample_flow_quarter(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return flow_up

        return flow_predictions


class SparseNetEighth(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=False)
        self.cnet = BasicEncoder(output_dim=256, norm_fn='batch', dropout=False)

        # correlation volume encoder
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=128, input_dim=405)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0
            光流值用两坐标的距离来计算
        """
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        #归一化
        image1 = 2 * (image1 / 255.0) - 1.0 
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature and context network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        B, _, H1, W1 = fmap1.shape

        # GRU  初始光流值为0
        coords0, coords1 = self.initialize_flow(image1)
        # 更新coords1 的坐标
        if flow_init is not None:
            coords1 = coords1 + flow_init 

        # Generate sparse cost volume for GRU 生成稀疏相关性向量
        #corr_val, coords0_cv, coords1_cv, batch_index_cv = compute_sparse_corr(fmap1, fmap2, k=self.args.num_k)
        corr_val, coords0_cv, coords1_cv, batch_index_cv = compute_sparse_corr(fmap1, fmap2, 8)
                
        delta_flow = torch.zeros_like(coords0) # 残差流

        flow_predictions = []

        search_range = 4  # 搜索范围4
        corr_val = corr_val.repeat(1, 4, 1)

        for itr in range(iters): # gru迭代次数
            with torch.no_grad():

                # need to switch order of delta_flow, also note the minus sign
                coords1_cv = coords1_cv - delta_flow[:, [1, 0], :, :].view(B, 2, 1, -1)  # [B, 2, k, H1*W1]

                mask_pyramid = [] # mask 金字塔
                weights_pyramid = [] # 权重金字塔
                coords_sparse_pyramid = [] # 相关性金字塔

                # Create multi-scale displacements 创建多级位移
                for i in range(5):
                    coords1_sp = coords1_cv * 0.5**i
                    weights, coords1_sp = compute_interpolation_weights(coords1_sp)
                    mask = (coords1_sp[:, 0].abs() <= search_range) & (coords1_sp[:, 1].abs() <= search_range)
                    batch_ind = batch_index_cv.permute(0, 2, 3, 1).repeat(1, 4, 1, 1)[mask]
                    coords0_sp = coords0_cv.permute(0, 2, 3, 1).repeat(1, 4, 1, 1)[mask]
                    coords1_sp = coords1_sp.permute(0, 2, 3, 1)[mask]

                    coords1_sp = coords1_sp + search_range
                    coords_sp = torch.cat([batch_ind, coords0_sp, coords1_sp], dim=1)
                    coords_sparse_pyramid.append(coords_sp)

                    mask_pyramid.append(mask)
                    weights_pyramid.append(weights)

            corr_val_pyramid = []
            for mask, weights in zip(mask_pyramid, weights_pyramid):
                corr_masked = (weights * corr_val)[mask].unsqueeze(1)
                corr_val_pyramid.append(corr_masked)

            sparse_tensor_pyramid = [torch.sparse.FloatTensor(coords_sp.t().long(), corr_resample, torch.Size([B, H1, W1, 9, 9, 1])).coalesce()
                                     for coords_sp, corr_resample in zip(coords_sparse_pyramid, corr_val_pyramid)]

            corr = torch.cat([sp.to_dense().view(B, H1, W1, -1) for sp in sparse_tensor_pyramid], dim=3).permute(0, 3, 1, 2)

            coords1 = coords1.detach()

            flow = coords1 - coords0

            # Not sure if it will affect results.
            with autocast(enabled=self.args.mixed_precision):

                # 4D net map to 2D dense vector
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow4(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return flow_up

        return flow_predictions
