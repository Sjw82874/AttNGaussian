import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_utils import calc_mean_std

class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN,self).__init__()
        # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128,64,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64,matrixSize,3,1,1))
        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)
    

class MulLayer(nn.Module):
    def __init__(self, matrixSize=32, adain=True):
        super(MulLayer,self).__init__()
        self.adain = adain
        if adain:
            return

        self.snet = CNN(matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, matrixSize)
        )
        self.unzip = nn.Sequential(
            nn.Linear(matrixSize, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )


    def forward(self,cF,sF, trans=True):
        '''
        input:
            point cloud features: [N, C]
            style image features: [1, C, H, W]
            D: matrixSize
        '''
        if self.adain:
            cF = cF.T # [C, N]
            style_mean, style_std = calc_mean_std(sF) # [1, C, 1]
            content_mean, content_std = calc_mean_std(cF.unsqueeze(0)) # [1, C, 1]

            style_mean = style_mean.squeeze(0)
            style_std = style_std.squeeze(0)
            content_mean = content_mean.squeeze(0)
            content_std = content_std.squeeze(0)

            cF = (cF - content_mean) / content_std
            cF = cF * style_std + style_mean
            return cF.T
      
        assert cF.size(1) == sF.size(1), 'cF and sF must have the same channel size'
        assert sF.size(0) == 1, 'sF must have batch size 1'
        N, C = cF.size()
        B, C, H, W = sF.size()

        # normalize point cloud features
        cF = cF.T # [C, N]
        style_mean, style_std = calc_mean_std(sF) # [1, C, 1]
        content_mean, content_std = calc_mean_std(cF.unsqueeze(0)) # [1, C, 1]

        content_mean = content_mean.squeeze(0)
        content_std = content_std.squeeze(0)

        cF = (cF - content_mean) / content_std # [C, N]
        # compress point cloud features
        compress_content = self.compress(cF.T).T # [D, N]

        # normalize style image features
        sF = sF.view(B,C,-1)
        sF = (sF - style_mean) / style_std  # [1, C, H*W]

        if(trans):
            # get style transformation matrix
            sMatrix = self.snet(sF.reshape(B,C,H,W)) # [B=1, D*D]
            sMatrix = sMatrix.view(self.matrixSize,self.matrixSize) # [D, D]

            transfeature = torch.mm(sMatrix, compress_content).T # [N, D]
            out = self.unzip(transfeature).T # [C, N]

            style_mean = style_mean.squeeze(0) # [C, 1]
            style_std = style_std.squeeze(0) # [C, 1]

            out = out * style_std + style_mean
            return out.T # [N, C]
        else:
            out = self.unzip(compress_content.T) # [N, C]
            out = out * content_std + content_mean
            return out


class LearnableIN(nn.Module):
    '''
    Input: (N, C, L) / (C, L)
    '''
    def __init__(self, dim=256):
        super().__init__()
        self.IN = torch.nn.InstanceNorm1d(dim, momentum=1e-4, track_running_stats=True)

    def forward(self, x):
        if x.size()[-1] <= 1:
            return x
        return self.IN(x)
    

class AdaAttN_new_IN(nn.Module):
    def __init__(self, dim=256):
        """
        Args:
            dim (int): query, key and value size.
        """
        super(AdaAttN_new_IN, self).__init__()

        self.q_embed = nn.Conv1d(dim, dim, 1)
        #self.k_embed = nn.Conv1d(dim, dim, 1)
        #self.s_embed = nn.Conv1d(dim, dim, 1)
        self.k_embed = nn.Conv2d(dim, dim, (1,1))
        self.s_embed = nn.Conv2d(dim, dim, (1,1))
        self.IN = LearnableIN(dim)

    def forward(self, q, k):
        """
        Args:
            q (float tensor, (1, C, N)): query (content) features.
            k (float tensor, (1, C, H*W)): key (style) features.
            c (float tensor, (1, C, N)): content value features.
            s (float tensor, (1, C, H*W)): style value features.

        Returns:
            cs (float tensor, (1, C, N)): stylized content features.
        """
        q = q.T.unsqueeze(0)                                     # [1, C, N]
        #k = k.flatten(2)                                         # [1, C, H*W]
        c, s = q, k

        # QKV attention with projected content and style features
        #q = self.q_embed(self.IN(q)).transpose(2,1)              # [1, N, C]
        q = self.q_embed(F.instance_norm(q)).transpose(2,1)      # [1, N, C]
        #k = self.k_embed(F.instance_norm(k))                     # [1, C, H*W]
        #s = self.s_embed(s).transpose(2,1)                       # [1, H*W, C]
        k = self.k_embed(F.instance_norm(k)).flatten(2)          # [1, C, H*W]
        s = self.s_embed(s).flatten(2).transpose(2,1)            # [1, H*W, C]

        scale = q.size(-1) ** -0.5
        attn = F.softmax(torch.bmm(q, k) * scale, -1)            # [1, N, H*W]
        #attn = F.softmax(torch.bmm(q, k), -1)                    # [1, N, H*W]
        
        # attention-weighted channel-wise statistics
        mean = torch.bmm(attn, s)                                # [1, N, C]
        var = F.relu(torch.bmm(attn, s ** 2) - mean ** 2)        # [1, N, C]
        mean =mean.transpose(2,1)                                # [1, C, N]
        std = torch.sqrt(var).transpose(2,1)                     # [1, C, N]
        
        #cs = self.IN(c) * std + mean                             # [1, C, N]
        cs = F.instance_norm(c) * std + mean                     # [1, C, N]
        return cs.squeeze(0).T