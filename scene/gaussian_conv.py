import torch
import torch_scatter  # 用于对张量的维度进行分组聚合等
from sklearn.neighbors import NearestNeighbors  # 邻近搜索算法
from fast_pytorch_kmeans import KMeans  # 快速KMeans聚类

class GaussianConv(torch.nn.Module):
    def __init__(self, xyz, input_channel=256, layers_channel=[256, 128, 64, 32, 3], downsample_layer=[], upsample_layer=[], K=8):
        # xyz：[N,D]; layers_channel：每层卷积的输出通道数；K：每个点云的邻居数量
        super(GaussianConv, self, ).__init__()  # 调用父类构造函数
        assert len(downsample_layer) == len(upsample_layer) == 0 or \
            (len(downsample_layer) == len(upsample_layer) and max(downsample_layer) < min(upsample_layer)) ,\
            'downsample_layer and upsample_layer must be the same length and satisfy max(downsample_layer) < min(upsample_layer) or both are empty lists'
        
        self.K = K
        self.N = xyz.shape[0]  #点云数量
        self.downsample_layer = downsample_layer
        self.upsample_layer = upsample_layer

        self.init_kmeans_knn(xyz, len(downsample_layer))  # 初始化KMeans聚类
        self.init_conv_params(input_channel, layers_channel)  # 初始化卷积层参数

    @torch.no_grad()  # 装饰器模式，禁用整个函数的梯度计算
    def init_kmeans_knn(self, xyz, len_sample_layer):
        self.knn_indices = []  # KNN索引
        self.kmeans_labels = []  # KMeans聚类标签

        # get original knn_indices
        xyz_numpy = xyz.cpu().numpy()  # 转换为Numpy数组
        nn = NearestNeighbors(n_neighbors=self.K, algorithm='auto')
        nn.fit(xyz_numpy)  # 计算每个点对应的索引
        _, knn_indices = nn.kneighbors(xyz_numpy) # [N, K]  记录每个点K个最近邻的索引
        self.knn_indices.append(knn_indices) 
        # 初始化点云数及其坐标
        last_N = self.N
        last_xyz = xyz

        for i in range(len_sample_layer):
            print('Using KMeans to cluster point clouds in level', i)
            kmeans = KMeans(n_clusters=last_N//self.K, mode='euclidean', verbose=1)  # 初始化聚类方法，计算聚类数，采用欧几里得距离
            self.kmeans_labels.append(kmeans.fit_predict(last_xyz)) # [N]  进行聚类并记录
            # 得到每个聚类的质心位置
            down_centroids = torch_scatter.scatter(last_xyz, self.kmeans_labels[-1], dim=0, reduce='mean') # [cluster_num=N//5, D]

            # get knn_indices for downsampled point clouds
            nn = NearestNeighbors(n_neighbors=self.K, algorithm='auto')
            nn.fit(down_centroids.cpu().numpy())  # 计算每个质心对应的索引
            _, knn_indices = nn.kneighbors(down_centroids.cpu().numpy())  # 记录每个质心K个最近邻的索引
            self.knn_indices.append(knn_indices)
            # 更新点云数及其坐标，准备下一层次的聚类
            last_N = down_centroids.shape[0]
            last_xyz = down_centroids

    def init_conv_params(self, input_channel, layers_channel):  # 记录每层卷积的卷积核与偏置
        self.kernels = []
        self.bias = []
        for out_channel in layers_channel:
            self.kernels.append(torch.randn(out_channel, self.K*input_channel)*0.1)  # [out_channel, K*input_channel]
            self.bias.append(torch.zeros(1, out_channel))  # [1, out_channel]
            input_channel = out_channel  # 更新输入通道数
        # 转化为可训练的参数
        self.kernels = torch.nn.ParameterList(self.kernels)
        self.bias = torch.nn.ParameterList(self.bias)

    def forward(self, features):
        '''
        Args:
            features: [N, D]
            D: input_channel
            S: output_channel
        '''
        sample_level = 0
        for i in range(len(self.kernels)):
            if i in self.downsample_layer:  # 下采样
                sample_level += 1
                features = torch_scatter.scatter(features, self.kmeans_labels[sample_level-1], dim=0, reduce='mean')  # 根据聚类标签进行特征聚合
            elif i in self.upsample_layer:  # 上采样
                sample_level -= 1
                features = features[self.kmeans_labels[sample_level]]  # 直接使用之前的聚类标签

            knn_indices = self.knn_indices[sample_level]  #得到当前层的最近邻索引

            knn_features = features[knn_indices] # [N, K, D]
            knn_features = knn_features.reshape(knn_features.size(0), -1) # [N, K*D]  # 展平最近邻特征
            features = knn_features @ self.kernels[i].T + self.bias[i] # [N, S]  # 计算卷积并加偏置
            features = torch.sigmoid(features) if i != len(self.kernels)-1 else features  # 除最后一层外对每一层使用sigmoid激活函数

        return features # [N, S]