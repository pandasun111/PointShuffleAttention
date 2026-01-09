import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F


class MultiScaleShuffleAttention3D(nn.Module):
    def __init__(self, channels, groups=4, sa_kernel_sizes=(3, 5, 3), sa_dilations=(1, 1, 2)):
        super().__init__()
        assert channels % groups == 0
        self.groups = groups
        group_ch = channels // groups

        # Channel Attention
        self.ca_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_ch, group_ch // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(group_ch // 4, group_ch, bias=False),
                nn.Sigmoid()
            ) for _ in range(groups)
        ])

        # Multi-scale Spatial Attention
        self.sa_convs = nn.ModuleList([
            nn.ModuleList([
                ME.MinkowskiConvolution(
                    group_ch, group_ch,
                    kernel_size=k, stride=1, dilation=d, dimension=3
                )
                for k, d in zip(sa_kernel_sizes, sa_dilations)
            ]) for _ in range(groups)
        ])

        self.scale_weight = nn.Parameter(torch.ones(len(sa_kernel_sizes)))

    def forward(self, x: ME.SparseTensor):
        feats = torch.chunk(x.F, self.groups, dim=1)
        outs = []
        scale_w = torch.softmax(self.scale_weight, dim=0)

        for i in range(self.groups):
            xi = ME.SparseTensor(
                feats[i],
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager
            )

            # Channel Attention
            gap = torch.mean(xi.F, dim=0, keepdim=True)
            ca = self.ca_fc[i](gap)
            x_ca = xi.F * ca

            # Spatial Attention
            sa_maps = [conv(xi).F for conv in self.sa_convs[i]]
            sa = sum(w * m for w, m in zip(scale_w, sa_maps))
            sa = torch.sigmoid(sa)

            outs.append(x_ca + xi.F * sa)

        out = torch.cat(outs, dim=1)
        out = ME.SparseTensor(
            out,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )

        return channel_shuffle_sparse(out, self.groups)

def channel_shuffle_sparse(x: ME.SparseTensor, groups: int):
    N, C = x.F.shape
    x_f = x.F.view(N, groups, C // groups)
    x_f = x_f.transpose(1, 2).contiguous()
    x_f = x_f.view(N, C)
    return ME.SparseTensor(
        x_f,
        coordinate_map_key=x.coordinate_map_key,
        coordinate_manager=x.coordinate_manager
    )

class SparseResBlock_MSSA(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        kernel_size=3,
        use_attn=True
    ):
        super().__init__()

        self.conv1 = ME.MinkowskiConvolution(
            in_ch, out_ch,
            kernel_size=kernel_size,
            stride=stride,
            dimension=3
        )
        self.bn1 = ME.MinkowskiBatchNorm(out_ch)

        self.conv2 = ME.MinkowskiConvolution(
            out_ch, out_ch,
            kernel_size=kernel_size,
            stride=1,
            dimension=3
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_ch)

        self.attn = MultiScaleShuffleAttention3D(out_ch) if use_attn else None
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_ch, out_ch,
                    kernel_size=1,
                    stride=stride,
                    dimension=3
                ),
                ME.MinkowskiBatchNorm(out_ch)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.attn is not None:
            out = self.attn(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(out + identity)

class GatedMultiScaleShuffleAttention3D(nn.Module):
    def __init__(self, channels, groups=4,
                 sa_kernel_sizes=(3, 5, 3),
                 sa_dilations=(1, 1, 2), use_sa=True):
        super().__init__()
        assert channels % groups == 0
        self.groups = groups
        group_ch = channels // groups

        # Channel Attention
        self.ca_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_ch, group_ch // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(group_ch // 4, group_ch, bias=False),
            ) for _ in range(groups)
        ])

        # Multi-scale Spatial Attention
        self.sa_convs = nn.ModuleList([
            nn.ModuleList([
                ME.MinkowskiConvolution(
                    group_ch, group_ch,
                    kernel_size=k,
                    stride=1,
                    dilation=d,
                    dimension=3
                )
                for k, d in zip(sa_kernel_sizes, sa_dilations)
            ]) for _ in range(groups)
        ])

        self.scale_weight = nn.Parameter(torch.ones(len(sa_kernel_sizes)))
        self.use_sa = use_sa

    def forward(self, x: ME.SparseTensor):
        feats = torch.chunk(x.F, self.groups, dim=1)
        scale_w = torch.softmax(self.scale_weight, dim=0)
        outs = []

        for i in range(self.groups):
            xi = ME.SparseTensor(
                feats[i],
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager
            )

            # Channel gate
            gap = torch.mean(xi.F, dim=0, keepdim=True)
            ca = self.ca_fc[i](gap)

            # Spatial gate
            if self.use_sa:
                sa_maps = [conv(xi).F for conv in self.sa_convs[i]]
                sa = sum(w * m for w, m in zip(scale_w, sa_maps))
            else:
                sa = 1.0

            # Gated residual attention
            gate = torch.sigmoid(ca * sa)
            outs.append(xi.F * (1.0 + gate))

        out = torch.cat(outs, dim=1)
        out = ME.SparseTensor(
            out,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )

        return channel_shuffle_sparse(out, self.groups)

class CrossLevelFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.scale = channels ** -0.5

    def forward(self, f_low: ME.SparseTensor, f_high: ME.SparseTensor):
        q = self.q(f_high.F)
        k = self.k(f_low.F)
        v = self.v(f_low.F)

        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) * self.scale)
        fused = f_high.F + attn * v

        return ME.SparseTensor(
            fused,
            coordinate_map_key=f_high.coordinate_map_key,
            coordinate_manager=f_high.coordinate_manager
        )

class ClusterEmbeddingHead(nn.Module):
    def __init__(self, in_ch=128, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=1)

class MS_SA_Deep3DBackbone_1(nn.Module):
    """
    Input : TensorField (n, 6)
    Output: torch.Tensor (n, 64)  # 聚类嵌入
    """
    def __init__(
        self,
        stem_kernel=3,
        block_kernel=3,
        stride_stage2=2,
        stride_stage3=1
    ):
        super().__init__()

        self.stem = ME.MinkowskiConvolution(
            6, 32,
            kernel_size=stem_kernel,
            stride=1,
            dimension=3
        )

        self.block1 = SparseResBlock_MSSA(
            32, 64, stride=1,
            kernel_size=block_kernel,
            use_attn=False
        )

        self.block2 = SparseResBlock_MSSA(
            64, 64,
            stride=stride_stage2,
            kernel_size=block_kernel
        )

        self.block3 = SparseResBlock_MSSA(
            64, 128,
            stride=stride_stage3,
            kernel_size=block_kernel
        )

        self.block4 = SparseResBlock_MSSA(
            128, 128,
            stride=1,
            kernel_size=block_kernel
        )

        self.block5 = SparseResBlock_MSSA(
            128, 128,
            stride=1,
            kernel_size=block_kernel
        )

        # replace attention
        # self.block1.attn = GatedMultiScaleShuffleAttention3D(64)
        # self.block2.attn = GatedMultiScaleShuffleAttention3D(64)

        # self.block3.attn = GatedMultiScaleShuffleAttention3D(128)
        # self.block4.attn = GatedMultiScaleShuffleAttention3D(128)
        self.block5.attn = GatedMultiScaleShuffleAttention3D(128)

        self.proj3 = ME.MinkowskiConvolution(128, 128, kernel_size=1, dimension=3)

        self.fusion = CrossLevelFusion(128)
        self.embed_head = ClusterEmbeddingHead(128, 128)

    def forward(self, x: ME.TensorField):
        in_field = x
        x = x.sparse()

        x = self.stem(x)

        f1 = self.block1(x)
        x = self.block2(f1)

        f3 = self.block3(x)
        x = self.block4(f3)
        f5 = self.block5(x)

        f3 = self.proj3(f3)
        out = self.fusion(f3, f5)

        feat = out.slice(in_field).F
        feat = self.embed_head(feat)

        return feat