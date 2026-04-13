# The based unit of graph convolutional networks. ST-GCN: https://github.com/yysijie/st-gcn

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class ConvTemporalGraphical1(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 node_types,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        self.conv1 = nn.Conv2d(in_channels, 5 * out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, 5 * out_channels, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_types = node_types
        self.num_semantic=5
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x1, x2 = None, None
        tmp_x = x
        x1 = self.conv1(tmp_x).reshape(n, 5, self.out_channels, t, v)
        x2 = self.conv2(tmp_x).reshape(n, 5, self.out_channels, t, v)
        x1 = x1.mean(dim=-2, keepdim=True)
        x2 = x2.mean(dim=-2, keepdim=True)
        node_type = self.node_type.to(x.device)
        semantic_masks = []
        for i in range(self.num_semantic):
            mask = (node_type == i)
            semantic_masks.append(mask)

        graph_list = []
        for i in range(self.num_semantic):
            mask = semantic_masks[i]
            # semantic nodes
            x1_sem = x1[:, i, :, :, mask]  # (N,C,T,Vs)
            # semantic mean
            x1_sem = x1_sem.mean(-1, keepdim=True)  # (N,C,T,1)

            # rest nodes
            x2_all = x2[:, i]  # (N,C,T,V)

            # one-vs-rest
            diff = x1_sem.unsqueeze(-1) - x2_all.unsqueeze(-2)

            # shape
            # (N,C,T,1,V)

            graph_list.append(diff)
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A