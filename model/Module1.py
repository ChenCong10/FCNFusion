import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H， W, C
        """
        x = x.permute(0, 2, 3, 1)   # [B, C, H, W] --> [B,  H, W, C,]

        B,H,W,C = x.shape
        #assert L == H * W, "input feature has wrong size"
        #
        # x = x.view(B, H, W, C)

        # padding
        # 因为宽和高需要降为原来的一半，所以宽高需要是二的整数倍
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # 只pad最后三个维度H,W,C，并且F.pad函数是倒着来的，头两个参数（0,0）代表Channel维度...
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        x = x.permute(0, 2, 1)


        x = x.view(B, C*2,int(H/2),int(W/2))
        #
        return x


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)



class gnconv(nn.Module):
    def __init__(self, dim, order, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        # print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        # B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,ffn_expansion_factor = 4, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features*ffn_expansion_factor)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W, = x.shape

        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B,  H, W, C,]
        x = x.view(B,H*W,C)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # [B,  H, W, C,] -->  [B, C, H, W]

        return x


class Gnblock(nn.Module):
    def __init__(self, dim, order):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim,order)  # depthwise conv
        self.act = nn.GELU()
        self.mlp=Mlp(dim)


    def forward(self, x):

        """
        x: B, C ,H， W,
        """
        shortcut1= x
        x = self.norm1(x)
        x = self.gnconv(x)
        x = x+shortcut1
        x = self.norm2(x)
        shortcut2=x
        x= self.mlp(x)+shortcut2

        return x



class up_black(nn.Module):

    def __init__(self,dim) :
        super().__init__()

        self.up = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.BNor = nn.BatchNorm2d(dim)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.up(x)
        x=self.BNor(x)
        x=self.relu(x)

        return x

class up_black2(nn.Module):

    def __init__(self,dim) :
        super().__init__()

        self.up = nn.ConvTranspose2d(dim*4, dim, kernel_size=2, stride=2)
        self.BNor = nn.BatchNorm2d(dim)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.up(x)
        x=self.BNor(x)
        x=self.relu(x)

        return x


class down_module(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.patch_embed=OverlapPatchEmbed(1,64)
        self.Gnblock1=Gnblock(dim,3)
        self.PatchMerging1 = PatchMerging(dim)
        self.Gnblock2 = Gnblock(dim*2,3)
        self.PatchMerging2 = PatchMerging(dim*2)
        self.Gnblock3 = Gnblock(dim * 4,3)
        self.PatchMerging3 = PatchMerging(dim * 4)
        self.up1= up_black(dim * 4)
        self.up2 = up_black2(dim * 2)
        self.up3 = up_black2(dim)
        self.up4 = nn.Conv2d(dim*2, 1, kernel_size=1,stride=1)
        self.ru=nn.Tanh()

        # self.c_down1=nn.Conv2d(dim * 8,dim * 4,kernel_size=3,padding=1,bias=False)
        # self.c_down2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=3, padding=1, bias=False)

    def forward(self,x):
        emb_x=self.patch_embed(x)      #B,1,H,W---->B,C,H,W
        x1_out=self.Gnblock1(emb_x)    #B,C,H,W---->B,C,H,W

        x=self.PatchMerging1(x1_out)      #B,C,H,W---->B,2*C,H/2,W/2
        x2_out=self.Gnblock2(x)           #B,2*C,H/2,W/2---->B,2*C,H/2,W/2

        x=self.PatchMerging2(x2_out)        #B,2*C,H/2,W/2---->B,4*C,H/4,W/4
        x3_out=self.Gnblock3(x)             #B,4*C,H/4,W/4---->B,4*C,H/4,W/4
        x=self.PatchMerging3(x3_out)        #B,4*C,H/4,W/4---->B,8*C,H/8,W/8

        x = self.up1(x)                    #B,8*C,H/8,W/8---->B,4*C,H/4,W/4
        x = self.Gnblock3(x)              #B,4*C,H/4,W/4---->B,4*C,H/4,W/4

        x = torch.cat((x, x3_out), axis=1)     #B,4*C,H/4,W/4---->B,8*C,H/4,W/4

        x = self.up2(x)                      #B,8*C,H/4,W/4---->B,2*C,H/2,W/2
        x = self.Gnblock2(x)                    #B,2*C,H/2,W/2---->B,2*C,H/2,W/2
        #
        x = torch.cat((x, x2_out), axis=1)   # B,4*C,H/4,W/4---->B,8*C,H/4,W/4
        # x = self.c_down2(x)  # B,8*C,H/4,W/4---->B,4*C,H/4,W/4
        #
        x = self.up3(x)
        x = self.Gnblock1(x)

        x = torch.cat((x, x1_out), axis=1)


        x = self.up4(x)
        x = self.ru(x)

        return x

x = torch.randn(4,1, 256, 256)



M_1=down_module(64)
x1=M_1(x)

print(x.shape)
print(x1.shape)






