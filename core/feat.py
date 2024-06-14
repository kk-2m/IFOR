from torch import nn
# 追加
import os
import numpy as np
import torch
#from timm.models import create_model

__all__ = ['BasicBlock', 'conv1x1', 'feat_extract']


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv10 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.dr4 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        x = x.view(-1, 512)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# カスタムResNet18-fractal用
class BasicBlock_fractal(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_fractal, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# カスタムの畳み込み層を定義する
class AdvBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_layer=None):
        super(AdvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # 1x1の畳み込み層を定義
        self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)
        # 3x3の畳み込み層を定義
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        # 正規化層を定義
        self.bn1 = norm_layer(planes)
        # relu活性化関数を定義し、非線形性が導入される
        self.relu = nn.ReLU(inplace=True)
        # 上と同様
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(planes)

    def forward(self, x):
        # 入力データに対して1x1の畳み込みを適用
        identity = self.conv0(x)
        #同様に以下レイヤーに適用
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # identityと畳み込みブロックの出力を足し合わせ、ショートカット接続を行なう。
        out += identity
        # 最後に活性化関数を通して、出力を返す。
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, norm_layer=None, branch=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.n_blocks = len(layers)
        self.branch = branch

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_0 = self._make_layer(block, 512, layers[3], stride=2)
        if branch:
            self.inplanes = 256 * block.expansion
            self.layer4_1 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        N = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        mu = self.layer4_0(x)
        if self.branch:
            sig = self.layer4_1(x)
            return [mu, sig]
        else:
            return mu

        return x


class ResNet10(nn.Module):

    def __init__(self, norm_layer=None, branch=False):
        super(ResNet10, self).__init__()
        # 正規化レイヤーが存在しないなら、これで初期化される
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # 畳み込み層の入力チャンネル数
        self.inplanes = 64
        # ブランチを管理？
        self.branch = branch
        # プーリング層を設定
        # 
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 最初の畳み込み層を定義する。入力チャンネル数３から出力チャンネル数64に変換
        self.layer1 = AdvBlock(3, 64)
        # 64から128に変換
        self.layer2 = AdvBlock(64, 128)
        self.layer3 = AdvBlock(128, 256)
        self.layer4_0 = AdvBlock(256, 512)
        # ブランチを使うなら別のレイヤーを追加する
        if branch:
            self.layer4_1 = AdvBlock(256, 512)

    def forward(self, x):
        N = x.size(0)
        # 次元数を上げてはプーリングをする。
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        mu = self.maxpool(self.layer4_0(x))
        print("mu:",mu.size())
        if self.branch:
            sig = self.maxpool(self.layer4_1(x))
            print("sig",sig.size())
            return [mu, sig]
        else:
            return mu

class ResNet_fractal(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_fractal, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print("x",x.size())
        # x = self.fc(x)
        return x


def resnet18_fractal(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fractal(BasicBlock_fractal, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

eps = 1e-10

def get_backbone(args):
    # original vit(not pretrained)
    # if args.arch == 'vit-origin':
    #     from .models import vision_transformer as vit
    #     model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
    #     print('No pretrained(original) ViT is used')
    if args.arch == 'vit-origin':
        from timm.models.vision_transformer import VisionTransformer
        from functools import partial
        model = VisionTransformer(
            patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=64)
        model.head = torch.nn.Identity()
        print('ViT origin is used')
    elif args.arch == 'vit-fractal-1k':
        from timm.models.vision_transformer import VisionTransformer
        from functools import partial
        model = VisionTransformer(
            patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=64)
        # ViT フラクタルの論文からダウンロードしたもの
        fractal_1k_path = "core/pretrained_models/deitt16_224_fractal1k_lr3e-4_300ep.pth"
        state_dict = torch.load(fractal_1k_path)
        for k in [ "model", "optimizer", "lr_scheduler", "scaler", "epoch"]:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        model.load_state_dict(state_dict, strict=True)
        model.head = torch.nn.Identity()
        print('fractal-1k ViT is used')

    elif args.arch == 'vit-fractal-10k':
        from .models import vision_transformer as vit
        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
        fractal_1k_path = "core/pretrained_models/deitt16_224_fractal10k_lr3e-4_100ep.pth"
        state_dict = torch.load(fractal_1k_path)

        for k in [ "model", "optimizer", "lr_scheduler", "scaler", "epoch"]:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        print('fractal-1k ViT is used')

    # ViTフラクタルをImageNetでファインチューニングしたもの。
    # そうすいに頼んでいたが、良い結果が得られず、頓挫した。
    elif args.arch == "vit-fractal-custom15":
        # deit_tiny_patch16_224_imagenet1k_001ep.pth
        from timm.models.vision_transformer import VisionTransformer
        from functools import partial
        # model = VisionTransformer(
        #     patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=64)
        model = VisionTransformer(
            patch_size=16, embed_dim=192, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=64)
        fractal_1k_path = "core/pretrained_models/deit_tiny_patch16_224_imagenet1k_150ep.pth"
        state_dict = torch.load(fractal_1k_path)["model"]
        print()
        # for k in [ "model", "optimizer", "lr_scheduler", "scaler", "epoch"]:
        #     if k in state_dict:
        #         print(f"removing key {k} from pretrained checkpoint")
        #         del state_dict[k]
        # print("model",model)
        # print("state_dict",state_dict)
        # model = torch.nn.parallel.DataParallel(model)
        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        model.head = torch.nn.Identity()
        model.load_state_dict(state_dict, strict=True)
        
        # print('fractal-1k ViT is used')
        print('fractal-1k-custom ViT is used')


    elif args.arch == 'vit_base_patch16_224_in21k':
        from .models.vit_google import VisionTransformer, CONFIGS

        config = CONFIGS['ViT-B_16']
        model = VisionTransformer(config, 224)

        # url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
        # pretrained_weights = 'pretrained_ckpts/vit_base_patch16_224_in21k.npz'
        pretrained_weights = 'core/pretrained_models/ViT-B_16.npz'

        # if not os.path.exists(pretrained_weights):
        #     try:
        #         import wget
        #         os.makedirs('pretrained_ckpts', exist_ok=True)
        #         wget.download(url, pretrained_weights)
        #     except:
        #         print(f'Cannot download pretrained weights from {url}. Check if `pip install wget` works.')

        model.load_from(np.load(pretrained_weights))
        print('Pretrained weights found at {}'.format(pretrained_weights))
    # default
    elif args.arch == 'dino_base_patch16':
        from .models import vision_transformer as vit
        # vit_baseをパッチサイズ16, クラス数0で初期化。なぜ0?
        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        # URLから事前学習済みの重みをダウンロード
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        # モデルにロードする
        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'deit_base_patch16':
        from .models import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'deit_small_patch16':
        from .models import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'deit_tiny_patch16':
        from .models import vision_transformer as vit

        model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=0)
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))
        print('Pretrained weights found at {}'.format(url))
    # deit。これを使用して、ViT x deitによる一般的な学習(headを用いる学習)を行った。
    elif args.arch == 'deit_tiny_patch16_head':
        from .models import vision_transformer as vit

        model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=28)
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {}'.format(url))
    # 上の一般的な学習で保存した重みを使用するコード。これでIFORの実験を行った。
    elif args.arch == 'deit_tiny_patch16_head_custom':
        from .models import vision_transformer as vit

        model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=28)
        path = "C:/Users/Image-lab/Ranmaru/Ranmaru/mymodel/deit_tiny_patch16_head.pth"
        state_dict = torch.load(path)
        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        model.head = torch.nn.Identity()
        model.load_state_dict(state_dict, strict=True)
        # print('Pretrained weights found at {}'.format(url))
        print("deit_tiny_patch16_head_custom is used")
    elif args.arch == 'dino_small_patch16':
        from .models import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)

        if not args.no_pretrain:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

            model.load_state_dict(state_dict, strict=True)
            print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'beit_base_patch16_224_pt22k':
        from .beit import default_pretrained_model
        model = default_pretrained_model(args)
        print('Pretrained BEiT loaded')

    elif args.arch == 'clip_base_patch16_224':
        from . import clip
        model, _ = clip.load('ViT-B/16', 'cpu')

    elif args.arch == 'clip_resnet50':
        from . import clip
        model, _ = clip.load('RN50', 'cpu')

    elif args.arch == 'dino_resnet50':
        from torchvision.models.resnet import resnet50

        model = resnet50(pretrained=False)
        model.fc = torch.nn.Identity()

        if not args.no_pretrain:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=False)

    elif args.arch == 'resnet50':
        from torchvision.models.resnet import resnet50

        pretrained = not args.no_pretrain
        model = resnet50(pretrained=pretrained)
        model.fc = torch.nn.Identity()

    elif args.arch == 'resnet18':
        from torchvision.models.resnet import resnet18

        pretrained = not args.no_pretrain
        # model = resnet18(pretrained=pretra ined)
        model = resnet18(pretrained=False)

        model.fc = torch.nn.Identity()

    elif args.arch == 'dino_xcit_medium_24_p16':
        model = torch.hub.load('facebookresearch/xcit:main', 'xcit_medium_24_p16')
        model.head = torch.nn.Identity()
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)

    elif args.arch == 'dino_xcit_medium_24_p8':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')

    elif args.arch == 'simclrv2_resnet50':
        import sys
        sys.path.insert(
            0,
            'cog',
        )
        import model_utils

        model_utils.MODELS_ROOT_DIR = 'cog/models'
        ckpt_file = os.path.join(args.pretrained_checkpoint_path, 'pretrained_ckpts/simclrv2_resnet50.pth')
        resnet, _ = model_utils.load_pretrained_backbone(args.arch, ckpt_file)

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(x, apply_fc=False)

        model = Wrapper(resnet)

    elif args.arch in ['mocov2_resnet50', 'swav_resnet50', 'barlow_resnet50']:
        from torchvision.models.resnet import resnet50

        model = resnet50(pretrained=False)
        ckpt_file = os.path.join(args.pretrained_checkpoint_path, 'pretrained_ckpts_converted/{}.pth'.format(args.arch))
        ckpt = torch.load(ckpt_file)

        msg = model.load_state_dict(ckpt, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        # remove the fully-connected layer
        model.fc = torch.nn.Identity()
    elif args.arch == 'resnet18-fractal-1k':
            # from torchvision.models.resnet import resnet18
            # pretrained = not args.no_pretrain
            # model = resnet18(pretrained=pretrained)
            # model = resnet18(pretrained=False)
            model = resnet18_fractal()
            fractal_1k_path = "D:\GraduationProject\Code\Models\Mymodel\mymodel\core\pretrained_models\FractalDB-1000_res18.pth"
            # fractal_1k_path = "core/pretrained_models/FractalDB-10000_res18.pth"
            if os.path.exists(fractal_1k_path):
                state_dict = torch.load(fractal_1k_path)
                print("load pretrain-model")
            else:
                print("error")
            # for k in ["fc.weight", "fc.bias"]:
            #     if k in state_dict:
            #         print(f"removing key {k} from pretrained checkpoint")
            #         del state_dict[k]

            model.load_state_dict(state_dict, strict=True)
            model.fc = torch.nn.Identity()
            print('fractal-1k ResNet18 is used')
    elif args.arch == 'resnet18-fractal-imgnet':
            # from torchvision.models.resnet import resnet18
            # pretrained = not args.no_pretrain
            # model = resnet18(pretrained=pretrained)
            # model = resnet18(pretrained=False)
            model = resnet18_fractal()
            fractal_1k_path = "D:/GraduationProject/Code/Models/Mymodel/mymodel/core/pretrained_models/ft_imagenet1k_resnet18_custom_epoch17.pth"
            # fractal_1k_path = "core/pretrained_models/FractalDB-10000_res18.pth"
            if os.path.exists(fractal_1k_path):
                state_dict = torch.load(fractal_1k_path)
                print("load pretrain-model")
            else:
                print("error")
            # for k in ["fc.weight", "fc.bias"]:
            #     if k in state_dict:
            #         print(f"removing key {k} from pretrained checkpoint")
            #         del state_dict[k]

            model.load_state_dict(state_dict, strict=True)
            model.fc = torch.nn.Identity()
            print('fractal-1k-imgnet ResNet18 is used')

    else:
        raise ValueError(f'{args.arch} is not conisdered in the current code.')

    return model

def feat_extract(opts,args, **kwargs):
    # defaultではこれ。Resnet10, resnet18の元コードを利用する場合。しかし実験では使用せず。
    if kwargs["structure"] == "resnet":
        if args.arch == 'resnet10':
            return ResNet10(branch=kwargs['branch']), AdvBlock.expansion
        elif args.arch == 'resnet18':
            block = BasicBlock
            layers = [2, 2, 2, 2]
            return ResNet(block, layers, branch=kwargs['branch']), block.expansion
        elif args.arch == 'conv':
            return ConvNet(), 1
        elif args.arch == 'resnet18-fractal-1k':
            # from torchvision.models.resnet import resnet18
            # pretrained = not args.no_pretrain
            # model = resnet18(pretrained=pretrained)
            # model = resnet18(pretrained=False)
            model = resnet18_fractal()
            fractal_1k_path = "D:\GraduationProject\Code\Models\Mymodel\mymodel\core\pretrained_models\FractalDB-1000_res18.pth"
            # fractal_1k_path = "core/pretrained_models/FractalDB-10000_res18.pth"
            if os.path.exists(fractal_1k_path):
                state_dict = torch.load(fractal_1k_path)
                print("load pretrain-model")
            else:
                print("error")
            # for k in ["fc.weight", "fc.bias"]:
            #     if k in state_dict:
            #         print(f"removing key {k} from pretrained checkpoint")
            #         del state_dict[k]

            model.load_state_dict(state_dict, strict=True)
            model.fc = torch.nn.Identity()
            print('fractal-1k ResNet18 is used')

            block = BasicBlock

            return model, block.expansion
    # 実験ではこちらを利用した。PMFのコードから抽出したコードを主に使っている。
    elif kwargs["structure"] == "vit":
        return get_backbone(args), AdvBlock.expansion
    else:
        raise NameError('structure not known {} ...'.format(kwargs['structure']))
