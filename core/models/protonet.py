import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        # コサイン類似度分類器に使用されるバイアス項
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        # コサイン類似度分類器のスケール項
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # backbone
        self.backbone = backbone

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim()-1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2) # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores
    # サポートセットとクエリセットからの画像を処理するためのフォワードパス関数を定義している関数
    def forward(self, supp_x, supp_y, x):
        # supp_xはサポートセットの入力データ（画像）、supp_yはそれに対応するラベル、xはクエリセットの入力データ（画像）
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        # サポートセットのラベルから最大値を取得し、クラスの数を推定します。+1はラベルが0から始まると仮定
        num_classes = supp_y.max() + 1 # NOTE: assume B==1
        # 
        B, nSupp, C, H, W = supp_x.shape
        # サポートセットの画像から特徴を抽出
        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        print("supp_f:",supp_f.size())
        # バッチ内のすべての画像をバッチ次元に結合
        supp_f = supp_f.view(B, nSupp, -1)
        # ラベルをワンホットエンコーディングに変換し、次元を変更して対応する形にします
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        # クラスごとの特徴ベクトルの平均を計算し、プロトタイプを生成
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        print("prototype:", prototypes.size())
        # 各クラスに対してサポートセットの画像が存在するかどうかを確認し、プロトタイプの平均を取る際に0で割ることがないようにします
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images
        print("prototype2:", prototypes.size())
        # クエリセットの画像から特徴を抽出
        feat = self.backbone.forward(x.view(-1, C, H, W))
        print("feat:",feat.size())
        feat = feat.view(B, x.shape[1], -1) # B, nQry, d
        print("feat:",feat.size())
        logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
        return logits
