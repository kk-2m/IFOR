import torch
from torch import nn
from torch.nn import functional as F

from core.feat import feat_extract, BasicBlock, conv1x1

class OpenNet(nn.Module):
    def __init__(self, opts, args):
        super(OpenNet, self).__init__()

        self.opts = opts
        self.args = args
        self.num_classes = opts.model.num_classes

        self._norm_layer = nn.BatchNorm2d

        # 追加部分
        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)


        opts.logger('Building up models ...')
        # feature extractor
        # train.modeはdefaulでopen_few
        if opts.train.mode == 'regular':
            self.feat_net, self.block_expansion = feat_extract(opts, structure=opts.model.structure, branch=False)
        elif opts.train.open_detect == 'center':
            # self.feat_net, self.block_expansion = feat_extract(structure=opts.model.structure, branch=False)
            self.feat_net, self.block_expansion = feat_extract(opts, args, structure=opts.model.structure)
        #defaultではこれが選択される
        else:
            # self.feat_net, self.block_expansion = feat_extract(structure=opts.model.structure, branch=True)
            self.feat_net, self.block_expansion = feat_extract(opts, args, structure=opts.model.structure)
                # args.structure=='vit'だったとき
                # feat_extractに引数が渡されると、feat_extract上でfeat.py/get_backboneが実行される
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sm = nn.Softmax(dim=1)

        # 分類損失を求めるために特徴量を尤度に落とすために用いる
        self.fc = nn.Linear(384 * self.block_expansion, self.num_classes)

        if opts.train.mode == 'openfew':
            self.cel_all = nn.CrossEntropyLoss()
            block = BasicBlock
            # self.inplanes = 512 * self.block_expansion
            # self.layer_sigs_0 = self._make_layer(block, 512, 2, stride=1)
            # self.inplanes = 512 * block.expansion * 2
            # self.layer_sigs_1 = self._make_layer(block, 512, 2, stride=1)

            # モデルの選択
            # 新しいモデルを入れたければここに追加する
            # feat.pyにも記述する
            if self.opts.train.aux:
                print("args.arch",args.arch)
                if self.opts.model.structure == "resnet":
                    self.fc = nn.Linear(512 * self.block_expansion, self.num_classes)
                elif self.opts.model.structure == "vit":
                    if args.arch == 'resnet18' or args.arch == 'resnet18-fractal-1k' or args.arch == "resnet18-fractal-imgnet":
                        self.fc = nn.Linear(512 * self.block_expansion, self.num_classes)
                        print("custom_resnet")
                    elif args.arch == 'clip_vit_base_patch16':
                        self.fc = nn.Linear(512 * self.block_expansion, self.num_classes)
                    elif args.arch == 'vit-fractal-1k' or args.arch == 'vit-fractal-custom15' or args.arch == "vit-origin" or args.arch == "deit_tiny_patch16":
                        self.fc = nn.Linear(192 * self.block_expansion, self.num_classes)
                        # self.fc = nn.Linear(192 * self.block_expansion, self.num_classes)
                    elif args.arch == "dino_base_patch16":
                        self.fc = nn.Linear(768 * self.block_expansion, self.num_classes)
                    else:
                        self.fc = nn.Linear(384 * self.block_expansion, self.num_classes)
                        print("custom_vit")
                    

        elif opts.train.mode == 'openmany':
            self.cel_all = nn.CrossEntropyLoss()
            block = BasicBlock
            # self.inplanes = 512 * self.block_expansion
            # self.layer_sigs_0 = self._make_layer(block, 512, 2, stride=1)
            # self.inplanes = 512 * block.expansion * 2
            # self.layer_sigs_1 = self._make_layer(block, 512, 2, stride=1)

        elif opts.train.mode == 'regular':
            self.cel_all = nn.CrossEntropyLoss()
            # self.fc = nn.Linear(512 * self.block_expansion, self.num_classes)
        else:
            raise NameError('Unknown mode ({})!'.format(opts.train.mode))

    def forward(self, batch, opts_runtime, train=True):

        if self.opts.train.mode == 'openmany':
            return self.forward_openmany(batch, opts_runtime, train)
        if self.opts.train.mode == 'openfew':
            return self.forward_openfew(batch, opts_runtime, train)
        if self.opts.train.mode == 'regular':
            return self.forward_regular(batch, train)

    def forward_openfew(self, batch, opts_runtime, train=True):
        # print("batch[0]:", batch[0].size())
            # batch[0]: torch.Size([250, 3, 224, 224])
        # print("batch[1]:", batch[1].size())
            # batch[1]: torch.Size([250])
        """
        学習時
        print("batch[0]:", batch[0].size())
            # batch[0]: torch.Size([250, 3, 224, 224])
        print("batch[1]:", batch[1].size())
            # batch[1]: torch.Size([250])
        
        batch[0]にはバッチに分けられた250枚の画像が入っている
            - 入力チャンネル: 3
            - 画像サイズ: 224 x 224
        batch[1]にはバッチ内の画像に対応するクラスラベルが250個入っている
        
        評価時
        print("batch[0]:", batch[0].size())
            # batch[0]: torch.Size([55, 10, 3, 224, 224])
        print("batch[1]:", batch[1].size())
            # batch[1]: torch.Size([55])
        
        batch[0]にはバッチに分けられた55*10枚の画像が入っている
            - 入力チャンネル: 3
            - 画像サイズ: 224 x 224
        batch[1]にはバッチ内の画像に対応するクラスラベルが55個入っている
        wcs_infrared.pyではT.TenCropによって画像1枚当たりを10枚に拡張するようにクロッピングされている
        したがって、実際は55枚の画像が入力されている
        """

        # batchから入力画像(input)とGT(target)を取得し、
        # それぞれをself.opts.ctrl.deviceで指定されたデバイス（CPUやGPUなど）に転送しています。
        # これは、計算を行うデバイスにデータを移すための操作です。
        input, target = batch[0].to(self.opts.ctrl.device), batch[1].to(self.opts.ctrl.device)
        if len(input.size()) > 4:
            # 入力データの最後の3次元を抽出しています。これらは通常、チャネル数(c)、高さ(h)、幅(w)を表します。
            c = input.size(-3)
            h = input.size(-2)
            w = input.size(-1)
            # (任意のバッチサイズ, チャネル数, 高さ, 幅)の形状に変更
            # -1は指定された他の次元に合わせて残りの次元数を自動的に調整することを意味
            input = input.view(-1, c, h, w)
        # print("input:", input.size())
        # print("target:", target)
        # 5
        n_way = opts_runtime.n_way
        # 1
        k_shot = opts_runtime.k_shot
        # 15
        m_query = opts_runtime.m_query
        # 5
        open_cls = opts_runtime.open_cls
        1
        open_shot = opts_runtime.open_shot
        # 15
        open_sample = opts_runtime.open_sample
        # 1
        aug_scale = opts_runtime.aug_scale
        # print('aug_scale1', aug_scale)
        # データの分割数を意味している
        # 1
        fold = opts_runtime.fold

        # 5
        closed_s_amount = n_way * k_shot
        # 75
        closed_q_amount = int(n_way * m_query / fold)
        # 50
        open_s_amount = open_cls * open_shot
        # 75
        open_q_amount = int(open_cls * open_sample / fold)
        # 55
        support_amount = closed_s_amount + open_s_amount

        features = torch.zeros(self.num_classes, 192, device=self.opts.ctrl.device) # shape=(image_number*100,64)
        total_id = torch.zeros(self.num_classes, dtype=torch.int32, device=self.opts.ctrl.device) # group number of the image

        # FEATURE EXTRACTION
        # x_allは特徴抽出した結果のすべて
        # print('input', target)
        x_all = self.feat_net(input)
        if self.opts.train.open_detect == 'center':
            # 特徴量を格納
            x_mu = x_all
            # print('x_mu', x_mu.size())
            # x_all torch.Size([1550, 192])
            # 特徴量: 1550 = {support-set(5) + query-set(75) + open-set(75)}*T.TenCrop(aug_scale; 10)
            # 特徴次元: 192次元
        else:
            # x_muに特徴量を格納
            x_mu = x_all[0]
            # print("x_mu", x_mu.size())
            # x_sigsは意味ベクトル or 標準偏差のこと？
            x_sigs = x_all[1]
            # print("x_sigs", x_sigs.size())

        # LOSS OR ACCURACY
        if train:
            # TRAIN
            # 重複を削除する（サポート、クエリ、オープンから）
            target_unique, target_fsl = self.get_unique_target(target[:closed_s_amount+closed_q_amount])
            # # [1,2,4,3,0]
            target_closed_s = target_fsl[:closed_s_amount]
            # # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            # # 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
            # # 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # # 0, 0, 0]
            target_closed_q = target_fsl[closed_s_amount:closed_s_amount+closed_q_amount]
            # print("target_fsl:", target_fsl)
            # print("target_closed_s:", target_closed_s)
            # print("target_closed_q:", target_closed_q.size())
            if self.opts.model.structure == "resnet":
                support_mu = x_mu[:closed_s_amount, :, :, :]
                query_mu = x_mu[closed_s_amount:closed_s_amount+closed_q_amount+open_q_amount, :, :, :]
                batch_size, feat_size, feat_h, feat_w = query_mu.size()
                # 5クラス分繰り返す, 要素インデックスを含む
                idxs = torch.stack([(target_closed_s == i).nonzero().squeeze() for i in range(self.opts.fsl.n_way)])
                if len(idxs.size()) < 2:
                    idxs.unsqueeze_(1)
                # サポート特徴量の平均を計算する
                mu = support_mu[idxs, :, :, :].mean(dim=1)
                # print("mu:",mu.size())
                # print("query_mu_whitten1:",query_mu.size())
                # default center
                if self.opts.train.open_detect == 'center':
                    closed_s_mu_whitten = mu
                    query_mu_whitten = query_mu
                    # print("closed_s_mu_whitten:",closed_s_mu_whitten.size())
                    # print("query_mu_whitten1:",query_mu_whitten.size())
                else:
                    sigs = support_sigs[idxs, :, :, :].mean(dim=1)
                    closed_s_mu_whitten = torch.mul(mu, sigs)
                    query_mu_whitten = torch.mul(query_mu.unsqueeze(1), sigs.unsqueeze(0))
                # 操作後：[5, 512, 1, 1]
                # サポートの特徴量をプーリングした。1x1になった。
                # print("mu_whitten1",closed_s_mu_whitten.size())
                # closed_s_mu_whitten = self.avgpool(closed_s_mu_whitten)
                # print("mu_whitten2",closed_s_mu_whitten.size())
                # print("closed_s_mu_whitten:",closed_s_mu_whitten.size())
                # [5, 512]
                # その形を整えた。
                closed_s_mu_whitten = closed_s_mu_whitten.view(-1, feat_size)
                # print("mu_whitten3",closed_s_mu_whitten.size())
                # print("closed_s_mu_whitten:",closed_s_mu_whitten.size())
                # print("query_mu_whitten1.5:",query_mu_whitten.view(-1, feat_size, feat_h, feat_w).size())
                # print("query_mu_whitten1",query_mu_whitten.size())
                # query_mu_whitten = self.avgpool(query_mu_whitten.view(-1, feat_size, feat_h, feat_w))
                # print("query_mu_whitten2",query_mu_whitten.size())
                # print("query_mu_whitten2:",query_mu_whitten.size())
                query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
                # print("query_mu_whitten3",query_mu_whitten.size())
            elif self.opts.model.structure == "vit":
                # closed_s_mu_whittenはサポートセットの特徴量を格納
                closed_s_mu_whitten = x_all[:closed_s_amount :]
                # print("mu_whitten1",closed_s_mu_whitten.size())

                # 各クラスに対応するサポートセットのインデックスを取得
                idxs = torch.stack([(target_closed_s == i).nonzero().squeeze() for i in range(self.opts.fsl.n_way)])
                # print("idx:",idxs)
                if len(idxs.size()) < 2:
                    idxs.unsqueeze_(1)
                    # print("idx2:",idxs)
                # 各クラスのプロトタイプを計算
                closed_s_mu_whitten = closed_s_mu_whitten[idxs, :].mean(dim=1)
                # print("mu_whitten2",closed_s_mu_whitten.size())

                # クエリセットとオープンセットの特徴量を格納
                query_mu_whitten = x_all[closed_s_amount:closed_s_amount+closed_q_amount+open_q_amount,:]
                # print("query_mu_whitten1",query_mu_whitten.size())

                batch_size, feat_size = query_mu_whitten.size()

            if self.args.distance == "euclidean":
                # tensorのサイズを(batch_sizew, -1, feat_size)に変更
                # -1は動的に決定
                query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
                # print("query_mu_whitten",query_mu_whitten.size())
                # print("mu_whitten2.5",closed_s_mu_whitten.unsqueeze(0).size())
                # 損失をマイナスではなく、そのまま入力してみる → 意味ない
                dist_few = -torch.norm(query_mu_whitten - closed_s_mu_whitten.unsqueeze(0), p=2, dim=2)
                # dist_few = torch.norm(query_mu_whitten - closed_s_mu_whitten.unsqueeze(0), p=2, dim=2)
                # print("dist_few_size:",dist_few.size())
                # print("dist_few:",dist_few)
            elif self.args.distance == "cosine":
                query_mu_whitten = query_mu_whitten.view(-1, batch_size, feat_size)
                
                # dist_fewは各クラスの推論確率
                dist_few = self.cos_classifier(self.args, closed_s_mu_whitten.unsqueeze(0), query_mu_whitten)
                dist_few = dist_few.squeeze(0)
                # print("dist_few_size:",dist_few.size())
                # print("dist_few:",dist_few)

            # ここから変更
            # query_mu_whitten = query_mu_whitten.view(-1, batch_size, feat_size)

            # query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
            # print("query_mu_whitten2",query_mu_whitten.size())


            # dist_few = self.cos_classifier(self.args, closed_s_mu_whitten.unsqueeze(0), query_mu_whitten)
            # dist_few = -torch.norm(query_mu_whitten - closed_s_mu_whitten.unsqueeze(0), p=2, dim=2)
            # 追加
            # dist_few = dist_few.squeeze(0)
            
            ## FSL損失
            # クエリセットのみの距離を格納
            dist_few_few = dist_few[:closed_q_amount, :]
            # print("dist_few_few",dist_few_few)
            # クロスエントロピー損失関数（cel_all）
            # クエリセットの予測結果をGTと比較して損失をとっている
            l_few = self.cel_all(dist_few_few, target_closed_q)
            # openfew loss
            # エントロピーコスト（entropy cost）を計算します。エントロピーコストは、分類の確信度や不確実性を測定するために使用される
            # default: true

            ## OSR損失
            if self.opts.train.entropy:
                # OpenSetクラスのサンプル間の距離を表す
                dist_few_open = dist_few[closed_q_amount:closed_q_amount+open_q_amount, :]
                # 各要素に対するソフトマックスと対数ソフトマックスを計算
                # 各サンプルのクラス確率分布と対数確率分布が計算される
                # クラス確率分布と対数確率分布を要素ごとの積をとり、各サンプルの各クラスに対するエントロピーコストを計算
                loss_open = F.softmax(dist_few_open, dim=1) * F.log_softmax(dist_few_open, dim=1)
                # 列に沿って(横方向に)足し合わせることによって、各サンプルに対するエントロピーコストを計算
                loss_open = loss_open.sum(dim=1)
                # エントロピーコストの平均を計算
                l_open = loss_open.mean()
            # 使わない場合０を代入
            else:
                l_open = torch.tensor([0])
            ## 分類損失
            # defaulでtrue
            # 補助タスクに関する損失計算
            if self.opts.model.structure == "resnet":
                target_base = target[closed_s_amount+closed_q_amount+open_q_amount:]
                # 補助タスクの入力として使用するサンプルの特徴量を抽出し、base_mu に格納
                base_mu = x_mu[closed_s_amount+closed_q_amount+open_q_amount:, :, :, :]
                if self.opts.train.aux:
                    # print("base_mu1",base_mu.size())
                    # 平均プーリングを実行し、特徴量の平均値を計算
                    dist_base = self.avgpool(base_mu)
                    # print("dist_base1",dist_base.size())
                    # テンソルの形状を変更
                    dist_base = dist_base.view(-1, feat_size)
                    # print("dist_base2",dist_base.size())

                    # ニューラルネットワークの全結合層 self.fc を使用して、補助タスクの予測を行います。
                    cls_pred = self.fc(dist_base)
                    # print("cls_pred",cls_pred.size())
                    # print("target_base",target_base)
                    # setを使って重複を除去し、ユニークな要素だけを取得
                    unique_numbers = set(target_base.tolist())

                    # ユニークな要素の数を取得
                    unique_count = len(unique_numbers)
                    # 44
                    # print(unique_count)
                    # クロスエントロピー損失関数（cel_all）を使用して、補助タスクの損失 l_aux を計算
                    l_aux = self.cel_all(cls_pred, target_base)
                else:
                    l_aux = torch.tensor([0])
            elif self.opts.model.structure == "vit":
                target_base = target[closed_s_amount+closed_q_amount+open_q_amount:]
                # 補助タスクの入力として使用するサンプルの特徴量を抽出し、base_mu に格納
                # 補助タスクとは分類損失をとるために、support-set, query-set, open-setを用いて、分類問題を解くタスク
                # 実質dist_baseとして扱える
                base_mu = x_mu[closed_s_amount+closed_q_amount+open_q_amount:,:]
                # print("base_mu1",base_mu.size())
                base_size, feat_size = base_mu.size()
                # print("base_mu",base_mu.size())
                # print("base_mu", base_mu)
                if self.opts.train.aux:
                    # ニューラルネットワークの全結合層 self.fc を使用して、補助タスクの予測を行います。
                    # ViTで特徴抽出 -> 全結合層を用いて予測を行う
                    cls_pred = self.fc(base_mu)
                    # print("cls_pred",cls_pred.size())

                    # クロスエントロピー損失関数（cel_all）を使用して、補助タスクの損失 l_aux を計算
                    l_aux = self.cel_all(cls_pred, target_base)
                else:
                    l_aux = torch.tensor([0])
            if self.opts.train.entropy and self.opts.train.aux:
                loss = l_few + l_open * self.opts.train.loss_scale_entropy + l_aux * self.opts.train.loss_scale_aux
                log_loss = {'l_few': l_few, f'l_open({self.opts.train.loss_scale_entropy})': l_open * self.opts.train.loss_scale_entropy, 'raw_open': l_open, f'l_aux({self.opts.train.loss_scale_aux})': l_aux * self.opts.train.loss_scale_aux, 'raw_aux': l_aux}
            elif self.opts.train.entropy:
                loss = l_few + l_open * self.opts.train.loss_scale_entropy
                log_loss = {'l_few': l_few, f'l_open({self.opts.train.loss_scale_entropy})': l_open * self.opts.train.loss_scale_entropy, 'l_open': l_open}
            elif self.opts.train.aux:
                loss = l_few + l_aux * self.opts.train.loss_scale_aux
                log_loss = {'l_few': l_few, f'l_aux({self.opts.train.loss_scale_aux})': l_aux * self.opts.train.loss_scale_aux, 'raw_aux': l_aux}
            else:
                loss = l_few
                log_loss = {'l_few': l_few}

            return loss, base_mu, log_loss

        else:
            # TEST
            # 存在するカテゴリ数の範囲内でラベルを再割り当てする
            # ex) [9, 3, 4, 5, 7] => [4, 0, 1, 2, 3]
            # target_fsl_unique: 返還前のラベルが格納されている
            # target_fsl: 変換後のラベルが格納されている
            target_closed = torch.cat((target[:closed_s_amount], target[support_amount:support_amount+closed_q_amount]), dim=0)
            print('target_closed_s', target[:closed_s_amount])
            target_fsl_unique, target_fsl = self.get_unique_target(target_closed)
            print('target_fsl_unique', target_fsl_unique)
            # クローズドサポートセットのGT
            target_closed_s = target_fsl[:closed_s_amount]
            print('target_closed_s', target_closed_s)
            # クローズドクエリセットのGT
            target_closed_q = target_fsl[closed_s_amount:closed_s_amount+closed_q_amount]
            # print('target_closed_q', target_closed_q)
            # print('target_fsl_unique', target_fsl_unique)

            target_open = torch.cat((target[closed_s_amount:support_amount], target[support_amount+closed_q_amount:]), dim=0)
            target_osr_unique, target_osr = self.get_unique_target(target_open)
            print('target_osr_unique', target_osr_unique)
            target_osr = target_osr + n_way
            target_open_s = target_osr[:open_s_amount]
            print('target_open_s', target_open_s)
            target_open_q = target_osr[open_s_amount:open_s_amount+open_q_amount]

            # target_unique, target_trans = self.get_unique_target(target)
            # target_closed_s = target_trans[:closed_s_amount]
            # print('target_closed_s', target_closed_s)
            # target_closed_q = target_trans[support_amount:support_amount+closed_q_amount]
            # target_open_s = target_trans[closed_s_amount:support_amount]
            # print('target_open_s', target_open_s)
            # target_open_q = target_trans[support_amount:support_amount+closed_q_amount+open_q_amount]

            if self.opts.model.structure == "resnet":
                support_mu = x_mu[:closed_s_amount*aug_scale, :, :, :]
                query_mu = x_mu[closed_s_amount*aug_scale:(closed_s_amount+closed_q_amount+open_q_amount)*aug_scale, :, :, :]
                # print("support_mu1:",support_mu.size())
                # print("query_mu1:",query_mu.size())
                batch_size, feat_size, feat_h, feat_w = query_mu.size()

                # fewshot
                # print("support_mu1.5:",support_mu.view(-1, aug_scale, feat_size, feat_h, feat_w).size())
                support_mu = support_mu.view(-1, aug_scale, feat_size, feat_h, feat_w).mean(dim=1)
                # print("support_mu2:",support_mu.size())

                if self.opts.train.open_detect == 'gauss':
                    support_sigs = support_sigs.view(-1, aug_scale, feat_size, feat_h, feat_w).mean(dim=1)
                idxs = torch.stack([(target_closed_s == i).nonzero().squeeze() for i in range(n_way)])
                # print("idx:",idxs)
                if len(idxs.size()) < 2:
                    idxs.unsqueeze_(1)
                    # print("idx2:",idxs)
                mu = support_mu[idxs, :, :, :].mean(dim=1)
                # print("mu:",mu.size())
                if self.opts.train.open_detect == 'center':
                    closed_s_mu_whitten = mu
                    query_mu_whitten = query_mu
                else:
                    sigs = support_sigs[idxs, :, :, :].mean(dim=1)
                    closed_s_mu_whitten = torch.mul(mu, sigs)
                    query_mu_whitten = torch.mul(query_mu.unsqueeze(1), sigs.unsqueeze(0))
                # print("aug_scale:",aug_scale)
                # print("mu_whitten1:",closed_s_mu_whitten.size())
                # closed_s_mu_whitten = self.avgpool(closed_s_mu_whitten)
                # print("mu_whitten2:",closed_s_mu_whitten.size())
                closed_s_mu_whitten = closed_s_mu_whitten.view(-1, feat_size)
                # print("mu_whitten3:",closed_s_mu_whitten.size())

                # print("query_mu_whitten1:",query_mu_whitten.size())
                # print("query_mu_whitten1.5:",query_mu_whitten.view(-1, feat_size, feat_h, feat_w).size())
                # query_mu_whitten = self.avgpool(query_mu_whitten.view(-1, feat_size, feat_h, feat_w))
                # print("query_mu_whitten2:",query_mu_whitten.size())
                query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
                # print("query_mu_whitten3:",query_mu_whitten.size())
            elif self.opts.model.structure == "vit":
            # prepare class gauss
            # support_mu = x_mu[:closed_s_amount*aug_scale, :, :, :]
            # query_mu = x_mu[closed_s_amount*aug_scale:(closed_s_amount+closed_q_amount+open_q_amount)*aug_scale, :, :, :]
            # if self.opts.train.open_detect == 'gauss':
            #     support_sigs_0 = x_sigs[:closed_s_amount*aug_scale, :, :, :]
            #     support_sigs_1 = self.layer_sigs_0(support_sigs_0).mean(dim=0, keepdim=True).expand_as(support_sigs_0)
            #     support_sigs_1 = torch.cat((support_sigs_0, support_sigs_1), dim=1)
            #     support_sigs = self.layer_sigs_1(support_sigs_1)
            
            # batch_size, feat_size, feat_h, feat_w = query_mu.size()

            ## 各setの特徴量を格納
                # query_mu_whitten: closed-query setとopen-query setの特徴抽出結果を格納している
                query_mu_whitten = x_all[support_amount*aug_scale:(support_amount+closed_q_amount+open_q_amount)*aug_scale,:]
                # print("query_mu_whitten1:",query_mu_whitten.size())
                # output>> query_mu_whitten1: torch.Size([1500, 192])
                # batchサイズ: 1500枚（closed-query set: 75*10, open-query set: 75*10）
                # 特徴次元: 192次元
                batch_size, feat_size = query_mu_whitten.size()
                # print('aug_scale2', aug_scale)

                ## closed-query setの特徴量を取得
                closed_q_mu_whitten = x_all[support_amount*aug_scale:(support_amount+closed_q_amount)*aug_scale, :]
                closed_q_mu_whitten = closed_q_mu_whitten.view(-1,aug_scale, feat_size).mean(dim=1)
                # print('closed_q_mu_whitten', closed_q_mu_whitten.size())
                # output>> closed_q_mu_whitten torch.Size([75, 192])

                ## open-query setの特徴量を取得
                open_q_mu_whitten = x_all[(support_amount+closed_q_amount)*aug_scale: , :]
                # print('open_q_mu_whitten1', open_q_mu_whitten.size())
                # output>> open_q_mu_whitten1 torch.Size([750, 192])
                open_q_mu_whitten = open_q_mu_whitten.view(-1,aug_scale, feat_size).mean(dim=1)
                # print('open_q_mu_whitten2', open_q_mu_whitten.size())
                # output>> open_q_mu_whitten2 torch.Size([75, 192])
                # print('open_q_mu_whitten2', open_q_mu_whitten)

                ## closed-support setの特徴量を取得
                # T.TenCrop( )によって拡張した分の画像を1つにまとめる
                # それぞれの画像に対して平均をとり、1つの特徴量として扱う
                # 5*10 => 5
                closed_s_mu_whitten = x_all[:closed_s_amount*aug_scale, :]
                # print("closed_s_mu_whitten1:",closed_s_mu_whitten.size())
                # output>> closed_s_mu_whitten1: torch.Size([50, 192])
                # print("closed_s_mu_whitten1.5:", closed_s_mu_whitten.view(-1,aug_scale, feat_size).size())
                # output>> closed_s_mu_whitten1.5: torch.Size([5, 10, 192])
                closed_s_mu_whitten = closed_s_mu_whitten.view(-1,aug_scale, feat_size).mean(dim=1)
                # print("closed_s_mu_whitten2:", closed_s_mu_whitten.size())
                # 上2つをまとめた書き方
                # closed_s_mu_whitten = x_all[:closed_s_amount*aug_scale, :].view(-1,aug_scale, feat_size).mean(dim=1)

                ## open-support setの特徴量を取得
                open_s_mu_whitten = x_all[closed_s_amount*aug_scale:support_amount*aug_scale, :]
                # print("open_s_mu_whitten1:", open_s_mu_whitten.size())
                # output>> open_s_mu_whitten1: torch.Size([500, 192])
                open_s_mu_whitten = open_s_mu_whitten.view(-1,aug_scale, feat_size).mean(dim=1)
                # print("open_s_mu_whitten2:", open_s_mu_whitten.size())
                # output>> open_s_mu_whitten2: torch.Size([50, 192])

            ## prototypeの作成
                ## closed-support set
                # 各クラスに属するサポートセットのインデックスを取得
                idxs = torch.stack([(target_closed_s == i).nonzero().squeeze() for i in range(n_way)])
                print("closed_idx:",idxs)
                # output>> closed_idx: tensor([2, 0, 4, 1, 3], device='cuda:0')
                # インデックスが1次元の場合に次元を調整する
                if len(idxs.size()) < 2:
                    # 次元を追加
                    idxs.unsqueeze_(1)
                    # print("idx2:",idxs)
                    """
                    idx2: tensor([[2],
                    [0],
                    [4],
                    [1],
                    [3]], device='cuda:0')
                    """
                # 各クラスの特徴量の平均をとり、代表点（プロトタイプ）を取得
                # 入力画像のラベルの大小関係にしたがって、プロトタイプのソートも行わている
                # すなわち、closed_prototypeのdim 0のインデックスはラベルに一致
                closed_prototype = closed_s_mu_whitten[idxs, :].mean(dim=1)
                # print("closed_prototype:", closed_prototype.size())
                # output>> closed_prototype: torch.Size([5, 192])

                ## open-support set
                idxs = torch.stack([(target_open_s == i).nonzero().squeeze() for i in range(n_way, n_way+open_cls)])
                print("open_idx:",idxs)
                if len(idxs.size()) < 2:
                    idxs.unsqueeze_(1)
                open_prototype = open_s_mu_whitten[idxs, :].mean(dim=1)
                # print("open_prototype:", open_prototype.size())
                # output>> open_prototype: torch.Size([5, 192])

                ## closed_prototype と open_prototype を結合
                prototype = torch.cat((closed_prototype, open_prototype), dim=0)
                # print('prototype:', prototype.size())
                # prototype: torch.Size([10, 192])
                # この場合dim 0において[0:5]はclosed-prototype, [5:10]はopen-prototype

            if self.args.distance == "euclidean":
                query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
                # print("query_mu_whitten2:",query_mu_whitten.size())
                # output>> query_mu_whitten2: torch.Size([1500, 1, 192])
                # 距離を求める
                # print('prototype.unsqueeze(0):', prototype.unsqueeze(0).size)
                # output>> prototype.unsqueeze(0): torch.Size([1, 10, 192])
                # dist_few: ユークリッド距離の計算結果を格納
                dist_few = torch.norm(query_mu_whitten - prototype.unsqueeze(0), p=2, dim=2)
                # print('dist_few1:', dist_few.size())
                # output>> dist_few1: torch.Size([1500, 10])
                # 150*10 => 150(query-set; 75. open-set: 75)

            elif self.args.distance == "cosine":
                query_mu_whitten = query_mu_whitten.view(-1, batch_size, feat_size)
                dist_few = self.cos_classifier(self.args, closed_s_mu_whitten.unsqueeze(0), query_mu_whitten)
                dist_few = dist_few.squeeze(0)

            # aug_scale分（T.TenCrop()で拡張した分）の距離の平均を計算
            dist_few = dist_few.view(-1, aug_scale, n_way+open_cls).mean(dim=1)
            # print('dist_few2:', dist_few.size())
            # output>> dist_few2: torch.Size([150, 10])

            # 求めた各クエリの距離に対してソフトマックス関数を適用し、各クラスに対する確率を求める
            dist_few_sm = self.sm(dist_few)
            # print("dist_few_sm",dist_few_sm.size())
            # dist_few_sm torch.Size([150, 10])
            # print("dist_few_sm",dist_few_sm)
            # 予測確率と予測ラベルをそれぞれall_score, all_predに格納
            if self.args.distance == "euclidean":
                all_score, all_pred = dist_few_sm.min(dim=1)
            elif self.args.distance == "cosine":
                all_score, all_pred = dist_few_sm.max(dim=1)
            # print("all_score",all_score.size())
            # all_score torch.Size([150])
            # print("all_pred",all_pred.size())
            # all_pred torch.Size([150])
            print("all_pred",all_pred)

            # 全予測結果からクエリセットのみの予測結果を抽出
            closed_pred = all_pred[:closed_q_amount]
            # print('closed_pred', closed_pred)
            open_pred = all_pred[closed_q_amount:closed_q_amount+open_q_amount]
            # print('open_pred', open_pred)
            closed_q = torch.ones(closed_q_amount)
            # print("closed_few:",closed_few)
            open_q = -torch.ones(open_q_amount)
            # print("closed_open:",closed_open)
            # closed-setかopen-setかのGT
            target_osr = torch.cat((closed_q, open_q), dim=0)
            return closed_pred.detach().cpu(), open_pred.detach().cpu(), all_pred.detach().cpu(), \
                   target_closed_q.detach().cpu(), target_open_q.detach().cpu(), all_score.detach().cpu(), \
                   target_fsl_unique.detach().cpu(), closed_q_mu_whitten.detach().cpu(), \
                   open_q_mu_whitten.detach().cpu(), target_osr, dist_few_sm.detach().cpu()

    def forward_regular(self, batch, train=True):
        input, target = batch[0].to(self.opts.ctrl.device), batch[1].to(self.opts.ctrl.device)

        if len(input.size()) > 4:
            # 入力データの最後の3次元を抽出しています。これらは通常、チャネル数(c)、高さ(h)、幅(w)を表します。
            c = input.size(-3)
            h = input.size(-2)
            w = input.size(-1)
            # (任意のバッチサイズ, チャネル数, 高さ, 幅)の形状に変更
            # -1は指定された他の次元に合わせて残りの次元数を自動的に調整することを意味
            input = input.view(-1, c, h, w)
        # print("input:", input.size())
        # print("target:", target)
        # test時はTenCrop()によって10枚に拡張している
        # 10
        # aug_scale = opts_runtime.aug_scale
        # print('aug_scale1', aug_scale)
        aug_scale = 10

        # FEATURE EXTRACTION
        # x_allは特徴抽出した結果のすべて
        x_all = self.feat_net(input)
        if self.opts.train.open_detect == 'center':
            # 特徴量を格納
            x_mu = x_all
            # print('x_mu', x_mu.size())
            # x_all torch.Size([1550, 192])
            # 特徴量: 1550 = {support-set(5) + query-set(75) + open-set(75)}*T.TenCrop(aug_scale; 10)
            # 特徴次元: 192次元
        else:
            # x_muに特徴量を格納
            x_mu = x_all[0]
            # print("x_mu", x_mu.size())
            # x_sigsは意味ベクトル or 標準偏差のこと？
            x_sigs = x_all[1]
            # print("x_sigs", x_sigs.size())
        if train:
            print('train mode')
        else :
            # TEST
            if self.opts.model.structure == "resnet":
                support_mu = x_mu[:closed_s_amount*aug_scale, :, :, :]
                query_mu = x_mu[closed_s_amount*aug_scale:(closed_s_amount+closed_q_amount+open_q_amount)*aug_scale, :, :, :]
                # print("support_mu1:",support_mu.size())
                # print("query_mu1:",query_mu.size())
                batch_size, feat_size, feat_h, feat_w = query_mu.size()

                # fewshot
                # print("support_mu1.5:",support_mu.view(-1, aug_scale, feat_size, feat_h, feat_w).size())
                support_mu = support_mu.view(-1, aug_scale, feat_size, feat_h, feat_w).mean(dim=1)
                # print("support_mu2:",support_mu.size())

                if self.opts.train.open_detect == 'gauss':
                    support_sigs = support_sigs.view(-1, aug_scale, feat_size, feat_h, feat_w).mean(dim=1)
                idxs = torch.stack([(target_closed_s == i).nonzero().squeeze() for i in range(n_way)])
                # print("idx:",idxs)
                if len(idxs.size()) < 2:
                    idxs.unsqueeze_(1)
                    # print("idx2:",idxs)
                mu = support_mu[idxs, :, :, :].mean(dim=1)
                # print("mu:",mu.size())
                if self.opts.train.open_detect == 'center':
                    closed_s_mu_whitten = mu
                    query_mu_whitten = query_mu
                else:
                    sigs = support_sigs[idxs, :, :, :].mean(dim=1)
                    closed_s_mu_whitten = torch.mul(mu, sigs)
                    query_mu_whitten = torch.mul(query_mu.unsqueeze(1), sigs.unsqueeze(0))
                # print("aug_scale:",aug_scale)
                # print("mu_whitten1:",closed_s_mu_whitten.size())
                # closed_s_mu_whitten = self.avgpool(closed_s_mu_whitten)
                # print("mu_whitten2:",closed_s_mu_whitten.size())
                closed_s_mu_whitten = closed_s_mu_whitten.view(-1, feat_size)
                # print("mu_whitten3:",closed_s_mu_whitten.size())

                # print("query_mu_whitten1:",query_mu_whitten.size())
                # print("query_mu_whitten1.5:",query_mu_whitten.view(-1, feat_size, feat_h, feat_w).size())
                # query_mu_whitten = self.avgpool(query_mu_whitten.view(-1, feat_size, feat_h, feat_w))
                # print("query_mu_whitten2:",query_mu_whitten.size())
                query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
                # print("query_mu_whitten3:",query_mu_whitten.size())
            elif self.opts.model.structure == "vit":
                
                batch_size, feat_size = x_all.size()
                x_all = x_all.view(-1,aug_scale, feat_size).mean(dim=1)
                print('feature', x_all)
                print('feature size', x_all.size())
                # x_all_sm = self.sm(x_all)

            return x_all.detach().cpu()

        """ # FEATURE EXTRACTION
        x = self.feat_net(input)
        x = self.avgpool(x)
        N = x.size(0)
        x = x.view(N, -1)
        print(x.size())

        # LOSS OR ACCURACY
        if train:
            cls_pred = self.fc(x)
            loss = self.cel_all(cls_pred, target)
            return loss
        else:
            # TEST
            cls_pred = self.fc(x)
            _, pred_cls = cls_pred.max(dim=1)
            correct = torch.eq(pred_cls, target)
            return correct.sum().item() """

    # f の各データポイントがどのクラス（w に表される）に最も類似しているかを評価
    def cos_classifier(self, args, w, f):
        """
        w.shape = B, nC, d
        wは学習によって得た各クラスの代表点
        f.shape = B, M, d
        fは入力画像の特徴ベクトル
        """
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim()-1, eps=1e-12)
        # print("f",f.size()),
        # print("w",w.size()),
        # f torch.Size([1, 150, 384])
        # w torch.Size([5, 384])
        cls_scores = f @ w.transpose(1, 2) # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    # 重複を削除する
    def get_unique_target(self, target):
        # サポートとクエリ、オープンのカテゴリから重複してないカテゴリを取得する
        unique_labels, target_transformation = target.unique(return_inverse=True)

        return unique_labels, target_transformation

    # # ニューラルネットワークモデル内で畳み込み層と正規化層を含むブロックを作成
    # def _make_layer(self, block, planes, blocks, stride=1):
    #     # 正規化層
    #     norm_layer = self._norm_layer
    #     # プーリング層
    #     downsample = None
    #     # ストライドが１以外かつ～の場合、プーリング層を設定する
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)