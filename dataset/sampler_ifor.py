import torch
import numpy as np


class MetaSampler_IFOR(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, opts_runtime, train=True):
        super(MetaSampler_IFOR, self).__init__(dataset)

        self.iterations = opts_runtime.iterations
        # 5
        self.n_way = opts_runtime.n_way
        # 1
        self.k_shot = opts_runtime.k_shot
        # 15
        self.m_query = opts_runtime.m_query
        # 75
        self.p_base = opts_runtime.p_base
        # 5
        self.open_cls = opts_runtime.open_cls
        # 15
        self.open_sample = opts_runtime.open_sample
        # 1
        self.fold = opts_runtime.fold
        self.train = train
        # train_test + testのサンプルリスト
        self.n_sample_list = dataset.n_sample_list
        # if train:
        if True:
            # データセットに含まれるクラス数(closed-set and open-set)
            # 11
            self.n_cls = dataset.cls_num
            self.base_cls = 0
            self.idx_list = []
            # このコードは、データセット内の各クラスに対して、どのサンプルがそのクラスに属しているかの
            # インデックス範囲を記録するために使用されると考えられます。
            # 例えば、self.n_sample_list が [3, 5, 2] であれば、最初のクラスには3つのサンプルがあり、
            # 次に5つのサンプルが別のクラスに属しており、そして最後に2つのサンプルが別のクラスに属している
            # ことを意味します。
            """
            以下のようにリストが追加されたself.idx_listが作成される
                [0, 1, 2, ..., 99] （クラス0用）
                [100, 101, 102, ..., 199] （クラス1用）
                [200, 201, 202, ..., 299] （クラス2用）
                    to be continued...
                [1000, 1001, 1002, ..., 1099] （クラス10用）
            """
            for i in range(self.n_cls):
                self.idx_list.append(np.arange(self.n_sample_list[:i].sum(), self.n_sample_list[:i+1].sum()))
        else:
            # オープンセットのクラス数？(valのカテゴリ数(16))
            self.n_cls = dataset.open_cls_num
            # クローズドセットのクラス数？(train_valのカテゴリ数)
            self.base_cls = dataset.cls_num
            self.idx_list = []
            for i in range(dataset.cls_num):
                self.idx_list.append(np.arange(self.n_sample_list[:i].sum(), self.n_sample_list[:i+1].sum()))

    def __iter__(self):
        for it in range(self.iterations):
            batch_s = torch.zeros(self.n_way * self.k_shot)
            batch_q = torch.zeros(self.n_way * self.m_query)
            batch_open = torch.zeros(self.open_cls * self.open_sample)
            # クラスをランダムに書き換える。0~n_cls(train(28), test(11))までの整数のリストをランダムに並び替える
            cls_all = torch.from_numpy(np.random.permutation(self.n_cls))
            # FSL用のクラスをn_way(5)選択
            cls_fsl = cls_all[:self.n_way]
            # OSR用のクラスをそれ以外からopen_cls(5)選択
            cls_open = cls_all[self.n_way: self.n_way + self.open_cls]
            # FSLのn_shot(1)を選択する。サポートとクエリーを取得し、バッチにして
            for c in range(self.n_way):
                # テスト時ならbaseの分も追加する。
                # n_sample_list: 各クラスに属するサンプルの数をリスト
                # cls_fsl: ランダムに選ばれた self.n_way のクラスを示すリスト
                # 現在のクラスのサンプル数を取得
                # item() メソッドでPythonの組み込み型(int)として取得する
                # test(100)
                n_sample = int(self.n_sample_list[cls_fsl[c]+self.base_cls].item())
                # クラス内画像をランダムにする
                # リストに16個（k_shot: 1, m_query: 15）の要素が格納される
                samples = np.random.permutation(n_sample)[:self.k_shot + self.m_query]
                # サポートをカテゴリーからk_shot選択
                supports = samples[:self.k_shot]
                # それ以外からk_shot選択
                querys = samples[self.k_shot:]
                # サポートセットとクエリセットのインデックスを格納
                batch_s[self.k_shot*c:self.k_shot*(c+1)] = torch.from_numpy(supports) + self.n_sample_list[:cls_fsl[c]+self.base_cls].sum()
                batch_q[self.m_query*c:self.m_query*(c+1)] = torch.from_numpy(querys) + self.n_sample_list[:cls_fsl[c]+self.base_cls].sum()
            # OSRの画像を選択する
            # オープンセットクラスの数（５）だけ繰り返す
            for c in range(self.open_cls):
                # オープンセットクラスからランダムに self.open_sample(15) 個のサンプルを選択
                n_sample = int(self.n_sample_list[cls_open[c]+self.base_cls].item())
                samples = np.random.permutation(n_sample)[:self.open_sample]
                batch_open[self.open_sample*c:self.open_sample*(c+1)] = torch.from_numpy(samples) + self.n_sample_list[:cls_open[c]+self.base_cls].sum()
            # idx_listを結合し、0 - 1100 まで格納した1次元配列にする
            idx_all = np.concatenate(self.idx_list)
            # idx_allをランダムにシャッフルする
            np.random.shuffle(idx_all)

            # 学習
            if self.train:
                # idx_allの最初から self.p_base(75)未満のインデックスを抽出
                batch_e = torch.from_numpy(idx_all[:self.p_base]).float()
                batch = torch.cat((batch_s, batch_q, batch_open, batch_e), dim=0).long().view(-1)
                yield batch
            # 評価
            else:
                if self.fold > 1:
                    fold_q = int(self.n_way * self.m_query / self.fold)
                    fold_open = int(self.open_cls * self.open_sample / self.fold)
                    for i in range(self.fold):
                        batch_q_fold = batch_q[fold_q*i:fold_q*(i+1)]
                        batch_open_fold = batch_open[fold_open*i:fold_open*(i+1)]
                        batch = torch.cat((batch_s, batch_q_fold, batch_open_fold), dim=0).long().view(-1)
                        yield batch
                else:
                    batch = torch.cat((batch_s, batch_q, batch_open), dim=0).long().view(-1)
                    yield batch

    def __len__(self):
        # 100回繰り返す。
        return self.iterations * self.fold
