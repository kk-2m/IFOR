import torch
import csv
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.svm import SVC
from sklearn import metrics       # 精度検証用

from core.general_utils import AttrDict


def roc_area_calc(args, dist, gt, descending, total_height, total_width):
    # _, p = dist.sort(descending=descending)
    # _, p = dist.sort(descending=False)
    if args.distance == "euclidean":
        # print("euclidian-distance")
        _, p = dist.sort(descending=True)

    elif args.distance == "cosine":
        # print("cosine-distance")
        _, p = dist.sort(descending=False)

    # 並べ替えられた順番に基づいたラベルを格納
    sorted_gt = gt[p]

    height = 0.0
    width = 0.0
    area = 0.0
    pre = 0  # (0: width; 1: height)

    for i in range(len(sorted_gt)):
        if sorted_gt[i] == -1:
            if pre == 0:
                area += height * width
                width = 0.0
                height += 1.0
                pre = 1
            else:
                height += 1.0
        else:
            pre = 0
            width += 1.0
    if pre == 0:
        area += height * width

    # 面積の正規化
    area = area / total_height / total_width
    return area

def to_device(obj, device, non_blocking=False):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking=non_blocking)
                for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_device(v, device, non_blocking=non_blocking)
                for v in obj]

    if isinstance(obj, tuple):
        return tuple([to_device(v, device, non_blocking=non_blocking)
                     for v in obj])

def update_loss_scale(opts, episode):
    # defaut 30000
    for i in range(len(opts.train.loss_scale_entropy_lut[0])):
        if episode < opts.train.loss_scale_entropy_lut[0][i]:
            # default 0.5
            opts.train.loss_scale_entropy = opts.train.loss_scale_entropy_lut[1][i]
            break
    # 30000
    for i in range(len(opts.train.loss_scale_aux_lut[0])):
        if episode < opts.train.loss_scale_aux_lut[0][i]:
            opts.train.loss_scale_aux = opts.train.loss_scale_aux_lut[1][i]
            break
    # 30000
    for i in range(len(opts.train.loss_scale_kmeans_lut[0])):
        if episode < opts.train.loss_scale_kemans_lut[0][i]:
            # default 0.1
            opts.train.loss_scale_kmeans = opts.train.loss_scale_kmeans_lut[1][i]
            break
    # 30000
    for i in range(len(opts.train.loss_scale_bc_lut[0])):
        if episode < opts.train.loss_scale_bc_lut[0][i]:
            # default 0.5
            opts.train.loss_scale_bc = opts.train.loss_scale_bc_lut[1][i]
            break


def save_model(opts, net, optimizer, scheduler, episode, save_file):
    file_to_save = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'opts': opts,
        'episode': episode
    }
    torch.save(file_to_save, save_file)


def create_runtime_opts(opts, split):
    if split == 'train':
        opts_runtime = AttrDict()
        # 5
        opts_runtime.n_way = opts.fsl.n_way
        # 1
        opts_runtime.k_shot = opts.fsl.k_shot
        # 15
        opts_runtime.m_query = opts.fsl.m_query
        # 75
        opts_runtime.p_base = opts.fsl.p_base
        # 100
        opts_runtime.iterations = opts.fsl.iterations
        # 5
        opts_runtime.open_cls = opts.open.n_cls
        # 1
        opts_runtime.open_shot = opts.open.k_shot
        # 15
        opts_runtime.open_sample = opts.open.m_sample
        opts_runtime.aug_scale = 1
        opts_runtime.fold = 1
    elif split == 'val':
        opts_runtime = AttrDict()
        opts_runtime.n_way = opts.fsl.n_way_val
        opts_runtime.k_shot = opts.fsl.k_shot_val
        opts_runtime.m_query = opts.fsl.m_query_val
        opts_runtime.p_base = 0
        opts_runtime.iterations = opts.fsl.iterations_val
        opts_runtime.open_cls = opts.open.n_cls_val
        opts_runtime.open_shot = opts.open.k_shot_val
        opts_runtime.open_sample = opts.open.m_sample_val
        opts_runtime.aug_scale = 1
        opts_runtime.fold = 1
    else:  # split == 'test'
        opts_runtime = AttrDict()
        # 5
        opts_runtime.n_way = opts.fsl.n_way_test
        # 1
        opts_runtime.k_shot = opts.fsl.k_shot_test
        # 15
        opts_runtime.m_query = opts.fsl.m_query_test
        opts_runtime.p_base = 0
        # 600
        opts_runtime.iterations = opts.fsl.iterations_test
        # 5
        opts_runtime.open_cls = opts.open.n_cls_test
        opts_runtime.open_shot = opts.open.k_shot_test
        # 15
        opts_runtime.open_sample = opts.open.m_sample_test
        # 1
        opts_runtime.aug_scale = opts.model.aug_scale_test
        # 1
        opts_runtime.fold = opts.model.fold_test

    return opts_runtime


def evaluation(args, net, input_db, mode, opts_eval):
    closed_s_amount = opts_eval.n_way * opts_eval.k_shot
    open_s_amount = opts_eval.open_cls * opts_eval.open_shot
    support_amount = closed_s_amount + open_s_amount
    open_q_amount = opts_eval.open_cls * opts_eval.open_sample

    net.eval()
    with torch.no_grad():
        total_counts = 0
        total_correct = 0.0

        closed_pred_list = []
        open_pred_list = []
        cq_label_list = []
        oq_label_list = []
        allq_label_list = []
        distance_score_list = []
        auroc_label_list = []
        classification_prob_list = []
        all_pred_list = []
        closed_q_feature_list = np.empty((0, 192))
        open_q_feature_list = np.empty((0, 192))
        
        closed_gt = []
        open_gt = []
        infe = []
        prob = []

        # print(len(input_db))
        # 1800
        for j, batch_test in enumerate(input_db):
            # print('batch_test: ', len(batch_test))
            # バッチサイズは1
            # 学習
            # output>> batch_test: 2
            # batch_test[0]: image
            # batch_test[1]: class label
            # print('batch test image: ', batch_test[0].size())
            # output>> batch test image:  torch.Size([75, 10, 3, 224, 224])
            # batch_test[0]: [サンプル数, カテゴリ数, チャンネル数, 画像の高さ, 画像の幅]
            # print('batch test label: ', batch_test[1].size())
            # batch test label:  torch.Size([75])

            # 評価
            # batch_test[0]: image
            # batch_test[1]: class label
            # print('batch test image: ', batch_test[0].size())
            # output>> batch test image:  torch.Size([55, 10, 3, 224, 224])
            # batch_test[0]: [サンプル数, T.TenCropによる拡張枚数, チャンネル数, 画像の高さ, 画像の幅]
            # print('batch test label: ', batch_test[1].size())
            # output>> batch test label:  torch.Size([55])

            # print('pre-aug_scale', opts_eval.aug_scale)
            if mode.startswith('open'):
                closed_pred, open_pred, all_pred, closed_q_label, open_q_label, distance_score, \
                    target_closed_unique, closed_q_feature, open_q_feature, auroc_label, classification_prob \
                        = net(batch_test, opts_eval, False)

                # closed-setに対する予測ラベル
                closed_pred_list.append(closed_pred)
                # print('closed_pred: ', closed_pred.size())
                # output>> closed_pred torch.Size([25])
                # print('closed_pred', closed_pred)

                # open-setに対する予測ラベル
                open_pred_list.append(open_pred)

                # closed-set, open-setに対する予測ラベル
                all_pred_list.append(all_pred)

                # closed-query setに対するGT (再割り当て後のラベル)
                cq_label_list.append(closed_q_label)
                # print('closed_q_label: ', closed_q_label.size())
                # output>> targtet torch.Size([25])
                # print('closed_q_label', closed_q_label)

                # open-query setに対するGT (再割り当て後のラベル)
                oq_label_list.append(open_q_label)

                # closed_query set, open-query setに対するGT (再割り当て後のラベル)
                allq_label_list.append(torch.cat([closed_q_label, open_q_label]))

                # query-set, open-setの+予測ラベルに対する距離 (最も小さいカテゴリが入力画像に対する予測ラベル)
                distance_score_list.append(distance_score)
                # print('distance_score: ', distance_score.size())
                # output>> distance_score torch.Size([50])
                # print('distance_score', distance_score)

                closed_q_feature_list = np.append(closed_q_feature_list, closed_q_feature.numpy(), axis=0)
                # print('query_feature', closed_q_feature.size())
                # print('query_feature', closed_q_feature)
                # output>> open_mu_whitten2 torch.Size([75, 192])

                # open-setの特徴量(192次元)
                # axis=0として、行方向にデータを追加（192次元の形状を維持）
                open_q_feature_list = np.append(open_q_feature_list, open_q_feature.numpy(), axis=0)
                # print('open_q_feature', open_q_feature.size())
                # output>> open_mu_whitten2 torch.Size([75, 192])
                # print('open_q_feature', open_q_feature_list)
                
                # closed-set(1), open-set(-1)のGT
                auroc_label_list.append(auroc_label)
                # print('closed: ', closed.size())
                # output>> closed torch.Size([50])
                # print('closed', closed)
                
                # 各エピソードの全カテゴリに対するquery-set, open-setの確率（classification probably）
                classification_prob_list.extend(classification_prob.tolist())
                # print('classification_prob', len(classification_prob))
                # print('classification_prob(query-set)', len(classification_prob[:len(closed_q_label)]))
                # output>> classification_prob 50
                # print('classification_prob', classification_prob)

                # Grand Truth(再割り当て前)
                closed_gt.extend((batch_test[1].tolist())[support_amount:-open_q_amount])
                # print('closed_q_gt', (batch_test[1].tolist())[support_amount:-open_q_amount])
                open_gt.extend((batch_test[1].tolist())[-open_q_amount:])
                # print('open_q_gt', (batch_test[1].tolist())[-open_q_amount:])

            elif mode == 'regular':
                feature = net(batch_test, opts_eval, False)
                # total_correct += correct
                # otalt_counts += batch_test[1].size(0)

                # Grand Truth(再割り当て前)
                closed_gt.extend(batch_test[1].tolist())
                print('closed_gt', batch_test[1].tolist())

                closed_q_feature_list = np.append(closed_q_feature_list, feature.numpy(), axis=0)
                # print('closed_q_feature', closed_q_feature.size())
                # print('closed_q_feature', closed_q_feature)
                # output>> closed_mu_whitten2 torch.Size([75, 192])

    # print('closed_pred_list', len(closed_pred_list))
    # output>> closed_pred_list 1800
    # print('cq_label_list', len(cq_label_list))
    # output>> cq_label_list 1800
    # print(cq_label_list[0])
    # output>> 
    #     tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    #     3]),
    #     tensor([3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    #     2])
    
    print('the length of closed_q_feature_list', len(closed_q_feature_list))
    print(closed_q_feature_list)
    # print('the length of open_q_feature_list', len(open_q_feature_list))
    # print(open_q_feature_list)
    for i in closed_gt:
        print(i)
        break
    # for i in open_gt:
    #     print(i)
    #     break

    net.train()

    if mode == 'openfew':
        closed_samples = opts_eval.n_way*opts_eval.m_query
        open_samples = opts_eval.open_cls*opts_eval.open_sample
        all_samples = opts_eval.n_way*opts_eval.m_query + opts_eval.open_cls*opts_eval.open_sample
        closed_all_pred = torch.cat(closed_pred_list, dim=0)
        open_all_pred = torch.cat(open_pred_list, dim=0)
        all_all_pred = torch.cat(all_pred_list, dim=0)
        cq_label_all = torch.cat(cq_label_list, dim=0)
        oq_label_all = torch.cat(oq_label_list, dim=0)
        allq_label_all = torch.cat(allq_label_list, dim=0)
        distance_all_score = torch.cat(distance_score_list, dim=0)
        auroc_all_label = torch.cat(auroc_label_list, dim=0)

        closed_accuracy = torch.eq(closed_all_pred, cq_label_all).sum().item() / closed_samples / opts_eval.iterations
        open_accuracy = torch.eq(open_all_pred, oq_label_all).sum().item() / open_samples / opts_eval.iterations
        all_accuracy = torch.eq(all_all_pred, allq_label_all).sum().item() / all_samples / opts_eval.iterations
        auroc_all = torch.zeros(opts_eval.iterations)
        for i in range(opts_eval.iterations):
            auroc_all[i] = roc_area_calc(args, dist=distance_all_score[(closed_samples+open_samples)*i:(closed_samples+open_samples)*(i+1)],
                                         gt=auroc_all_label[(closed_samples+open_samples)*i:(closed_samples+open_samples)*(i+1)],
                                         descending=True, total_height=open_samples, total_width=closed_samples)
        auroc = auroc_all.mean()
        
        return closed_accuracy, open_accuracy, all_accuracy, auroc
    elif mode == 'regular':
        # accuracy = total_correct / total_counts

        # k-means
        closed_true_labels = np.array(closed_gt)
        print('open_true_labels:', closed_true_labels.shape)
        print(closed_true_labels)
        pred_k_means = KMeans(n_clusters=11, random_state=0).fit_predict(closed_q_feature_list)
        # print(pred_k_means)
        # 調整Rand指数（ARI）の計算
        ari = adjusted_rand_score(closed_true_labels, pred_k_means)
        print("Adjusted Rand Index (ARI):", ari)
        # 相互情報量（NMI）の計算
        nmi = normalized_mutual_info_score(closed_true_labels, pred_k_means)
        print("Normalized Mutual Information (NMI):", nmi)

        # svm
        # model = SVC(C=1.0, kernel='rbf')
        # model.fit(closed_q_feature_list, closed_true_labels)
        # pred_svm = model.predict(closed_q_feature_list) # テストデータへの予測実行
        # # print('pred_svm', pred_svm)
        # # print('closed_true_labels', closed_true_labels)
        # svm_acc = metrics.accuracy_score(closed_true_labels, pred_svm)
        # print("SVM Accuracy(C=1.0, kernel='rbf'):", svm_acc)

        # model = SVC(C=1.0, kernel='linear')
        # model.fit(closed_q_feature_list, closed_true_labels)
        # pred_svm = model.predict(closed_q_feature_list) # テストデータへの予測実行
        # # print('pred_svm', pred_svm)
        # # print('closed_true_labels', closed_true_labels)
        # svm_acc = metrics.accuracy_score(closed_true_labels, pred_svm)
        # print("SVM Accuracy(C=1.0, kernel='linear'):", svm_acc)

        # model = SVC(C=1.0, kernel='poly')
        # model.fit(closed_q_feature_list, closed_true_labels)
        # pred_svm = model.predict(closed_q_feature_list) # テストデータへの予測実行
        # # print('pred_svm', pred_svm)
        # # print('closed_true_labels', closed_true_labels)
        # svm_acc = metrics.accuracy_score(closed_true_labels, pred_svm)
        # print("SVM Accuracy(C=1.0, kernel='poly'):", svm_acc)

        return ari

    else:
        raise NameError('Unknown mode ({})!'.format(mode))


def run_validation(opts,args, val_db, net, episode, opts_val):

    _curr_str = '\tEvaluating at episode {}, with {} iterations ... (be patient)'.format(episode, opts_val.iterations)
    opts.logger(_curr_str)

    if opts.train.mode.startswith('open'):
        accuracy, _, _, auroc, _, _, _ = evaluation(args, net, val_db, opts.train.mode, opts_val)

        eqn = '>' if accuracy > opts.ctrl.best_accuracy else '<'
        _curr_str = '\t\tCurrent accuracy is {:.4f} {:s} ' \
                    'previous best accuracy is {:.4f} (ep{})'.format(accuracy, eqn, opts.ctrl.best_accuracy, opts.ctrl.best_episode)
        opts.logger(_curr_str)
        _curr_str = '\t\tOpen-Set AUROC: {:.4f}'.format(auroc)
        opts.logger(_curr_str)
    elif opts.train.mode == 'regular':
        accuracy = evaluation(args,net, val_db, opts.train.mode, opts_val)

        eqn = '>' if accuracy > opts.ctrl.best_accuracy else '<'
        _curr_str = '\t\tCurrent accuracy is {:.4f} {:s} ' \
                    'previous best accuracy is {:.4f} (ep{})'.format(accuracy, eqn, opts.ctrl.best_accuracy, opts.val.best_episode)
        opts.logger(_curr_str)
    else:
        raise NameError('Unknown mode ({})!'.format(opts.train.mode))

    if accuracy > opts.ctrl.best_accuracy:
        opts.ctrl.best_accuracy = accuracy
        opts.ctrl.best_episode = episode
        return True, accuracy, auroc
    else:
        return False, accuracy, auroc


def run_test(opts, args, test_db, net, opts_test, image_files=[]):
    _curr_str = 'Evaluating with {} iterations ... (be patient)'.format(opts_test.iterations)
    opts.logger(_curr_str)

    if opts.train.mode.startswith('open'):
        closed_accuracy, open_accuracy, all_accuracy, auroc = evaluation(args, net, test_db, opts.train.mode, opts_test)

        # All Accuracyのclosed-set部分
        _curr_str = '\tClosed Accuracy: {:.4f}'.format(closed_accuracy)
        opts.logger(_curr_str)
        # All Accuracyのopen-set部分
        _curr_str = '\tOpen Accuracy: {:.4f}'.format(open_accuracy)
        opts.logger(_curr_str)
        # closed-set, open-setに対する分類精度
        _curr_str = '\tAll Accuracy: {:.4f}'.format(all_accuracy)
        opts.logger(_curr_str)
        _curr_str = '\tAUROC: {:.4f}'.format(auroc)
        opts.logger(_curr_str)

        # 評価結果をcsvファイルに出力
        # with open(opts.file.result_csv, "w", newline="") as f:
        #     fieldnames = ["file_name", "GT", "pred_label"]

        #     # 1~11カテゴリを追加
        #     for i in range(11):
        #         fieldnames.append("category"+str(i))
        #     writer = csv.writer(f)
        #     writer.writerow(fieldnames)

        #     # {query-set(75) + open-set(75)} * episode(600) = 9000
        #     print('image_files:', len(image_files))
        #     print('gt:', len(gt))
        #     print('infe:', len(infe))
        #     print('prob:', len(prob))

        #     for i, image_name in enumerate(image_files):
        #         writer.writerow([image_name, gt[i], infe[i]] + prob[i])
    elif opts.train.mode == 'regular':
        ari = evaluation(args, net, test_db, opts.train.mode, opts_test)

        _curr_str = '\tOpen-Set ARI: {:.4f}'.format(ari)
        opts.logger(_curr_str)
        # _curr_str = '\tOpen-Set SVM Accuracy: {:.4f}'.format(svm_acc)
        # opts.logger(_curr_str)

        # _curr_str = '\tAccuracy: {:.4f}'.format(accuracy)
        # opts.logger(_curr_str)
    