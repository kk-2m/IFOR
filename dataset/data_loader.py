from torch.utils.data import DataLoader
import csv
from dataset.miniimagenet import miniImagenet
from dataset.sampler import MetaSampler
from dataset.sampler_ifor import MetaSampler_IFOR
from dataset.wcs_color import colorWCS
from dataset.wcs_infrared import infraredWCS

def data_loader(opts, opts_runtime, split):

    if split == 'train':
        _curr_str = 'Train data ...'
        # 1
        augment = opts.data.augment
        is_train = True
        # default 1
        batch_size = opts.train.batch_size
        is_shuffle = True
    elif split == 'val':
        _curr_str = 'Val data ...'
        # 0
        augment = opts.data.augment_val
        is_train = False
        # default 1
        batch_size = opts.train.batch_size_val
        is_shuffle = False
    else:  # split == 'test':
        _curr_str = 'Test data ...'
        # 2
        augment = opts.data.augment_test
        is_train = False
        # default 1
        batch_size = opts.train.batch_size_test
        is_shuffle = False

    # create data_loader
    if opts.data.name == 'miniimagenet':

        relative_path = 'dataset/miniImageNet/'

        opts.logger(_curr_str)
        data = miniImagenet(
            root=relative_path,
            resize=opts.data.im_size, split=split, mode=opts.train.mode, augment=augment)
        opts.logger('\t\tFind {:d} closed samples'.format(data.closed_samples))
        if data.open_samples > 0:
            opts.logger('\t\tFind {:d} open samples'.format(data.open_samples))

    elif opts.data.name == "wcs_color":
        relative_path = './dataset/wcs/dataset/color_dataset/'

        opts.logger(_curr_str)
        data = colorWCS(
            root=relative_path,
            resize=opts.data.im_size, split=split, mode=opts.train.mode, augment=augment)
        opts.logger('\t\tFind {:d} closed samples'.format(data.closed_samples))
        if data.open_samples > 0:
            opts.logger('\t\tFind {:d} open samples'.format(data.open_samples))

    elif opts.data.name == "wcs_infrared":
        relative_path = './dataset/wcs/dataset/infrared_dataset/'

        opts.logger(_curr_str)
        # image_sizeをinfraredWCSに渡していないから, defaultの224が入力画像サイズとなっている
        data = infraredWCS(
            root=relative_path,
            resize=opts.data.im_size, split=split, mode=opts.train.mode, augment=augment)
        opts.logger('\t\tFind {:d} number of classes'.format(data.cls_num))
        opts.logger('\t\tFind {:d} closed samples'.format(data.closed_samples))
        if data.open_samples > 0:
            opts.logger('\t\tFind {:d} open samples'.format(data.open_samples))
        
        # for image, label in data:
        #     print('labael', label)
        # print('data', len(data))
        # data 1100
        # indexes = [410,  475,  402,  424,  421,  444,  407,
        #  416,  491,  468,  422,  445,  460,  476,  452,  929,  964,  998,  972,
        #  988,  905,  915,  912,  917,  961,  976,  909,  978,  980,  907,  296,
        #  258,  214,  272,  204,  247,  264,  257,  263,  206,  238,  213,  259,
        #  266,  271, 1084, 1033, 1053, 1002, 1049, 1048, 1082, 1041, 1066, 1029,
        # 1037, 1069, 1050, 1097, 1059,  656,  637,  646,  602,  690,  613,  615,
        #  658,  667,  616,  617,  664,  638,  640,  614]
        # for i in indexes:
        #     print('data', i, (data.data[i])[1])
        
    elif opts.dataset.name == 'cifar10':
        raise NameError('cifar10 not implemented.')

    else:
        raise NameError('Unknown dataset ({})!'.format(opts.dataset.name))

    # turn data_loader into db
    if opts.train.mode == 'openfew':
        if opts.data.name == "wcs_infrared" or opts.data.name == "wcs_color":
            data_sampler = MetaSampler_IFOR(data, opts_runtime, train=is_train)
            # print('data_sampler', len(data_sampler))
            # data_sampler 1800
            # 1800はイテレーションの数
            # 600 x 3が実行されるため1800の要素が格納されている
            # yieldによって、1エピソードは 3 batch で構成される
            # print('idx_list', len(data_sampler.idx_list))
            # idx_list 11
            # print('n_way', data_sampler.n_way)
            # print('k_shot', data_sampler.k_shot)
            # print('m_query', data_sampler.m_query)
            # print('open_sample', data_sampler.open_sample)
            # data_sampler2 = copy.deepcopy(data_sampler)
            # for batch in data_sampler:
            #     print('length', len(batch))
            #     print('batch1', batch)
            #     break
            # for batch in data_sampler:
            #     print('length', len(batch))
            #     print('batch2', batch)
            #     break

            # data_samplerをcsvにあらかじめ出力しておく
            # 使い終わったらコメントアウトしておく
            # closed-setのdataが欲しいため、support-set, open-setは除く
            # if split == 'test':
            #     with open(opts.file.test_episode_csv, "w", newline="") as f:
            #         writer = csv.writer(f)
            #         support_amount = opts.fsl.n_way_test * opts.fsl.k_shot_test
            #         open_amount = opts.open.n_cls_test * opts.open.m_sample_test
            #         for episode_datasets in data_sampler:
            #             print('episode_datasets', episode_datasets)
            #             print('data amount is ', len(episode_datasets))
            #             writer.writerow(episode_datasets.tolist()[support_amount : -open_amount])

        else:
        # FSL, OSR用の画像選択アルゴリズム
            data_sampler = MetaSampler(data, opts_runtime, train=is_train)
        # データセットから取得する
        db = DataLoader(data, batch_sampler=data_sampler, num_workers=8, pin_memory=True)
        # for images, labels in db:
        #     print('label1', labels)
        #     break
        # for images, labels in db1:
        #     print('label2', labels)
        #     break
    elif opts.train.mode == 'openmany':
        data_sampler = MetaSampler(data, opts_runtime, train=is_train)
        db = DataLoader(data, batch_sampler=data_sampler, num_workers=8, pin_memory=True)
    elif opts.train.mode == 'regular':
        db = DataLoader(data, batch_size, shuffle=is_shuffle, num_workers=8, pin_memory=True)

    return db