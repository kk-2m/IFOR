import argparse
import csv
import os
import glob
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from core.model import OpenNet
from dataset.data_loader import data_loader
from core.config import Config
from core.workflow import run_validation, run_test, create_runtime_opts, save_model, update_loss_scale, to_device

# ここから追加分(ViT)
from core.models import get_model
import utils.deit_util as utils
import random
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
import sys

#ここから追加（自分）
import time
from utils.csv_util import get_directory_structure, get_image_paths
from utils.nearest_neighbor import k_center, k_center_simple

from timm.data import Mixup
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

if __name__ == '__main__':
    # 経過時間測定用
    start = time.time()

    # ViTのコード用のコマンドライン引数など
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/openfew/default.yaml', type=str)
    parser.add_argument('--test', default=False, action='store_true')
    
    # 追加 ------------------------
    # General
    # parser.add_argument('--batch-size', default=1, type=int)
    # parser.add_argument('--num_classes', default=1000, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--fp16', action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.set_defaults(fp16=True)
    # command logの出力ディレクトリの指定
    parser.add_argument('--command_dir', default='command_log/tmp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='cuda:gpu_id for single GPU training')
    parser.add_argument('--seed', default=0, type=int)

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--pretrained-checkpoint-path', default='.', type=str,
                        help='path which contains the directories pretrained_ckpts and pretrained_ckpts_converted')
    parser.add_argument("--dataset", choices=["cifar_fs_elite", "cifar_fs", "mini_imagenet", "meta_dataset"],
                        default="cifar_fs",
                        help="Which few-shot dataset.")

    # Few-shot parameters (Mini-ImageNet & CIFAR-FS)
    parser.add_argument("--nClsEpisode", default=5, type=int,
                        help="Number of categories in each episode.")
    parser.add_argument("--nSupport", default=1, type=int,
                        help="Number of samples per category in the support set.")
    parser.add_argument("--nQuery", default=15, type=int,
                        help="Number of samples per category in the query set.")
    parser.add_argument("--nValEpisode", default=120, type=int,
                        help="Number of episodes for validation.")
    parser.add_argument("--nEpisode", default=2000, type=int,
                        help="Number of episodes for training / testing.")

    # MetaDataset parameters
    parser.add_argument('--image_size', type=int, default=128,
                        help='Images will be resized to this value')
    parser.add_argument('--base_sources', nargs="+", default=['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower'],
                        help='List of datasets to use for training')
    parser.add_argument('--val_sources', nargs="+", default=['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower'],
                        help='List of datasets to use for validation')
    parser.add_argument('--test_sources', nargs="+", default=['traffic_sign', 'mscoco', 'ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower'],
                        help='List of datasets to use for meta-testing')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether or not to shuffle data for TFRecordDataset')
    parser.add_argument('--train_transforms', nargs="+", default=['random_resized_crop', 'jitter', 'random_flip', 'to_tensor', 'normalize'],
                        help='Transforms applied to training data',)
    parser.add_argument('--test_transforms', nargs="+", default=['resize', 'center_crop', 'to_tensor', 'normalize'],
                        help='Transforms applied to test data',)
    parser.add_argument('--num_ways', type=int, default=None,
                        help='Set it if you want a fixed # of ways per task')
    parser.add_argument('--num_support', type=int, default=None,
                        help='Set it if you want a fixed # of support samples per class')
    parser.add_argument('--num_query', type=int, default=None,
                        help='Set it if you want a fixed # of query samples per class')
    parser.add_argument('--min_ways', type=int, default=5,
                        help='Minimum # of ways per task')
    parser.add_argument('--max_ways_upper_bound', type=int, default=50,
                        help='Maximum # of ways per task')
    parser.add_argument('--max_num_query', type=int, default=10,
                        help='Maximum # of query samples')
    parser.add_argument('--max_support_set_size', type=int, default=500,
                        help='Maximum # of support samples')
    parser.add_argument('--max_support_size_contrib_per_class', type=int, default=100,
                        help='Maximum # of support samples per class')
    parser.add_argument('--min_examples_in_class', type=int, default=0,
                        help='Classes that have less samples will be skipped')
    parser.add_argument('--min_log_weight', type=float, default=np.log(0.5),
                        help='Do not touch, used to randomly sample support set')
    parser.add_argument('--max_log_weight', type=float, default=np.log(2),
                        help='Do not touch, used to randomly sample support set')
    parser.add_argument('--ignore_bilevel_ontology', action='store_true',
                        help='Whether or not to use superclass for BiLevel datasets (e.g Omniglot)')
    parser.add_argument('--ignore_dag_ontology', action='store_true',
                        help='Whether to ignore ImageNet DAG ontology when sampling \
                              classes from it. This has no effect if ImageNet is not  \
                              part of the benchmark.')
    parser.add_argument('--ignore_hierarchy_probability', type=float, default=0.,
                        help='if using a hierarchy, this flag makes the sampler \
                              ignore the hierarchy for this proportion of episodes \
                              and instead sample categories uniformly.')

    # CDFSL parameters
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support')
    parser.add_argument('--cdfsl_domains', nargs="+", default=['EuroSAT', 'ISIC', 'CropDisease', 'ChestX'], help='CDFSL datasets')

    # Model params
    parser.add_argument('--arch', default='dino_base_patch16', type=str,
                        help='Architecture of the backbone.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--unused_params', action='store_true')
    parser.add_argument('--no-pretrain', action='store_true')

    # Deployment params
    parser.add_argument("--deploy", type=str, default="vanilla",
                        help="Which few-shot model to be deployed for meta-testing.")
    parser.add_argument('--num_adapters', default=1, type=int, help='Number of adapter tokens')
    parser.add_argument('--ada_steps', default=40, type=int, help='Number of feature adaptation steps')
    parser.add_argument('--ada_lr', default=5e-2, type=float, help='Learning rate of feature adaptation')
    parser.add_argument('--aug_prob', default=0.9, type=float, help='Probability of applying data augmentation during meta-testing')
    parser.add_argument('--aug_types', nargs="+", default=['color', 'translation'],
                        help='color, offset, offset_h, offset_v, translation, cutout')

    # Other model parameters
    parser.add_argument('--img-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    # 変更する
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR (step scheduler)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Misc
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # ここから分散学習に関する引数を追加(modify)

    # 分散学習に関する引数を追加
    parser.add_argument('--distributed', action='store_true', default=False)

    # parser.add_argument('--rank', default=0, type=int, help='rank of the current process')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # ------------------------------
    # さらに追加---------------------------------

    parser.add_argument('--distance', default='euclidean', help='euclidean distance is used default. If you use Dino as pretrained model, use cosine')
    # parser.add_argument('--pretrain', default='euclidean', help='euclidean distance is used default. If you use Dino as pretrained model, use cosine')
    # ---------------------------------

    main_opts = parser.parse_args()
    # meta-open用
    opts = Config(main_opts.cfg)
    opts.setup(test=main_opts.test)
    opts.ctrl.cfg = main_opts.cfg
    # output_logの保存ディレクトリや、configurationを表示
    opts.print_args()
    # vit用
    args = main_opts = parser.parse_args()
    # 追加 ---------------------------
    # 並列モードを使うか否か
    # いらないかも
    # utils.init_distributed_mode(args) 

    # parserをすべて表示
    # 見やすいように変更する余地あり
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # なんこれ？
    cudnn.benchmark = True

# command logの記録
    output_dir = Path(args.command_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "log.txt").open("a") as f:
            f.write(" ".join(sys.argv) + "\n")

    # 分散タスク（並列化）をするか否か。
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    # Mixup regularization (by default OFF)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nClsEpisode)
    # ------------------------------------

    with torch.cuda.device(int(opts.ctrl.gpu_id)):
        # テスト
        if main_opts.test:
            opts.logger('[Testing starts] ...\n')

            # print('previous aug_scale in main', opts.model.aug_scale_test)
            opts_test = create_runtime_opts(opts, 'test')
            # print('before aug_scale in main', opts_test.aug_scale)
            opts.logger('Preparing dataset: {:s} ...'.format(opts.data.name))
            test_db = data_loader(opts, opts_test, 'test')
            # for images, labels in test_db:
            #     print('image', images.size())
            #     print('label', labels)
            #     break
            # print('test_db:', len(test_db))
            # # test_db: 1800
            # for batch in test_db:
            #     print('batch', len(batch))
            #     print('image', batch[0].size())
            #     # print(batch[0])
            #     print('label', batch[1].size())
            #     # print(batch[1])
            # openNetでモデルを初期化する
            net = OpenNet(opts,args).to(opts.ctrl.device)
            # 追加 -------------------------
            # net = get_model(opts)
            # net.to(device)

            model_ema = None # (by default OFF)
            if args.model_ema:
                # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                model_ema = ModelEma(
                    net,
                    decay=args.model_ema_decay,
                    device='cpu' if args.model_ema_force_cpu else '',
                    resume='')

            model_without_ddp = net
            # true
            # 分散トレーニングの設定
            if args.distributed:
                net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],
                                                                find_unused_parameters=args.unused_params)
                model_without_ddp = net.module
            n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print('number of params:', n_parameters)

            # Optimizer & scheduler & criterion
            # if args.fp16:
            #     scale = 1 / 8 # the default lr is for 8 GPUs
            #     linear_scaled_lr = args.lr * utils.get_world_size() * scale
            #     args.lr = linear_scaled_lr

            # loss_scaler = NativeScaler() if args.fp16 else None

            # #optimizer = create_optimizer(args, model_without_ddp)
            # optimizer = torch.optim.SGD(
            #     [p for p in model_without_ddp.parameters() if p.requires_grad],
            #     args.lr,
            #     momentum=args.momentum,
            #     weight_decay=0, # no weight decay for fine-tuning
            # )

            # lr_scheduler, _ = create_scheduler(args, optimizer)

            if args.mixup > 0.:
                # smoothing is handled with mixup label transform
                criterion = SoftTargetCrossEntropy()
            elif args.smoothing:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()

            # ------------------------------

            # 重みをダウンロード
            # Cofig.yamlで設定されているio.root/train.mode/io.exp_nameのディレクトリから
            # bestな重みをロードする
            state_dict_path = opts.io.best_model_file
            checkpoints = torch.load(state_dict_path, map_location='cuda:0')
            opts.logger('loading check points from {}'.format(state_dict_path))
            net.load_state_dict(checkpoints['model'], strict=True)

            # ## 評価結果をcsvファイルへ出力するための設定
            # # 基本パス
            # base_path = opts.file.test_data_path
            # print('test_dataset_path is', base_path)
            # # ディレクトリ構造を取得
            # folder_names, file_names = get_directory_structure(base_path)
            # image_paths = []

            # # episode内のclosed-setの構成が記されているcsvを読み込む
            # with open(opts.file.test_episode_csv, "r") as f:
            #     reader = csv.reader(f)
            #     data_sampler = list(reader)
            #     # 画像ファイルのパスを取得
            #     for episode_datasets in data_sampler:
            #         # print('episode_dataset1', episode_datasets)
            #         # break
            #         image_paths.extend(get_image_paths(episode_datasets, folder_names, file_names, base_path))
            
            # run_test(opts, args, test_db, net, opts_test, image_paths)
            run_test(opts, args, test_db, net, opts_test)

            # k-meansとsvmの実行
            opts.train.mode = 'regular'
            opts.train.batch_size_test = 25
            test_db2 = data_loader(opts, opts_test, 'test')
            # for x, y in test_db2:
            #     print('image size of', x.size())
            #     print('label name', y)
                # break
            run_test(opts, args, test_db2, net, opts_test)

            print('Testing done!')        
        # 学習
        else:
            # 学習率を探索するモード
            lr_search = False
            if lr_search:
                # category_idからcategory_nameを取得し、カテゴリーフォルダパスを作成
                lr_list_folder_path = "./lr_list/"

                # lr_listフォルダが存在しなかったら新しく作成する。
                if not os.path.isdir(lr_list_folder_path):
                    print("mkdir:",lr_list_folder_path)
                    os.makedirs(lr_list_folder_path)

                # category_idからcategory_nameを取得し、カテゴリーフォルダパスを作成
                model_lr_folder_path = "./lr_list/" + args.arch + "/"

                # lr_listフォルダが存在しなかったら新しく作成する。
                if not os.path.isdir(model_lr_folder_path):
                    print("mkdir:",model_lr_folder_path)
                    os.makedirs(model_lr_folder_path)

                # 学習率を変えて実験する。6回繰り返す
                for _lr in range(5):
                    _lr = 1e-6 * (10 **_lr)
                    print('leaning rate: ', _lr)
                        # ex) leaning rate: 1e-06
                    _result = []
                    _result.append(str(_lr))
                    #ネットワークを初期化
                    net = OpenNet(opts, args).to(opts.ctrl.device)
                    # Building up models ...
                    # ViT origin is used
                    # args.arch vit-origin

                    # 追加 -------------------------
                    # net = get_model(opts)
                    # net.to(device)

                    model_ema = None # (by default OFF)
                    if args.model_ema:
                        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                        model_ema = ModelEma(
                            net,
                            decay=args.model_ema_decay,
                            device='cpu' if args.model_ema_force_cpu else '',
                            resume='')

                    model_without_ddp = net
                    # true
                    # 分散トレーニングの設定
                    if args.distributed:
                        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],
                                                                        find_unused_parameters=args.unused_params)
                        model_without_ddp = net.module
                    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    print('number of params:', n_parameters)

                    # Optimizer & scheduler & criterion
                    # if args.fp16:
                    #     scale = 1 / 8 # the default lr is for 8 GPUs
                    #     linear_scaled_lr = args.lr * utils.get_world_size() * scale
                    #     args.lr = linear_scaled_lr

                    # loss_scaler = NativeScaler() if args.fp16 else None

                    #optimizer = create_optimizer(args, model_without_ddp)
                    # optimizer = torch.optim.SGD(
                    #     [p for p in model_without_ddp.parameters() if p.requires_grad],
                    #     args.lr,
                    #     momentum=args.momentum,
                    #     weight_decay=0, # no weight decay for fine-tuning
                    # )

                    # lr_scheduler, _ = create_scheduler(args, optimizer)

                    if args.mixup > 0.:
                        # smoothing is handled with mixup label transform
                        criterion = SoftTargetCrossEntropy()
                    elif args.smoothing:
                        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
                    else:
                        criterion = torch.nn.CrossEntropyLoss()

                    # ------------------------------

                    # optimizer and lr_scheduler
                    # optimizerをいじってみる ---------------------------
                    # optimizer = optim.Adam(net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay)
                    # CNNの場合の最適化関数
                    # if opts.model.structure=="resnet":
                    #     optimizer = optim.Adam(net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay)
                    # # ViTの場合の最適化関数
                    # elif opts.model.structure == "vit":
                    #     optimizer = torch.optim.SGD(
                    #         [p for p in model_without_ddp.parameters() if p.requires_grad],
                    #         args.lr,
                    #         momentum=args.momentum,
                    #         weight_decay=0, # no weight decay for fine-tuning
                    #     )
                    optimizer = optim.Adam(net.parameters(), lr=_lr, weight_decay=opts.train.weight_decay)

                    scheduler = MultiStepLR(optimizer, milestones=opts.train.lr_scheduler, gamma=opts.train.lr_gamma)

                    opts.logger('[Training starts] ...\n')
                    # エピソード数は、default 3
                    total_ep = opts.train.nep
                    # 3*10000 = 30000エピソード
                    total_episode = total_ep * opts.fsl.iterations
                    # 基本設定（n_shotなどを格納）
                    opts_train = create_runtime_opts(opts, 'train')
                    opts_val = create_runtime_opts(opts, 'val')

                    loss_episode = 0
                    total_loss = 0.0
                    # 3回繰り返す

                    # print(model_lr_folder_path + opts.data.name + "_" + args.arch + "_lr" + str(_lr) + ".txt")

                    for epoch in range(total_ep):
                        # DATA
                        opts.logger('Preparing dataset: {:s} ...'.format(opts.data.name))
                        # 基本のFSOSRの設定
                        train_db = data_loader(opts, opts_train, 'train')
                        val_db = data_loader(opts, opts_val, 'val')

                        # adjust learning rate
                        old_lr = optimizer.param_groups[0]['lr']
                        if epoch == 0:
                            opts.logger('Start lr is {:.8f}, at epoch {}\n'.format(old_lr, epoch))
                        scheduler.step(epoch)
                        new_lr = optimizer.param_groups[0]['lr']
                        if new_lr != old_lr:
                            opts.logger('LR changes from {:.8f} to {:.8f} at episode {:d}\n'.format(old_lr, new_lr, epoch*opts.fsl.iterations))
                        # バッチの数(1*)だけ繰り返す。
                        for step, batch in enumerate(train_db):
                            # fsl.iterations = 100:

                            # (0,1,2)*(100)+(1~100)
                            episode = epoch*opts.fsl.iterations + step

                            # adjust loss scale
                            update_loss_scale(opts, episode)

                            # ネットにバッチを入力してロスを取得。
                            loss = net(batch, opts_train, True)
                            total_loss += loss.item()
                            # ロスでネットを更新
                            optimizer.zero_grad()
                            loss.backward()
                            if opts.train.clip_grad:
                                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                            optimizer.step()

                            # SHOW TRAIN LOSS
                            loss_episode += 1
                            if episode % opts.ctrl.ep_vis_loss == 0 or episode == total_episode - 1:
                                opts.logger(' [ep {:04d} ({})] loss: {:.4f}'.format(episode, total_episode, total_loss/loss_episode))
                                loss_episode = 0
                                total_loss = 0.0

                            # SAVE MODEL
                            if episode % opts.ctrl.ep_save == 0 or episode == total_episode - 1:
                                save_file = opts.io.model_file.format(episode)
                                save_model(opts, net, optimizer, scheduler, episode, save_file)
                                opts.logger('\tModel saved to: {}, at [episode {}]\n'.format(save_file, episode))

                            # VALIDATION and SAVE BEST MODEL
                            if episode % opts.ctrl.ep_val == 0 or episode == total_episode - 1:
                                output_list = run_validation(opts, args, val_db, net, episode, opts_val)
                                _result.append(f"{episode},{output_list[1]},{output_list[2]}")
                                if output_list[0]:
                                    save_file = opts.io.best_model_file
                                    save_model(opts, net, optimizer, scheduler, episode, save_file)
                                    opts.logger('\tBest model saved to: {}, at [episode {}]\n'.format(save_file, episode))
                                    #結果をlistに保存
                    opts.logger('')
                    opts.logger('Training done!')
                    with open(model_lr_folder_path + opts.data.name + "_" + args.arch + "_lr" + str(_lr) + ".txt", mode='w') as f:
                        f.write('\n'.join(_result))
            # 学習率を探索するモードじゃないとき
            else:
                #ネットワークを初期化
                net = OpenNet(opts, args).to(opts.ctrl.device)

                # 追加 -------------------------
                # net = get_model(opts)
                # net.to(device)

                model_ema = None # (by default OFF)
                if args.model_ema:
                    # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                    model_ema = ModelEma(
                        net,
                        decay=args.model_ema_decay,
                        device='cpu' if args.model_ema_force_cpu else '',
                        resume='')

                model_without_ddp = net
                # true
                # 分散トレーニングの設定
                if args.distributed:
                    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],
                                                                    find_unused_parameters=args.unused_params)
                    model_without_ddp = net.module
                n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print('number of params:', n_parameters)

                # Optimizer & scheduler & criterion
                # if args.fp16:
                #     scale = 1 / 8 # the default lr is for 8 GPUs
                #     linear_scaled_lr = args.lr * utils.get_world_size() * scale
                #     args.lr = linear_scaled_lr

                # loss_scaler = NativeScaler() if args.fp16 else None

                #optimizer = create_optimizer(args, model_without_ddp)
                # optimizer = torch.optim.SGD(
                #     [p for p in model_without_ddp.parameters() if p.requires_grad],
                #     args.lr,
                #     momentum=args.momentum,
                #     weight_decay=0, # no weight decay for fine-tuning
                # )

                # lr_scheduler, _ = create_scheduler(args, optimizer)

                if args.mixup > 0.:
                    # smoothing is handled with mixup label transform
                    criterion = SoftTargetCrossEntropy()
                elif args.smoothing:
                    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
                else:
                    criterion = torch.nn.CrossEntropyLoss()

                # ------------------------------

                # optimizer and lr_scheduler
                # optimizerをいじってみる ---------------------------
                # optimizer = optim.Adam(net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay)
                # CNNの場合の最適化関数
                # if opts.model.structure=="resnet":
                #     optimizer = optim.Adam(net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay)
                # # ViTの場合の最適化関数
                # elif opts.model.structure == "vit":
                #     optimizer = torch.optim.SGD(
                #         [p for p in model_without_ddp.parameters() if p.requires_grad],
                #         args.lr,
                #         momentum=args.momentum,
                #         weight_decay=0, # no weight decay for fine-tuning
                #     )
                optimizer = optim.Adam(net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay)

                scheduler = MultiStepLR(optimizer, milestones=opts.train.lr_scheduler, gamma=opts.train.lr_gamma)

                opts.logger('[Training starts] ...\n')
                # エピソード数は、default 3
                total_ep = opts.train.nep
                # 3*10000 = 30000エピソード
                total_episode = total_ep * opts.fsl.iterations
                # 基本設定（n_shotなどを格納）
                opts_train = create_runtime_opts(opts, 'train')
                opts_val = create_runtime_opts(opts, 'val')

                loss_episode = 0
                total_loss = 0.0
                # 3回繰り返す
                # # k-meansとsvmの実行
                # opts.train.mode = 'regular'
                # opts.train.batch_size_test = 25
                # train_kmeans_db = data_loader(opts, opts_train, 'train')
                # opts.train.mode = 'openfew'
                # opts.train.batch_size_test = 1

                features_toepisode5 = torch.zeros(opts.fsl.p_base, 192, opts.train.kmeans_ep-1, device=opts.ctrl.device)
                labels_toepisode5 = torch.zeros(opts.fsl.p_base, opts.train.kmeans_ep-1, device=opts.ctrl.device)

                for epoch in range(total_ep):
                    # DATA
                    opts.logger('Preparing dataset: {:s} ...'.format(opts.data.name))
                    # 基本のFSOSRの設定
                    train_db = data_loader(opts, opts_train, 'train')
                    val_db = data_loader(opts, opts_val, 'val')

                    # adjust learning rate
                    old_lr = optimizer.param_groups[0]['lr']
                    if epoch == 0:
                        opts.logger('Start lr is {:.8f}, at epoch {}\n'.format(old_lr, epoch))
                    scheduler.step(epoch)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        opts.logger('LR changes from {:.8f} to {:.8f} at episode {:d}\n'.format(old_lr, new_lr, epoch*opts.fsl.iterations))
                    
                    # バッチの数(1*)だけ繰り返す。
                    for step, batch in enumerate(train_db):
                        # initialization for k-means
                        toepisode5_n = 0
                        n_samples = 0
                        features_k = torch.zeros(opts.fsl.p_base, 192, device=opts.ctrl.device) # shape=(image_number,192)
                        result_k = torch.zeros(opts.fsl.p_base, dtype=torch.int32, device=opts.ctrl.device) # group number of the image
                        # (0,1,2)*(100)+(1~100)
                        episode = epoch*opts.fsl.iterations + step + 1
                        opts.logger("episode: {}".format(episode))

                        # adjust loss scale
                        update_loss_scale(opts, episode)
                        # ネットにバッチを入力してロスを取得。
                        loss, feature_k, log_loss = net(batch, opts_train, train=True)
                        [opts.logger("{}: {}".format(key, value)) for key, value in log_loss.items()]
                        # print('feature_k size', feature_k.shape)
                        timing = 0

                        if opts.train.kmeans and episode > timing:
                            # clustering for first time
                            if episode == opts.train.kmeans_ep+timing:
                                # print('features_toepisode5', features_toepisode5.size())
                                features_toepisode5 = features_toepisode5.view(features_toepisode5.shape[0]*(opts.train.kmeans_ep-1),192) # shape=(image_number*kmeans_ep,192)
                                # クラスタ数kを求める
                                unique_labels_toepisode5 = torch.unique(labels_toepisode5.view(labels_toepisode5.shape[0]*(opts.train.kmeans_ep-1)))
                                k_toepisode5 = len(unique_labels_toepisode5)
                                # print('k_toepisode =',k_toepisode5)
                                # クラスタ中心を求める
                                _, best_c = k_center(features_toepisode5, groups=opts.model.num_classes, device=opts.ctrl.device) # shape=(groups,192)
                            # 入力から特徴量を抽出、エピソードを構成しているラベルを格納
                            if episode < opts.train.kmeans_ep+timing:
                                features_toepisode5[toepisode5_n:(toepisode5_n+feature_k.shape[0]),:,episode-timing-1] = feature_k.data
                                labels_toepisode5[toepisode5_n:(toepisode5_n+feature_k.shape[0]),episode-timing-1] = batch[1][-opts.fsl.p_base:]
                                # print('target base', batch[1][-opts.fsl.p_base:])
                                toepisode5_n += feature_k.shape[0]
                            elif episode > opts.train.kmeans_ep+timing and episode % opts.train.kmeans_ep == 0:
                                # 入力パッチの特徴量と現在のクラスタ中心との距離を求める
                                distance_k = torch.sum(torch.pow((feature_k.expand(best_c.shape[0],feature_k.shape[0],feature_k.shape[1]).permute(1,0,2)-best_c.unsqueeze(0)),2), dim=2) # shape=(N,groups)
                                # 各パッチのクラスタ割り当て結果を格納
                                y_id = torch.argsort(distance_k, dim=1)[:,0].type(torch.int32)
                                # 抽出された特徴量とクラスタ割り当て結果を格納
                                features_k = feature_k.data
                                result_k = y_id
                                
                                # 各パッチと割り当てられたクラスタ中心との距離の最小値の平均をとる
                                loss_kmeans = distance_k.min(dim=1).values.mean()
                                opts.logger("loss_kmeans: {}".format(loss_kmeans))
                                # print("loss_kmeans",loss_kmeans)

                                # loss += loss_kmeans * opts.train.loss_scale_bc

                                unique_labels = torch.unique(batch[1][-opts.fsl.p_base:])
                                k = len(unique_labels)
                                # print('k =', k)
                                # print('result_k', result_k)
                                # print('best_c', best_c.size())
                                # print('features_k', features_k.size())
                                new_result, best_c, _ = k_center_simple(features_k, result_k, best_c, groups=opts.model.num_classes, device=opts.ctrl.device)
                                # print('new result:',new_result)

                                distance_matrix = torch.cdist(best_c, best_c)
                                # loss_bc_pos = 1/torch.mean(distance_matrix)
                                # opts.logger("loss_between-class positive: {}".format(loss_bc_pos))
                                loss_bc_neg = -torch.mean(distance_matrix)
                                opts.logger("loss_between-class negative: {}".format(loss_bc_neg))

                                # loss += loss_bc_pos * opts.train.loss_scale_bc
                                loss += loss_bc_neg * opts.train.loss_scale_bc
                                # loss += loss_kmeans*loss_bc_pos * opts.train.loss_scale_bc
                                # loss += loss_kmeans + loss_bc_pos * opts.train.loss_scale_bc
                                # loss += loss_kmeans + loss_bc_neg * opts.train.loss_scale_bc
                                # bad loss
                                # loss += loss_bc_neg * opts.train.loss_scale_bc
                                # loss += loss_kmeans/loss_bc_neg * opts.train.loss_scale_bc
                                # loss += loss_kmeans + (1/loss_bc_neg * opts.train.loss_scale_bc)

                        total_loss += loss.item()
                        # ロスでネットを更新
                        optimizer.zero_grad()
                        loss.backward()
                        if opts.train.clip_grad:
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                        optimizer.step()

                        # SHOW TRAIN LOSS
                        loss_episode += 1
                        if episode % opts.ctrl.ep_vis_loss == 0 or episode == total_episode - 1:
                            opts.logger(' [ep {:04d} ({})] loss: {:.4f}'.format(episode, total_episode, total_loss/loss_episode))
                            loss_episode = 0
                            total_loss = 0.0

                        # SAVE MODEL
                        if episode % opts.ctrl.ep_save == 0 or episode == total_episode - 1:
                            save_file = opts.io.model_file.format(episode)
                            save_model(opts, net, optimizer, scheduler, episode, save_file)
                            opts.logger('\tModel saved to: {}, at [episode {}]\n'.format(save_file, episode))

                        # VALIDATION and SAVE BEST MODEL
                        if episode % opts.ctrl.ep_val == 0 or episode == total_episode - 1:
                            if run_validation(opts, args, val_db, net, episode, opts_val):
                                save_file = opts.io.best_model_file
                                save_model(opts, net, optimizer, scheduler, episode, save_file)
                                opts.logger('\tBest model saved to: {}, at [episode {}]\n'.format(save_file, episode))

                opts.logger('')
                opts.logger('Training done!')
    end = time.time()
    time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
    opts.logger('')
    opts.logger("Execution time: {}".format(time_diff))  # 処理にかかった時間データを使用