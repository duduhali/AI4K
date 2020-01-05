import os
import random
from glob import glob
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn.parallel import DataParallel
import torch.backends.cudnn as cudnn
from dataloder import EvalDataset
import models.archs.EDVR_arch as EDVR_arch
from tqdm import tqdm
from collections import OrderedDict
import cv2

def main(args):
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    print("===> Loading datasets")
    data_set = EvalDataset(args.test_lr,n_frames=args.n_frames, interval_list=args.interval_list,)
    eval_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.workers)

    #### random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    #cudnn.deterministic = True

    print("===> Building model")
    #### create model
    model = EDVR_arch.EDVR(nf=args.nf, nframes=args.n_frames, groups=args.groups, front_RBs=args.front_RBs,
                           back_RBs=args.back_RBs,
                           center=args.center, predeblur=args.predeblur, HR_in=args.HR_in, w_TSA=args.w_TSA)
    print("===> Setting GPU")
    gups = args.gpus if args.gpus!=0 else torch.cuda.device_count()
    device_ids = list(range(gups))
    model = DataParallel(model,device_ids=device_ids)
    model = model.cuda()

    # print(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isdir(args.resume):
            # 获取目录中最后一个
            pth_list = sorted(glob(os.path.join(args.resume, '*.pth')))
            if len(pth_list) > 0:
                args.resume = pth_list[-1]
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                namekey = 'module.' + k  # remove `module.`
                new_state_dict[namekey] = v
            model.load_state_dict(new_state_dict)

    #### training
    print("===> Eval")
    model.eval()
    with tqdm(total=(len(data_set) - len(data_set) % args.batch_size)) as t:
        for data in eval_loader:
            data_x = data['LRs'].cuda()
            names = data['files']

            with torch.no_grad():
                outputs = model(data_x).data.float().cpu()
            outputs = outputs*255.
            outputs = outputs.clamp_(0, 255).numpy()
            for img,file in zip(outputs,names):
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
                img = img.round()

                arr = file.split('/')
                dst_dir = os.path.join(args.outputs_dir,arr[-2])
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                dst_name = os.path.join(dst_dir,arr[-1])

                cv2.imwrite(dst_name, img)
            t.update(len(names))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataloader
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument('--test_lr', type=str, metavar='PATH', default='./../test_lr')
    parser.add_argument('--outputs-dir', default='./../output_img', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--n_frames', default=5, type=int)
    parser.add_argument('--interval_list', default=[1], type=int, nargs='+')
    parser.add_argument('--center', default=0, type=int)

    # model
    parser.add_argument('--nf', default=64, type=int)
    parser.add_argument('--groups', default=8, type=int)
    parser.add_argument('--front_RBs', default=5, type=int)
    parser.add_argument('--back_RBs', default=10, type=int)
    parser.add_argument('--predeblur', default=False, type=bool)  # 是否使用滤波
    parser.add_argument('--HR_in', default=False, type=bool)  # 很重要！！输入与输出是同样分辨率，就要求设置为true
    parser.add_argument('--w_TSA', default=True, type=bool)  # 是否使用TSA模块
    #model
    parser.add_argument('--seed', default=123, type=int)
    # check point
    parser.add_argument("--resume", default='checkpoint', type=str)
    parser.add_argument("--checkpoint", default='checkpoint', type=str)
    args = parser.parse_args()


    main(args)

    # pgrep python3 | xargs kill -s 9
    # python3 eval.py --batch_size 2

    # nvidia-smi