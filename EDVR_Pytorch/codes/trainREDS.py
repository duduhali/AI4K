import os
import math
import random
from glob import glob
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.distributed as dist

from utils import util
from models import create_model
from dataloderREDS import DatasetLoader
from models.Video_base_model import VideoBaseModel as M

def main(opt):
    parser = argparse.ArgumentParser()
    # dataloader
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='E:/2file/lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='E:/2file/hr_small')
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--scale', default=1, type=int)
    parser.add_argument('--n_frames', default=5, type=int)
    parser.add_argument('--interval_list', default=[1], type=int, nargs='+')


    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()

    device_id = torch.cuda.current_device()

    #### random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create dataloader
    file_name = sorted(os.listdir(args.data_lr))
    lr_list = []
    hr_list = []
    for one in file_name:
        lr_tmp = sorted(glob(os.path.join(args.data_lr, one, '*.png')))
        lr_list.extend(lr_tmp)
        hr_tmp = sorted(glob(os.path.join(args.data_hr, one, '*.png')))
        if len(hr_tmp) != 100:
            print(one)
        hr_list.extend(hr_tmp)

    data_set = DatasetLoader(lr_list, hr_list, patch_size=args.patch_size,
                             scale=args.scale, n_frames=args.n_frames, interval_list=args.interval_list)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                              pin_memory=False, drop_last=True)

    #### create model
    model = M(opt)

    #### resume training
    current_step = 0
    start_epoch = 0


    #### training
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['model'] in ['sr', 'srgan'] and rank <= 0:  # image restoration validation
                    # does not support multi-GPU validation
                    pbar = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.
                    idx = 0
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                        pbar.update('Test {}'.format(img_name))

                    avg_psnr = avg_psnr / idx

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)
                else:  # video restoration validation
                    if opt['dist']:
                        # multi-GPU testing
                        psnr_rlt = {}  # with border and center frames
                        if rank == 0:
                            pbar = util.ProgressBar(len(val_set))
                        for idx in range(rank, len(val_set), world_size):
                            val_data = val_set[idx]
                            val_data['LQs'].unsqueeze_(0)
                            val_data['GT'].unsqueeze_(0)
                            folder = val_data['folder']
                            idx_d, max_idx = val_data['idx'].split('/')
                            idx_d, max_idx = int(idx_d), int(max_idx)
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                               device='cuda')
                            # tmp = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                            model.feed_data(val_data)
                            model.test()
                            visuals = model.get_current_visuals()
                            rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                            gt_img = util.tensor2img(visuals['GT'])  # uint8
                            # calculate PSNR
                            psnr_rlt[folder][idx_d] = util.calculate_psnr(rlt_img, gt_img)

                            if rank == 0:
                                for _ in range(world_size):
                                    pbar.update('Test {} - {}/{}'.format(folder, idx_d, max_idx))
                        # # collect data
                        for _, v in psnr_rlt.items():
                            dist.reduce(v, 0)
                        dist.barrier()

                        if rank == 0:
                            psnr_rlt_avg = {}
                            psnr_total_avg = 0.
                            for k, v in psnr_rlt.items():
                                psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                                psnr_total_avg += psnr_rlt_avg[k]
                            psnr_total_avg /= len(psnr_rlt)
                            log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                            for k, v in psnr_rlt_avg.items():
                                log_s += ' {}: {:.4e}'.format(k, v)
                            logger.info(log_s)
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                                for k, v in psnr_rlt_avg.items():
                                    tb_logger.add_scalar(k, v, current_step)
                    else:
                        pbar = util.ProgressBar(len(val_loader))
                        psnr_rlt = {}  # with border and center frames
                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        for val_data in val_loader:
                            folder = val_data['folder'][0]
                            idx_d = val_data['idx'].item()
                            # border = val_data['border'].item()
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[folder] = []

                            model.feed_data(val_data)
                            model.test()
                            visuals = model.get_current_visuals()
                            rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                            gt_img = util.tensor2img(visuals['GT'])  # uint8

                            # calculate PSNR
                            psnr = util.calculate_psnr(rlt_img, gt_img)
                            psnr_rlt[folder].append(psnr)
                            pbar.update('Test {} - {}'.format(folder, idx_d))
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = sum(v) / len(v)
                            psnr_total_avg += psnr_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                        for k, v in psnr_rlt_avg.items():
                            log_s += ' {}: {:.4e}'.format(k, v)
                        logger.info(log_s)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

def getOpt():
    opt = dict()
    opt['name'] = '001_EDVR_OURS'# ** 实验名
    opt['use_tb_logger'] = True
    opt['model'] = 'VideoSR_base'
    opt['distortion'] = 'sr'
    opt['scale'] = 4
    opt['gpu_ids'] = [0]


    #logger
    logger = dict()
    logger['print_freq'] = 10 # 每多少个iterations打印日志
    logger['save_checkpoint_freq'] = 1e2 # 每多少个iterations保存模型
    opt['logger'] = logger

    #train
    train = dict()
    train['lr_G'] = 4e-4
    train['lr_scheme'] = 'CosineAnnealingLR_Restart'
    train['beta1'] = 0.9
    train['beta2'] = 0.99
    train['niter'] = 600000
    train['warmup_iter'] = -1  # -1: no warm up
    train['T_period'] = [150000, 150000, 150000, 150000]
    train['restarts'] = [150000, 300000, 450000]
    train['restart_weights'] = [1, 1, 1]
    train['eta_min'] = 1e-7
    train['pixel_criterion'] = 'cb'
    train['pixel_weight'] = 1.0
    train['val_freq'] = 2e3
    train['manual_seed'] = 0
    opt['train'] = train

    #path
    path = dict()
    path['pretrain_model_G'] = None
    path['strict_load'] = True
    path['resume_state'] = None
    opt['path'] = path

    #network_G  网络结构
    network_G = dict()
    network_G['which_model_G'] = 'EDVR'
    network_G['nf'] = 64
    network_G['nframes'] = 5
    network_G['groups'] = 8
    network_G['front_RBs'] = 5
    network_G['back_RBs'] = 10
    network_G['predeblur'] = True # ** 是否使用一个预编码层，它的作用是对输入 HxW 经过下采样得到 H/4xW/4 的feature，以便符合后面的网络
    network_G['HR_in'] = True    # ** 很重要！！只要你的输入与输出是同样分辨率，就要求设置为true
    network_G['w_TSA'] = True    # ** 是否使用TSA模块
    opt['network_G'] = network_G

    #datasets
    # datasets = dict()

    return opt

if __name__ == '__main__':
    opt = getOpt()
    main(opt)

