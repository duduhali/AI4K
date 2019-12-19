import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)

        self.netG.train()
        #### loss
        self.cri_pix = CharbonnierLoss().to(self.device) #pixel_criterion: cb
        self.l_pix_w = 1.0 #pixel_weight

        #### optimizers
        wd_G = 0

        optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        self.optimizer_G = torch.optim.Adam(optim_params, lr=4e-4,
                                            weight_decay=wd_G,
                                            betas=(0.9, 0.99))
        self.optimizers.append(self.optimizer_G)

        for optimizer in self.optimizers:
            self.schedulers.append(
                lr_scheduler.CosineAnnealingLR_Restart(optimizer, [150000, 150000, 150000, 150000], eta_min=1e-7,
                    restarts=[150000, 300000, 450000], weights=[1, 1, 1]))

        self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()


    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict


    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
