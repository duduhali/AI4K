import time
import numpy as np
import torch
import math
from torch.autograd import Variable


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def psnr_cal(pred, gt):
    batch = pred.shape[0]
    psnr = 0
    for i in range(batch):
        for j in range(3):
            pr = pred[i, j, :, :]
            hd = gt[i, j, :, :]

            imdff = pr - hd
            rmse = math.sqrt(np.mean(imdff ** 2))
            if rmse == 0:
                psnr = psnr + 45
                continue
            psnr = psnr + 20 * math.log10(255.0 / rmse)
    return psnr / (batch*3)


def train(training_data_loader, optimizer, model, criterion, epoch, opt, iterations):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    psnrs = AverageMeter()

    model.train()
    end = time.time()
    for iteration, batch in enumerate(training_data_loader):
        data_time.update(time.time() - end)
        data_x, data_y = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        data_x = data_x.type(torch.FloatTensor)
        data_y = data_y.type(torch.FloatTensor)

        data_x = data_x.cuda()
        data_y = data_y.cuda()

        pred = model(data_x)
        # pix loss
        loss = criterion(pred, data_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = pred.cpu()
        pred = pred.detach().numpy().astype(np.float32)

        data_y = data_y.cpu()
        data_y = data_y.numpy().astype(np.float32)

        psnr = psnr_cal(pred, data_y)

        mean_loss = loss.item() / (opt.batch_size*opt.n_colors*((opt.patch_size*opt.scale)**2))
        losses.update(mean_loss)
        psnrs.update(psnr)

        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % opt.print_freq == 0:
            print('Epoch:[{0}/{1}][{2}/{3}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {losses.val:.3f} ({losses.avg:.3f})\t'
                  'PNSR: {psnrs.val:.3f} ({psnrs.avg:.3f})'
                  .format(epoch, opt.epochs, iteration, iterations//opt.batch_size,
                          batch_time=batch_time, data_time=data_time, losses=losses, psnrs=psnrs))
