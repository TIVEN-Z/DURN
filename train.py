import argparse
import os
import time
from os.path import join, isdir
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.utils.data import DataLoader
from datasets_mural.data_one2one_mural import MURAL_Loader
from models.DURN import DURN
from utils import save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=16, type=int, metavar='BT', help='batch size')
parser.add_argument('--LR', '--learning_rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int, metavar='SS', help='learning rate step size')
parser.add_argument('--maxepoch', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
parser.add_argument('--tmp', help='tmp folder', default='./tmp/')
parser.add_argument('--dataset', help='root folder of dataset', default='../DATA/MURAL/')
parser.add_argument('--itersize', default=1, type=int, metavar='IS', help='iter size')
parser.add_argument('--std_weight', default=1, type=float, help='weight for std loss')
parser.add_argument('--size', default=320, type=int, help='train images size')
parser.add_argument('--model', default="DURN", help='method name')
args = parser.parse_args()

random_seed = 666
if random_seed > 0:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed)


def cross_entropy_loss(prediction, labelef):
    label = labelef.long()
    mask = label.float()
    num_positive = torch.sum((mask == 0).float()).float()
    num_negative = torch.sum((mask == 1).float()).float()
    num_two = torch.sum((mask == 2).float()).float()
    assert num_negative + num_positive + num_two == label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]
    assert num_two == 0
    mask[mask == 0] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 1] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = F.binary_cross_entropy(
        prediction, labelef, weight=mask.detach(), reduction='sum')
    return cost


def step_lr_scheduler(optimizer, epoch, init_lr=args.LR, lr_decay_epoch=3):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def main():
    args.cuda = True
    train_dataset = MURAL_Loader(root=args.dataset, split="train", size=args.size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32, shuffle=True)
    model = DURN().cuda()  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    start_time = time.time()

    train_info = (
        'LR:{1},WD:{2},dataset:{3},batch_size:{4},epoch:{5},size:{6},model:{8}'.
        format(args.LR, args.weight_decay, args.dataset, args.batch_size, args.maxepoch,
               args.size, args.model))
    print(train_info)

    if not isdir(args.tmp + args.model):
        os.makedirs(args.tmp + args.model)

    file_name = args.tmp + args.model + '/' + 'log.txt'
    with open(file_name, 'w') as f:
        f.write(train_info + '\n')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    for epoch in range(args.start_epoch, args.maxepoch):
        train(train_loader, model, optimizer, epoch, save_dir=join(args.tmp + args.model, 'epoch-%d' % epoch))

    end_time = time.time()
    all_time = end_time - start_time
    print("Time:", all_time)


def train(train_loader, model, optimizer, epoch, save_dir):
    optimizer = step_lr_scheduler(optimizer, epoch)
    # switch to train mode
    model.train()
    # print(epoch, optimizer.state_dict()['param_groups'][0]['lr'])
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        image, label = image.cuda(), label.cuda()
        mean, std = model(image)
        # ------------------------------------------------------
        outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
        outputs = torch.sigmoid(outputs_dist.rsample())
        LP = cross_entropy_loss(outputs, label)
        LVE = torch.sum(((std - label) ** 2))
        loss = LP + LVE
        counter += 1
        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            currect_time = time.ctime()
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {time} '.format(time=currect_time) + \
                   'Loss {loss:f} '.format(loss=loss)
            print(info)
            with open(args.tmp + args.model + '/' + 'log.txt', 'a') as f:
                f.write(info + '\n')

        # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))


if __name__ == '__main__':
    main()

