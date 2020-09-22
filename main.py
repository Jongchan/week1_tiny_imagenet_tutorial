import argparse
from time import gmtime, strftime
import os

import torch
from torch.utils.data import DataLoader

from arch.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from batch_manager import BatchManagerTinyImageNet
import train
import val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18'])
    parser.add_argument('--lr_base', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr_drop_epochs', type=int, default=[30, 60, 90], nargs='+')
    parser.add_argument('--lr_drop_rate', type=float, default=0.1)
    args = parser.parse_args()

    # define model
    if args.arch.startswith('resnet'):
        if args.arch == 'resnet18':
            model = resnet18(num_classes=200)
        elif args.arch == 'resnet34':
            model = resnet34(num_classes=200)
        elif args.arch == 'resnet50':
            model = resnet50(num_classes=200)
        elif args.arch == 'resnet101':
            model = resnet101(num_classes=200)
        elif args.arch == 'resnet152':
            model = resnet152(num_classes=200)
        else:
            raise NotImplementedError(f"architecture {args.arch} is not implemented")
    else:
        raise NotImplementedError(f"architecture {args.arch} is not implemented")
    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model)

    # define batch_manager
    dataloader_train = DataLoader(BatchManagerTinyImageNet(split='train'), shuffle=True, num_workers=10, batch_size=args.batch_size)
    dataloader_val = DataLoader(BatchManagerTinyImageNet(split='val'), shuffle=False, num_workers=10, batch_size=args.batch_size)

    # LR schedule
    lr = args.lr_base
    lr_per_epoch = []
    for epoch in range(args.epochs):
        if epoch in args.lr_drop_epochs:
            lr *= args.lr_drop_rate
        lr_per_epoch.append(lr)

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)

    # save_path
    current_time = strftime('%Y-%m-%d_%H:%M', gmtime())
    save_dir = os.path.join(f'checkpoints/{current_time}')
    os.makedirs(save_dir,  exist_ok=True)

    # train and val
    best_perform, best_epoch = -100, -100
    for epoch in range(1, args.epochs+1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_per_epoch[epoch]
        print(f"Training at epoch {epoch}. LR {lr_per_epoch[epoch]}")

        train.train(model, dataloader_train, criterion, optimizer, epoch=epoch)
        acc1, acc5 = val.val(model, dataloader_val, epoch=epoch)

        save_data = {'epoch': epoch,
                     'acc1': acc1,
                     'acc5': acc5,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        
        torch.save(save_data, os.path.join(save_dir, f'{epoch:03d}.pth.tar'))
        if epoch > 1:
            os.remove(os.path.join(save_dir, f'{epoch-1:03d}.pth.tar'))
        if acc1 >= best_perform:
            torch.save(save_data, os.path.join(save_dir, 'best.pth.tar'))
            best_perform = acc1
            best_epoch = epoch
        print(f"best performance {best_perform} at epoch {best_epoch}")