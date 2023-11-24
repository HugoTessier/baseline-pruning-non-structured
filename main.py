"""Basic script to train CIFAR10 and ImageNet close to SOTA with ResNets"""

import torch
import torch.nn as nn

import os
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import time
import numpy as np
from accelerate import Accelerator
import thop
import resnet
from utils import ExponentialMovingAverage, RandomMixup, RandomCutmix
from torch.utils.data.dataloader import default_collate
from pruning_utils import mask_network, apply_pruning_and_unmask_network
import copy


def main():
    accelerator = Accelerator(split_batches=True)

    args = parse_args()

    # deterministic mode for reproducibility
    set_deterministic_mode(accelerator, args)

    # prepare dataloaders
    input_size, large_input, num_classes, test_loader, train_loader = create_dataset(args)

    # Prepare model, EMA and parameter sets
    net, num_parameters = create_model(accelerator, args, input_size, large_input, num_classes)
    net, train_loader, test_loader = accelerator.prepare(net, train_loader, test_loader)
    ema = ExponentialMovingAverage(net, decay=args.ema_decay)
    ema.to(accelerator.device)
    ema.eval()

    if args.pruning_rate != 0.:
        mask_network(net, args.pruning_rate, args.pruning_type)

    nowd, wd = separates_wd_and_nowd_params(net, num_parameters)

    # define criterion and aggregators
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_losses = []
    peak, peak_step, peak_ema, peak_step_ema = 0, 0, 0, 0

    if args.test_only:
        test(net, ema, test_loader, accelerator)
        exit()

    start_time = 0
    epoch = 0
    target, idx_target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0

    net.train()
    last_print = 0
    for era in range(1 if args.adam or args.no_warmup else 0, args.eras + 1):
        step, idx_target = 0, 0
        if accelerator.is_main_process:
            accelerator.print("{:s}".format("Era " + str(era) if era > 0 else "Warming up"))

        # define optimizers/schedulers
        optimizer, scheduler = update_optimizer_and_scheduler(accelerator, args, era, nowd, train_loader, wd)

        total_steps_for_era = args.steps if era > 0 else 5 * len(train_loader)

        if start_time == 0:
            start_time = time.time()

        while step < total_steps_for_era:
            remaining_time, step = train_one_epoch(accelerator, args, criterion, ema, era, last_print, net, optimizer,
                                                   scheduler, start_time, step, total_steps_for_era, train_loader,
                                                   train_losses)

            score, score_ema = test(net, ema, test_loader, accelerator)
            if 100 * score > peak:
                peak = 100 * score
                peak_step = epoch
                if args.save_model != "":
                    unmasked_net = copy.deepcopy(net)
                    apply_pruning_and_unmask_network(unmasked_net)
                    torch.save(unmasked_net.state_dict(), args.save_model + "_best.pt")
            if 100 * score_ema > peak_ema:
                peak_ema = 100 * score_ema
                peak_step_ema = epoch
                if args.save_model != "":
                    unmasked_ema = copy.deepcopy(ema.module)
                    apply_pruning_and_unmask_network(unmasked_ema)
                    torch.save(unmasked_ema.state_dict(), args.save_model + "_best_ema.pt")
            accelerator.print(
                " {:4d}h{:02d}m epoch {:4d}".format(int(remaining_time / 3600),
                                                    (int(remaining_time) % 3600) // 60,
                                                    epoch + 1), end='')
            epoch += 1
            if (era == 0 and step >= total_steps_for_era) or (
                    era > 0 and step / total_steps_for_era > target[idx_target]):
                accelerator.print()
                if era > 0:
                    idx_target += 1

    total_time = time.time() - start_time
    accelerator.print()
    accelerator.print("total time is {:4d}h{:02d}m".format(int(total_time / 3600),
                                                           (int(total_time) % 3600) // 60))
    accelerator.print(
        "Peak perf is {:6.2f}% at epoch {:d} ({:6.2f}% at epoch {:d})".format(peak, peak_step, peak_ema,
                                                                              peak_step_ema))
    accelerator.print()
    accelerator.print()

    if args.save_model != "":
        unmasked_net = copy.deepcopy(net)
        apply_pruning_and_unmask_network(unmasked_net)
        unmasked_ema = copy.deepcopy(ema.module)
        apply_pruning_and_unmask_network(unmasked_ema)
        torch.save(unmasked_net.state_dict(), args.save_model + ".pt")
        torch.save(unmasked_ema.state_dict(), args.save_model + "_ema.pt")


def train_one_epoch(accelerator, args, criterion, ema, era, last_print, net, optimizer, scheduler, start_time, step,
                    total_steps_for_era, train_loader, train_losses):
    remaining_time = None
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        step += 1

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()
        if step % 32 == 0:
            ema.update_parameters(net)
            if era == 0:
                ema.n_averaged.fill_(0)

        train_losses.append(loss.item())
        train_losses = train_losses[-len(train_loader):]

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        if time.time() - last_print > 0.05 or batch_idx + 1 == len(train_loader):
            accelerator.print("\r{:6.2f}% loss:{:.4e} lr:{:.3e}".format(100 * step / total_steps_for_era,
                                                                        torch.mean(torch.tensor(train_losses)).item(),
                                                                        lr), end="")
            last_print = time.time()

        step_time = (time.time() - start_time) / (args.steps * (era - 1 if era > 0 else 0) + step + (
            5 * len(train_loader) if not args.adam and era > 0 else 0))
        remaining_time = (total_steps_for_era - step + (args.eras - era) * args.steps) * step_time
    return remaining_time, step


def update_optimizer_and_scheduler(accelerator, args, era, nowd, train_loader, wd):
    if era == 0:
        optimizer = torch.optim.SGD([{"params": wd, "weight_decay": args.weight_decay},
                                     {"params": nowd, "weight_decay": 0}],
                                    lr=0.5, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=len(train_loader) * 5)
    else:
        if args.adam:
            optimizer = torch.optim.AdamW([{"params": wd, "weight_decay": args.weight_decay},
                                           {"params": nowd, "weight_decay": 0}],
                                          lr=1e-3 * (0.9 ** (era - 1)))
        else:
            optimizer = torch.optim.SGD([{"params": wd, "weight_decay": args.weight_decay},
                                         {"params": nowd, "weight_decay": 0}],
                                        lr=0.5 * (0.9 ** (era - 1)), momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, args.eta_min)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    return optimizer, scheduler


def test(net, ema, test_loader, accelerator):
    net.eval()
    correct = 0
    total = 0
    correct_ema = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = net(inputs)
            outputs_ema = ema(inputs)
            _, predicted = outputs.max(1)
            _, predicted_ema = outputs_ema.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()
            correct_ema += predicted_ema.eq(targets).sum()
    accelerator.print("{:6.2f}% (ema: {:6.2f}%)".format(100. * correct / total, 100. * correct_ema / total), end='')
    net.train()
    return correct / total, correct_ema / total


def separates_wd_and_nowd_params(net, num_parameters):
    modules = [x for x in net.modules()]
    wd = []
    nowd = []
    trained_parameters = 0
    for x in modules:
        if isinstance(x, nn.BatchNorm2d):
            nowd.append(x.weight)
            trained_parameters += x.weight.numel()
            nowd.append(x.bias)
            trained_parameters += x.bias.numel()
        elif isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
            wd.append(x.weight)
            trained_parameters += x.weight.numel()
            if hasattr(x, 'bias'):
                if x.bias is not None:
                    wd.append(x.bias)
                    trained_parameters += x.bias.numel()
        elif hasattr(x, 'gamma'):
            if isinstance(x.gamma, nn.Parameter):
                wd.append(x.gamma)
                trained_parameters += x.gamma.numel()
    assert (num_parameters == trained_parameters)
    return nowd, wd


def create_model(accelerator, args, input_size, large_input, num_classes):
    if args.model in resnet.__all__:
        net = eval("resnet." + args.model)(num_classes=num_classes, large_input=large_input, width=args.width)
    else:
        net = eval(args.model)
    new_size = False
    if args.load_model != "":
        loaded_dict = torch.load(args.load_model, map_location="cpu")
        if list(loaded_dict.keys())[0][:6] == "module":
            loaded_dict = {k.replace("module.", ""): v for k, v in loaded_dict.items()}
        current_dict = net.state_dict()
        for key in current_dict.keys():
            if current_dict[key].shape == loaded_dict[key].shape:
                current_dict[key] = loaded_dict[key]
            else:
                new_size = True
                cd = current_dict[key]
                ld = loaded_dict[key]
                if len(cd.shape) == 1:
                    cd = ld[:cd.shape[0]]
                elif len(cd.shape) == 2:
                    cd = ld[:cd.shape[0], :cd.shape[1]]
                elif len(cd.shape) == 3:
                    cd = ld[:cd.shape[0], :cd.shape[1], :cd.shape[2]]
                elif len(cd.shape) == 4:
                    cd = ld[:cd.shape[0], :cd.shape[1], :cd.shape[2], :cd.shape[3]]
                else:
                    print("Error in loading model!!!")
                    exit()
        net.load_state_dict(current_dict)
    if new_size:
        print("WARNING!!!! CHANGE OF SIZE WHEN LOADING MODEL!!!!")
    if accelerator.is_main_process:
        macs, params = thop.profile(net, inputs=(torch.rand(input_size),), verbose=False)
        accelerator.print("Total mult-adds: {:d} ({:,})".format(int(macs), int(params)))
    num_parameters = int(torch.tensor([x.numel() for x in net.parameters()]).sum().item())
    accelerator.print("Params: {:d} ({:,})".format(num_parameters, num_parameters))
    return net, num_parameters


def create_dataset(args):
    input_size, large_input, num_classes, train_dataset, test_dataset = None, None, None, None, None
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dataset.lower() == "imagenet":
        train_dataset = torchvision.datasets.ImageNet(
            root=args.dataset_path,
            split="train",
            transform=transforms.Compose([
                transforms.RandomResizedCrop(176, antialias=True),
                transforms.RandomHorizontalFlip(),
                torchvision.transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                normalize,
                torchvision.transforms.RandomErasing(0.1)
            ]))
        test_dataset = torchvision.datasets.ImageNet(
            root=args.dataset_path,
            split="val",
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(232, antialias=True),
                transforms.CenterCrop(224),
                normalize]))
        num_classes, large_input, input_size = 1000, True, (1, 3, 224, 224)
    if args.dataset.lower() == "cifar10" or args.dataset.lower() == "cifar100":
        if args.dataset.lower() == "cifar10":
            tvdset = torchvision.datasets.CIFAR10
            num_classes = 10
        else:
            tvdset = torchvision.datasets.CIFAR100
            num_classes = 100
        train_dataset = tvdset(
            root=args.dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                normalize,
                transforms.Resize(args.cifar_resize, antialias=True),
                transforms.RandomErasing(0.1)
            ]))
        test_dataset = tvdset(
            root=args.dataset_path,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
                transforms.Resize(args.cifar_resize, antialias=True)
            ]))
        large_input = False
        input_size = (1, 3, args.cifar_resize, args.cifar_resize)
    mixupcutmix = torchvision.transforms.RandomChoice(
        [RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha),
         RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha)])

    def collate_fn(batch):
        return mixupcutmix(*default_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=min(10, os.cpu_count()), drop_last=True, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(10, os.cpu_count()), pin_memory=True,
        persistent_workers=True)
    return input_size, large_input, num_classes, test_loader, train_loader


def set_deterministic_mode(accelerator, args):
    if args.seed >= 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        if accelerator.is_main_process:
            accelerator.print("Random seed:", args.seed)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, default="CIFAR10", help="CIFAR10, CIFAR100 or ImageNet")
    parser.add_argument('--steps', type=int, default=750000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--width', type=int, default=64, help="number of feature maps for first layers")
    parser.add_argument('--dataset-path', type=str, default=os.getenv("DATASETS"))
    parser.add_argument('--cifar-resize', type=int, default=32)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--adam', action="store_true")
    parser.add_argument('--eta-min', type=float, default=0)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=-1)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--cutmix-alpha', type=float, default=1.)
    parser.add_argument('--eras', type=int, default=1)
    parser.add_argument('--save-model', type=str, default="")
    parser.add_argument('--load-model', type=str, default="")
    parser.add_argument('--test-only', action="store_true")
    parser.add_argument('--no-warmup', action="store_true")
    parser.add_argument('--pruning-rate', type=float, default=0.)
    parser.add_argument('--pruning-type', type=str, default='random', help='Either "random" or "regular"')
    args = parser.parse_args()
    args.steps = 10 * (args.steps // 10)
    if args.weight_decay < 0:
        args.weight_decay = 0.05 if args.adam else 2e-5
    return args


if __name__ == '__main__':
    main()
