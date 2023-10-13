"""Train CIFAR10 with PyTorch."""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from resnet import *
import utils
from lip_conv import bounds


class Trainer:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self.name_dir = "runs/" + self.args.evalfile
        self.best_acc = 0  # best test accuracy
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device", self.device)

        # Data
        print("==> Preparing data..")
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.args.bs, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2
        )

        self.net = ResNet18()
        self.net = self.net.to(self.device)
        self.do_parallel_cuda = True
        if self.do_parallel_cuda and self.device == "cuda":
            devices = list(range(torch.cuda.device_count()))
            print("devices", devices)
            self.net = nn.DataParallel(self.net, device_ids=devices)
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=self.args.wd,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs
        )

        # Training Loop
        for epoch in range(0, self.args.epochs):
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()

    def train(self, epoch):
        # Training
        print("\nEpoch: %d" % epoch)
        self.net.train()
        sum_train_loss = 0
        sum_reg_loss = 0
        correct = 0
        total = 0
        start = time.time()

        if self.args.adaptative_bound_n_iter:
            self.args.bound_n_iter = utils.get_n_iter_epoch(epoch)
        print("bound_n_iter", self.args.bound_n_iter)

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            train_loss = self.criterion(outputs, targets)

            reg_loss_conv = torch.tensor([0.0], requires_grad=True).to(
                device=train_loss.device
            )
            if self.args.r:
                for (kernel, input_size) in self.net.module.get_all_kernels():
                    bound = bounds.estimate(
                        kernel,
                        n=input_size[0],
                        name_func=self.args.bound,
                        n_iter=self.args.bound_n_iter,
                    )
                    reg_loss_conv = reg_loss_conv + F.threshold(
                        bound, self.args.threshold_reg, 0.0
                    )
                reg_loss_conv = self.args.r * reg_loss_conv

            loss = train_loss + reg_loss_conv
            loss.backward()
            self.optimizer.step()

            sum_train_loss += train_loss.item()
            sum_reg_loss += reg_loss_conv.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        total_time = time.time() - start
        log_train_loss = sum_train_loss / (batch_idx + 1)
        log_reg_loss = sum_reg_loss / (batch_idx + 1)

        print("LossTrain", log_train_loss, "LossReg", log_reg_loss)
        print("Time Epoch", total_time)

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        normalize_test_loss = test_loss / (batch_idx + 1)
        accuracy_test = 100.0 * correct / total
        with open(self.args.evalfile, "a") as f:
            f.write(
                f"epoch {epoch}"
                f" lr {self.args.lr}"
                f" test_loss {normalize_test_loss}"
                f" test_accuracy {accuracy_test}\n"
            )
        print("Accuracy test", accuracy_test)
        # Save checkpoint.
        accuracy_test = 100.0 * correct / total
        if accuracy_test > self.best_acc:
            print("Saving..")
            state = {
                "net": self.net.state_dict(),
                "acc": accuracy_test,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/ckpt.pth")
            self.best_acc = accuracy_test
