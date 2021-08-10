# -*- ecoding: utf-8 -*-
# @ModuleName: main
# @Author: BetaHu
# @Time: 2021.8.5 19:24
import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MFI_WHU(Dataset):
    def __init__(self, path):
        super(MFI_WHU, self).__init__()
        self.path = path
        self.file_list = os.listdir(self.path + '/source_1')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(std=1, mean=0)
        ])

    def __getitem__(self, item):
        focus_1, _, _ = Image.open(self.path + '/source_1/' + self.file_list[item]).convert('YCbCr').split()
        focus_2, _, _ = Image.open(self.path + '/source_2/' + self.file_list[item]).convert('YCbCr').split()
        focus_f, _, _ = Image.open(self.path + '/image/' + self.file_list[item]).convert('YCbCr').split()
        return self.transform(focus_1), self.transform(focus_2), self.transform(focus_f)

    def __len__(self):
        return len(self.file_list)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_1_4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_2_4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.cat_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.cat_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.cat_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.cat_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_output = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x_1, x_2):
        x_1_1 = self.conv_1_1(x_1)
        x_2_1 = self.conv_2_1(x_2)

        x_1_2 = self.conv_1_2(x_1_1)
        x_2_2 = self.conv_2_2(x_2_1)
        x_1_2 = self.cat_1_1(torch.cat((x_1_2, x_2_2), dim=1))
        x_2_2 = self.cat_1_2(torch.cat((x_1_2, x_2_2), dim=1))

        x_1_3 = self.conv_1_3(torch.cat((x_1_1, x_1_2), dim=1))
        x_2_3 = self.conv_2_3(torch.cat((x_2_1, x_2_2), dim=1))
        x_1_3 = self.cat_2_1(torch.cat((x_1_3, x_2_3), dim=1))
        x_2_3 = self.cat_2_2(torch.cat((x_1_3, x_2_3), dim=1))

        x_1_4 = self.conv_1_4(torch.cat((x_1_1, x_1_2, x_1_3), dim=1))
        x_2_4 = self.conv_2_4(torch.cat((x_2_1, x_2_2, x_2_3), dim=1))

        output = self.conv_output(torch.cat((x_1_1, x_1_2, x_1_3, x_1_4, x_2_1, x_2_2, x_2_3, x_2_4), dim=1))

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.linear = nn.Linear(in_features=8 * 8 * 256, out_features=1, bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        [B, C, W, H] = x.shape
        x = self.linear(x.reshape(B, C * W * H))
        return x


def loss_g_adv(pred):
    return torch.mean((pred - torch.rand(1, dtype=pred.dtype, device='cuda:0') / 2 + 0.7) ** 2)


def loss_g_con(x_f, x_1, x_2):
    s_1 = torch.sign(blur_2th(x_1) - torch.min(blur_2th(x_1), blur_2th(x_2)))
    s_2 = 1 - s_1

    grad_1 = gradient(x_1)
    grad_2 = gradient(x_2)
    grad_f = gradient(x_f)

    loss_ints = torch.mean(s_1 * (x_f - x_1) ** 2 + s_2 * (x_f - x_2) ** 2)
    loss_grad = torch.mean(s_1 * (grad_f - grad_1) ** 2 + s_2 * (grad_f - grad_2) ** 2)
    return loss_ints + 5 * loss_grad


def loss_d_adv(p_fused, p_joint):
    return torch.mean((p_joint - torch.rand(1, dtype=p_joint.dtype, device='cuda:0') / 2 + 0.7) ** 2 +
                      (p_fused - torch.rand(1, dtype=p_fused.dtype, device='cuda:0') / 3) ** 2)


def blur_2th(x):
    kernel = torch.tensor([[[[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]]]],
                          dtype=x.dtype, device='cuda:0')
    blur = F.conv2d(x, kernel, padding=1)
    diff = torch.abs(x - blur)
    return diff


def gradient(x):
    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=x.dtype, device='cuda:0')
    grad = F.conv2d(x, kernel, padding=1)
    return grad


def joint_grad(x_f, x_1, x_2):
    grad_1 = torch.abs(gradient(x_1))
    grad_2 = torch.abs(gradient(x_2))
    grad_f = torch.abs(gradient(x_f))
    grad_j, _ = torch.max(torch.cat((grad_1, grad_2), dim=1), dim=1, keepdim=True)
    return grad_j, grad_f


if __name__ == '__main__':
    device = 'cuda:0'

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_g = torch.optim.Adam(params=generator.parameters(), lr=1e-4, eps=1e-4)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4, eps=1e-4)

    datas = MFI_WHU('../input/fusion')
    dataset = DataLoader(datas, batch_size=32, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    max_loss = 1e6
    print("|\tepoch\t|\tloss_g\t|\tloss_d\t|\ttime\t|")
    for epoch in range(1, 100):
        loss_d_epoch = 0
        loss_g_epoch = 0
        st = time()
        for idx, data in enumerate(dataset):
            image_1, image_2, _ = data
            image_1 = image_1.to(device)
            image_2 = image_2.to(device)

            if idx % 3 != 2:
                # train discriminator
                fused = generator(image_1, image_2).to(device)

                grad_j, grad_f = joint_grad(fused, image_1, image_2)
                neg_prob = discriminator(grad_j).to(device)
                pos_prob = discriminator(grad_f).to(device)

                loss_d = loss_d_adv(pos_prob, neg_prob)
                loss_d_epoch += loss_d.item()

                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

            else:
                # train generator
                fused = generator(image_1, image_2).to(device)

                _, grad_f = joint_grad(fused, image_1, image_2)
                pos_prob = discriminator(grad_f.to(device)).to(device)

                loss_g = loss_g_adv(pos_prob) + 10 * loss_g_con(fused, image_1, image_2)
                loss_g_epoch += loss_g.item()

                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

        if loss_g_epoch < max_loss:
            max_loss = loss_g_epoch
            torch.save(generator.state_dict(), './generator.pth')

        print("|\t%02d\t|\t%6.2f\t|\t%6.2f\t|\t%5.2f\t|" % (epoch, loss_g_epoch, loss_d_epoch, time() - st))
