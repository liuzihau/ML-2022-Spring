import os
from datetime import datetime
from config.config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import DataLoader

import numpy as np
import logging
from tqdm import tqdm

from model.dcgan import Generator as dc_G
from model.dcgan import Discriminator as dc_D

from model.wgan_gp import Generator as w_G
from model.wgan_gp import Discriminator as w_D

from dataset import get_dataset


class TrainerGAN:
    def __init__(self, config):
        self.config = config


        # model
        if self.config['model_type'] == "GAN":
            self.G = dc_G(100)
            self.D = dc_D(3)
        elif self.config['model_type'] == "WGAN-GP":
            cuda = True if torch.cuda.is_available() else False
            self.G = w_G(100)
            self.D = w_D(3)
            self.lambda_gp = config['lambda_gp']
            self.tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.loss = nn.BCELoss()

        # optim
        if self.config['model_type'] in ["GAN","WGAN-GP"]:
            self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
            self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        elif self.config['model_type'] == "STYLE-GAN":
            pass
        else:
            pass

        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')

        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')

        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).cuda()

    def prepare_environment(self):
        """
        Use this funciton to prepare function
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # create dataset by the above function
        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0)

        # model preparation
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.G.train()
        self.D.train()

    def train(self):
        """
        Use this function to train generator and discriminator
        """
        self.prepare_environment()
        epoch_start = 0
        if self.config["continue_train"]:
            G_epoch_list = sorted(
                [int(file.split('.')[0].split('_')[-1]) for file in os.listdir(self.ckpt_dir) if 'G' in file])
            D_epoch_list = sorted(
                [int(file.split('.')[0].split('_')[-1]) for file in os.listdir(self.ckpt_dir) if 'D' in file])
            if G_epoch_list:
                G_path = os.path.join(self.ckpt_dir, f'G_{G_epoch_list[-1]}.pth')
                self.G.load_state_dict(torch.load(G_path))
                epoch_start = G_epoch_list[-1]
                print(f"[Info] Continue training from epoch {epoch_start + 1}")
            if D_epoch_list:
                D_path = os.path.join(self.ckpt_dir, f'D_{D_epoch_list[-1]}.pth')
                self.D.load_state_dict(torch.load(D_path))

        for e, epoch in enumerate(range(epoch_start + 1, epoch_start + self.config["n_epoch"] + 1)):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {epoch + 1}")
            for i, data in enumerate(progress_bar):
                imgs = data.cuda()
                bs = imgs.size(0)

                # *********************
                # *    Train D        *
                # *********************
                z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                r_imgs = Variable(imgs).cuda()
                f_imgs = self.G(z)
                r_label = torch.ones((bs)).cuda()
                f_label = torch.zeros((bs)).cuda()

                # Discriminator forwarding
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                # Loss for discriminator
                if self.config['model_type'] == 'GAN':
                    r_loss = self.loss(r_logit, r_label)
                    f_loss = self.loss(f_logit, f_label)
                    loss_D = (r_loss + f_loss) / 2
                elif self.config['model_type'] == 'WGAN-GP':
                    gp = self.gp(r_imgs,f_imgs)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + self.lambda_gp * gp
                else:  # temporary use dcgan
                    r_loss = self.loss(r_logit, r_label)
                    f_loss = self.loss(f_logit, f_label)
                    loss_D = (r_loss + f_loss) / 2

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                """
                NOTE FOR SETTING WEIGHT CLIP:

                WGAN: below code
                
                WGAN-GP need CLIP?
                if config['model_type'] == 'WGAN-GP':
                    for p in self.D.parameters():
                        p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])
                """

                if self.steps % self.config["n_critic"] == 0:

                    # *******************
                    # *    Train G      *
                    # *******************
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                    f_imgs = self.G(z)

                    # Generator forwarding
                    f_logit = self.D(f_imgs)

                    # Loss for the generator.
                    if self.config['model_type'] == 'GAN':
                        loss_G = self.loss(f_logit, r_label)
                    elif self.config['model_type'] == 'WGAN-GP':
                        loss_G = -torch.mean(f_logit)
                    else:
                        loss_G = self.loss(f_logit, r_label)

                    # Generator backwarding
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch + 1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

            # Show some images during training.
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            torchvision.utils.save_image(grid_img, f'{self.config["workspace_dir"]}/sample/{self.config["model_type"]}_train_epoch_{e}_example.jpg')

            self.G.train()

            if (e + 1) % 5 == 0 or e == 0:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{epoch}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{epoch}.pth'))

        logging.info('Finish training')

    def inference(self, G_path, n_generate=1000, n_output=30, show=False):
        """
        1. G_path is the path for Generator ckpt
        2. You can use this function to generate final answer
        """

        self.G.load_state_dict(torch.load(G_path))
        self.G.cuda()
        self.G.eval()
        z = Variable(torch.randn(n_generate, self.config["z_dim"])).cuda()
        imgs = (self.G(z).data + 1) / 2.0

        os.makedirs('output', exist_ok=True)
        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], f'output/{i + 1}.jpg')

        if show:
            row, col = n_output // 10 + 1, 10
            grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            torchvision.utils.save_image(grid_img, f'{self.config["workspace_dir"]}/sample/inference_example.jpg')

    def gp(self, real_samples, fake_samples):
        """
        Implement gradient penalty function
        """
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        # d_interpolates = torch.unsqueeze(d_interpolates, 1)
        fake = Variable(self.tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        # print(d_interpolates.shape)
        # print(fake.shape)
        # import time
        # time.sleep(10)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
