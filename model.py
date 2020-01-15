import torch
from torch import nn
from networks import network
from data import build_dataloader
from utils import peak_signal_to_noise_ratio

class Model():
    def __init__(self, opt):
        self.opt = opt
        train_dataloader, valid_dataloader = build_dataloader(opt)
        self.dataloader = {'train': train_dataloader, 'valid': valid_dataloader}
        self.network = network(self.opt.channels, self.opt.height, self.opt.width, -1, self.opt.schedsamp_k,
                               self.opt.use_state, self.opt.num_masks, self.opt.model=='STP', self.opt.model=='CDNA', self.opt.model=='DNA', self.opt.context_frames)
        self.mse_loss = nn.MSELoss()
        self.w_state = 1e-4
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.opt.learning_rate)

    def train_epoch(self, epoch):
        for iter_, (images, actions, states) in enumerate(self.dataloader['train']):
            # TODO: transpose Batch dimension and T dimension, for the state, only take the first time data to feed in
            images = images.permute([1, 0, 2, 3, 4]).unbind(0)
            actions = actions.permute([1, 0, 2]).unbind(0)
            states = states.permute([1, 0, 2]).unbind(0)
            gen_images, gen_states = self.network(images, actions, states[0])

            # TODO: compute loss
            loss, psnr= 0.0, 0.0
            for i, (image, gen_image) in enumerate(zip(images[self.opt.context_frames:], gen_images[self.opt.context_frames-1:])):
                recon_loss = self.mse_loss(image, gen_image)
                psnr_i = peak_signal_to_noise_ratio(image, gen_image)
                loss += recon_loss
                psnr += psnr_i

            for i, (state, gen_state) in enumerate(zip(states[self.opt.context_frames:], gen_states[self.opt.context_frames-1:])):
                state_loss = self.mse_loss(state, gen_state) * self.w_state
                loss += state_loss
            loss /= torch.tensor(len(images) - self.opt.context_frames)
            loss.backward()
            self.optimizer.step()

            if iter_ % self.opt.print_intercal == 0:
                print("epoch: %03d, iterations: %06d loss: %06f" % (epoch, iter_, loss))

            # TODO: optimize, priting stuff
            self.network.iter_num += 1

    def train(self):
        for epoch_i in range(1, self.opt.epochs):
            self.train_epoch(epoch_i)

    def test(self):
        pass

    def load_weight(self):
        pass
        # if self.opt.pret
        # self.network.load(torch.load(self.opt.)