import torch
from torch import nn
from networks import network

class Model():
    def __init__(self, opt):
        self.opt = opt
        self.dataloader = None # input dataloader
        self.network = network(self.opt.channels, self.opt.height, self.opt.width, -1, self.opt.schedsamp_k,
                               self.opt.use_state, self.opt.num_masks, self.opt.model=='STP', self.opt.model=='CDNA', self.opt.model=='DNA', self.opt.context_frames)
        self.mse_loss = nn.MSELoss()
        self.w_state = 1e-4
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.opt.learning_rate)

    def train(self):
        for iter_, (image, action, state) in enumerate(self.dataloader['train']):
            # TODO: transpose Batch dimension and T dimension, for the state, only take the first time data to feed in

            gen_images, gen_states = self.network(image, state, action)

            # TODO: compute loss


            # TODO: optimize, priting stuff
            self.network.iter_num += 1

    def test(self):
        pass

