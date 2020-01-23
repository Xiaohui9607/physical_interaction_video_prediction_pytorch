import math
import torch
from torch import nn
from torch.nn import functional as F

RELU_SHIFT = 1e-12
DNA_KERN_SIZE = 5
STATE_DIM = 5
ACTION_DIM = 5


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, forget_bias=1.0, padding=0):
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=out_channels + in_channels, out_channels=4 * out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.forget_bias = forget_bias

    def forward(self, inputs, states):
        if states is None:
            states = (torch.zeros([inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]], device=inputs.device),
                      torch.zeros([inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]], device=inputs.device))
        if not isinstance(states, tuple):
            raise TypeError("states type is not right")

        c, h = states
        if not (len(c.shape) == 4 and len(h.shape) == 4 and len(inputs.shape) == 4):
            raise TypeError("")

        inputs_h = torch.cat((inputs, h), dim=1)
        i_j_f_o = self.conv(inputs_h)
        i, j, f, o = torch.split(i_j_f_o,  self.out_channels, dim=1)

        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)

        return new_h, (new_c, new_h)


class network(nn.Module):
    def __init__(self, channels=3,
                 height=64,
                 width=64,
                 iter_num=-1.0,
                 k=-1,
                 use_state=True,
                 num_masks=10,
                 stp=False,
                 cdna=True,
                 dna=False,
                 context_frames=2):
        super(network, self).__init__()
        if stp + cdna + dna != 1:
            raise ValueError('More than one, or no network option specified.')
        lstm_size = [32, 32, 64, 64, 128, 64, 32]
        self.dna = dna
        self.stp = stp
        self.cdna = cdna
        self.channels = channels
        self.use_state = use_state
        self.num_masks = num_masks
        self.height = height
        self.width = width
        self.context_frames = context_frames
        self.k = k
        self.iter_num = iter_num

        self.STATE_DIM = STATE_DIM
        self.ACTION_DIM = ACTION_DIM
        if not self.use_state:
            self.STATE_DIM = 0
            self.ACTION_DIM = 0
        # N * 3 * H * W -> N * 32 * H/2 * W/2
        self.enc0 = nn.Conv2d(in_channels=channels, out_channels=lstm_size[0], kernel_size=5, stride=2, padding=2)
        self.enc0_norm = nn.LayerNorm([lstm_size[0], self.height//2, self.width//2])
        # N * 32 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm1 = ConvLSTM(in_channels=32, out_channels=lstm_size[0], kernel_size=5, padding=2)
        self.lstm1_norm = nn.LayerNorm([lstm_size[0], self.height//2, self.width//2])
        # N * 32 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm2 = ConvLSTM(in_channels=lstm_size[0], out_channels=lstm_size[1], kernel_size=5, padding=2)
        self.lstm2_norm = nn.LayerNorm([lstm_size[1], self.height//2, self.width//2])

        # N * 32 * H/4 * W/4 -> N * 32 * H/4 * W/4
        self.enc1 = nn.Conv2d(in_channels=lstm_size[1], out_channels=lstm_size[1], kernel_size=3, stride=2, padding=1)
        # N * 32 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm3 = ConvLSTM(in_channels=lstm_size[1], out_channels=lstm_size[2], kernel_size=5, padding=2)
        self.lstm3_norm = nn.LayerNorm([lstm_size[2], self.height//4, self.width//4])
        # N * 64 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm4 = ConvLSTM(in_channels=lstm_size[2], out_channels=lstm_size[3], kernel_size=5, padding=2)
        self.lstm4_norm = nn.LayerNorm([lstm_size[3], self.height//4, self.width//4])
        # pass in state and action

        # N * 64 * H/4 * W/4 -> N * 64 * H/8 * W/8
        self.enc2 = nn.Conv2d(in_channels=lstm_size[3], out_channels=lstm_size[3], kernel_size=3, stride=2, padding=1)
        # N * (10+64) * H/8 * W/8 -> N * 64 * H/8 * W/8
        self.enc3 = nn.Conv2d(in_channels=lstm_size[3]+self.STATE_DIM+self.ACTION_DIM, out_channels=lstm_size[3], kernel_size=1, stride=1)
        # N * 64 * H/8 * W/8 -> N * 128 * H/8 * W/8
        self.lstm5 = ConvLSTM(in_channels=lstm_size[3], out_channels=lstm_size[4], kernel_size=5, padding=2)
        self.lstm5_norm = nn.LayerNorm([lstm_size[4], self.height//8, self.width//8])
        # N * 128 * H/8 * W/8 -> N * 128 * H/4 * W/4
        self.enc4 = nn.ConvTranspose2d(in_channels=lstm_size[4], out_channels=lstm_size[4], kernel_size=3, stride=2, output_padding=1, padding=1)
        # N * 128 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm6 = ConvLSTM(in_channels=lstm_size[4], out_channels=lstm_size[5], kernel_size=5, padding=2)
        self.lstm6_norm = nn.LayerNorm([lstm_size[5], self.height//4, self.width//4])

        # N * 64 * H/4 * W/4 -> N * 64 * H/2 * W/2
        self.enc5 = nn.ConvTranspose2d(in_channels=lstm_size[5]+lstm_size[1], out_channels=lstm_size[5]+lstm_size[1], kernel_size=3, stride=2, output_padding=1, padding=1)
        # N * 64 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm7 = ConvLSTM(in_channels=lstm_size[5]+lstm_size[1], out_channels=lstm_size[6], kernel_size=5, padding=2)
        self.lstm7_norm = nn.LayerNorm([lstm_size[6], self.height//2, self.width//2])
        # N * 32 * H/2 * W/2 -> N * 32 * H * W
        self.enc6 = nn.ConvTranspose2d(in_channels=lstm_size[6]+lstm_size[0], out_channels=lstm_size[6], kernel_size=3, stride=2, output_padding=1, padding=1)
        self.enc6_norm = nn.LayerNorm([lstm_size[6], self.height, self.width])

        if self.dna:
            # N * 32 * H * W -> N * (DNA_KERN_SIZE*DNA_KERN_SIZE) * H * W
            self.enc7 = nn.ConvTranspose2d(in_channels=lstm_size[6], out_channels=DNA_KERN_SIZE**2, kernel_size=1, stride=1)
        else:
            # N * 32 * H * W -> N * 3 * H * W
            self.enc7 = nn.ConvTranspose2d(in_channels=lstm_size[6], out_channels=channels, kernel_size=1, stride=1)
            if self.cdna:
                # a reshape from lstm5: N * 128 * H/8 * W/8 -> N * (128 * H/8 * W/8)
                # N * (128 * H/8 * W/8) -> N * (10 * 5 * 5)
                in_dim = int(lstm_size[4] * self.height * self.width / 64)
                self.fc = nn.Linear(in_dim, DNA_KERN_SIZE * DNA_KERN_SIZE * self.num_masks)
            else:
                in_dim = int(lstm_size[4] * self.height * self.width / 64)
                self.fc = nn.Linear(in_dim, 100)
                self.fc_stp = nn.Linear(100, (self.num_masks-1) * 6)
        #  N * 32 * H * W -> N * 11 * H * W
        self.maskout = nn.ConvTranspose2d(lstm_size[6], self.num_masks+1, kernel_size=1, stride=1)
        self.stateout = nn.Linear(STATE_DIM+ACTION_DIM, STATE_DIM)

    def forward(self, images, actions, init_state):
        '''

        :param inputs: T * N * C * H * W
        :param state: T * N  * C
        :param action: T * N * C
        :return:
        '''

        lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
        lstm_state5, lstm_state6, lstm_state7 = None, None, None
        gen_images, gen_states = [], []
        current_state = init_state
        if self.k == -1:
            feedself = True
        else:
            num_ground_truth = round(images[0].shape[1] * (self.k / (math.exp(self.iter_num/self.k) + self.k)))
            feedself = False

        for image, action in zip(images[:-1], actions[:-1]):

            done_warm_start = len(gen_images) >= self.context_frames

            if feedself and done_warm_start:
                # Feed in generated image.
                image = gen_images[-1]
            elif done_warm_start:
                # Scheduled sampling
                image = self.scheduled_sample(image, gen_images[-1], num_ground_truth)
            else:
                # Always feed in ground_truth
                image = image

            enc0 = self.enc0_norm(torch.relu(self.enc0(image)))

            lstm1, lstm_state1 = self.lstm1(enc0, lstm_state1)
            lstm1 = self.lstm1_norm(lstm1)

            lstm2, lstm_state2 = self.lstm2(lstm1, lstm_state2)
            lstm2 = self.lstm2_norm(lstm2)

            enc1 = torch.relu(self.enc1(lstm2))

            lstm3, lstm_state3 = self.lstm3(enc1, lstm_state3)
            lstm3 = self.lstm3_norm(lstm3)

            lstm4, lstm_state4 = self.lstm4(lstm3, lstm_state4)
            lstm4 = self.lstm4_norm(lstm4)

            enc2 = torch.relu(self.enc2(lstm4))

            # pass in state and action
            state_action = torch.cat([action, current_state], dim=1)
            smear = torch.reshape(state_action, list(state_action.shape)+[1, 1])
            smear = smear.repeat(1, 1, enc2.shape[2], enc2.shape[3])
            if self.use_state:
                enc2 = torch.cat([enc2, smear], dim=1)
            enc3 = torch.relu(self.enc3(enc2))

            lstm5, lstm_state5 = self.lstm5(enc3, lstm_state5)
            lstm5 = self.lstm5_norm(lstm5)
            enc4 = torch.relu(self.enc4(lstm5))

            lstm6, lstm_state6 = self.lstm6(enc4, lstm_state6)
            lstm6 = self.lstm6_norm(lstm6)
            # skip connection
            lstm6 = torch.cat([lstm6, enc1], dim=1)

            enc5 = torch.relu(self.enc5(lstm6))

            lstm7, lstm_state7 = self.lstm7(enc5, lstm_state7)
            lstm7 = self.lstm7_norm(lstm7)
            # skip connection
            lstm7 = torch.cat([lstm7, enc0], dim=1)

            enc6 = self.enc6_norm(torch.relu(self.enc6(lstm7)))

            enc7 = torch.relu(self.enc7(enc6))

            if self.dna:
                if self.num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [self.dna_transformation(image, enc7)]
            else:
                transformed = [torch.sigmoid(enc7)]
                _input = lstm5.view(lstm5.shape[0], -1)
                if self.cdna:
                    transformed += self.cdna_transformation(image, _input)
                else:
                    transformed += self.stp_transformation(image, _input)

            masks = torch.relu(self.maskout(enc6))
            masks = torch.softmax(masks, dim=1)
            mask_list = torch.split(masks, split_size_or_sections=1, dim=1)

            output = mask_list[0] * image
            for layer, mask in zip(transformed, mask_list[1:]):
                output += layer * mask

            gen_images.append(output)

            current_state = self.stateout(state_action)
            gen_states.append(current_state)

        return gen_images, gen_states

    def stp_transformation(self, image, stp_input):
        identity_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(1).repeat(1, self.num_masks-1)

        stp_input = self.fc(stp_input)
        stp_input = self.fc_stp(stp_input)
        stp_input = stp_input.view(-1, 6, self.num_masks-1) + identity_params
        params = torch.unbind(stp_input, dim=-1)

        transformed = [F.grid_sample(image, F.affine_grid(param.view(-1, 3, 2), image.size())) for param in params]
        return transformed

    def cdna_transformation(self, image, cdna_input):
        batch_size, height, width = image.shape[0], image.shape[2], image.shape[3]

        cdna_kerns = self.fc(cdna_input)
        cdna_kerns = cdna_kerns.view(batch_size, self.num_masks, 1, DNA_KERN_SIZE, DNA_KERN_SIZE)
        cdna_kerns = torch.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = torch.sum(cdna_kerns, dim=[2,3,4], keepdim=True)
        cdna_kerns /= norm_factor

        cdna_kerns = cdna_kerns.view(batch_size*self.num_masks, 1, DNA_KERN_SIZE,DNA_KERN_SIZE)
        image = image.permute([1, 0, 2, 3])

        transformed = torch.conv2d(image, cdna_kerns, stride=1, padding=[2, 2], groups=batch_size)

        transformed = transformed.view(self.channels, batch_size, self.num_masks, height, width)
        transformed = transformed.permute([1, 0, 3, 4, 2])
        transformed = torch.unbind(transformed, dim=-1)

        return transformed

    def dna_transformation(self, image, dna_input):
        image_pad = F.pad(image, [2, 2, 2, 2, 0, 0, 0, 0], "constant", 0)
        height, width = image.shape[2], image.shape[3]

        inputs = []

        for xkern in range(DNA_KERN_SIZE):
            for ykern in range(DNA_KERN_SIZE):
                inputs.append(image_pad[:, :, xkern:xkern+height, ykern:ykern+width].clone().unsqueeze(dim=1))
        inputs = torch.cat(inputs, dim=4)

        kernel = torch.relu(dna_input-RELU_SHIFT)+RELU_SHIFT
        kernel = kernel / torch.sum(kernel, dim=1, keepdim=True).unsqueeze(2)

        return torch.sum(kernel*inputs, dim=1, keepdim=False)

    def scheduled_sample(self, ground_truth_x, generated_x, num_ground_truth):
        generated_examps = torch.cat([ground_truth_x[:num_ground_truth, ...], generated_x[num_ground_truth:, :]], dim=0)
        return generated_examps



