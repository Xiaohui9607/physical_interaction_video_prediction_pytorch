import os
import numpy as np
import torch

from options import Options
from model import Model
from torchvision.transforms import functional as F

opt = Options().parse()


def save_to_local(tensor_list, folder):
    for idx_, tensor in enumerate(tensor_list):
        img = F.to_pil_image(tensor.squeeze())
        img.save(os.path.join(folder, "predict_%s.jpg" % idx_))


def predict(net, data, save_path=None):
    images, actions, states = data
    images = [F.to_tensor(F.resize(F.to_pil_image(im), (opt.height, opt.width))).unsqueeze(0).to(opt.device)
              for im in torch.from_numpy(images).unbind(0)]
    actions = [ac.unsqueeze(0).to(opt.device) for ac in torch.from_numpy(actions).unbind(0)]
    states = [st.unsqueeze(0).to(opt.device) for st in torch.from_numpy(states).unbind(0)]

    with torch.no_grad():
        gen_images, gen_states = net(images, actions, states[0])
        save_images = images[:opt.context_frames] + gen_images[opt.context_frames-1:]
        if save_path:
            save_to_local(save_images, save_path)


if __name__ == '__main__':
    images, actions, states = np.load("data/processed/push/push_testseen/image/batch_1_0.npy"), \
                           np.load("data/processed/push/push_testseen/action/batch_1_0.npy"), \
                           np.load("data/processed/push/push_testseen/state/batch_1_0.npy")

    m = Model(opt)
    # m.load_weight()
    net = m.net

    predict(net, (images, actions, states), save_path="predict/")




