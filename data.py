import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

IMG_EXTENSIONS = ('.npy',)


def make_dataset(path):
    image_folders = os.path.join(path, 'image')
    action_folders = os.path.join(path, 'action')
    state_folders = os.path.join(path, 'state')

    if os.path.exists(image_folders) + os.path.exists(action_folders) + os.path.exists(state_folders) != 3:
        raise FileExistsError('some subfolders from data set do not exists!')

    samples = []
    for sample in os.listdir(image_folders):
        image, action, state = os.path.join(image_folders, sample), os.path.join(action_folders, sample), os.path.join(state_folders, sample)
        samples.append((image, action, state))
    return samples


def npy_loader(path):
    samples = torch.from_numpy(np.load(path))
    return samples


class PushDataset(Dataset):
    def __init__(self, root, image_transform=None, action_transform=None, state_transform=None, loader=npy_loader, device='cpu'):
        if not os.path.exists(root):
            raise FileExistsError('{0} does not exists!'.format(root))
        # self.subfolders = [f[0] for f in os.walk(root)][1:]
        self.image_transform = image_transform
        self.action_transform = action_transform
        self.state_transform = state_transform
        self.samples = make_dataset(root)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.loader = loader
        self.device = device
    def __getitem__(self, index):
        image, action, state = self.samples[index]
        image, action, state = self.loader(image), self.loader(action), self.loader(state)

        if self.image_transform is not None:
            image = torch.cat([self.image_transform(single_image).unsqueeze(0) for single_image in image.unbind(0)], dim=0)
        if self.action_transform is not None:
            action = torch.cat([self.action_transform(single_action).unsqueeze(0) for single_action in action.unbind(0)], dim=0)
        if self.state_transform is not None:
            state = torch.cat([self.state_transform(single_state).unsqueeze(0) for single_state in state.unbind(0)], dim=0)

        return image.to(self.device), action.to(self.device), state.to(self.device)

    def __len__(self):
        return len(self.samples)


def build_dataloader(opt):
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.height, opt.width)),
        transforms.ToTensor()
    ])

    train_ds = PushDataset(
        root=os.path.join(opt.data_dir, 'push_train'),
        image_transform=image_transform,
        loader=npy_loader,
        device=opt.device
    )

    testseen_ds = PushDataset(
        root=os.path.join(opt.data_dir, 'push_testseen'),
        image_transform=image_transform,
        loader=npy_loader,
        device=opt.device
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    testseen_dl = DataLoader(dataset=testseen_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return train_dl, testseen_dl


if __name__ == '__main__':
    from options import Options
    from matplotlib import  pyplot as plt
    from torchvision.transforms import functional as F
    opt = Options().parse()
    opt.device = 'cuda'
    dataloader, _ = build_dataloader(opt)
    for image, action, state in dataloader:
        # test = [F.to_pil_image(im) for im in image[0].unbind(dim=0)]
        # for tmp in test:
        #     plt.figure(1)
        #     plt.imshow(tmp)
        #     plt.show()
        pass