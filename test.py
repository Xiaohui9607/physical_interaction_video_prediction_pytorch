import numpy as np
import  torch
import  matplotlib.pyplot as plt
from torchvision import  transforms
data = torch.from_numpy(np.load("/home/golf/code/physical_interaction_video_prediction_pytorch/data/processed/push/push_train/image/batch_0_0.npy"))
image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor()
    ])

for d in data:
    # d = image_transform(d)
    plt.imshow(d.squeeze().detach().cpu().numpy().transpose([1, 2, 0])/255.0)
    plt.show()