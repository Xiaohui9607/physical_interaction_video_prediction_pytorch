import torch
from torch.nn import functional as F

def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * torch.log(torch.tensor(1.0) / F.mse_loss(true, pred)) / torch.log(torch.tensor(10.0))