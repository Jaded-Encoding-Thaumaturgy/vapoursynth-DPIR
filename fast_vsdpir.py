#
# This is not a wrapper. This is used to trace the model, but you can also just use it as a replacement to vs-dpir by HolyWu
#

from typing import Optional
import torch
import os.path
import numpy as np
import torch.nn as nn
import vapoursynth as vs
from functools import partial

core = vs.core


def conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, device=None):
  return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, device=device)


def refield_tensor(img_L, model):
  refield = 64
  h, w = img_L.size()[-2:]
  r_h, r_w = h // 2, w // 2
  s_h, s_w = (r_h // refield + 1) * refield, (r_w // refield + 1) * refield

  top = slice(0, s_h)
  bottom = slice(h - s_h, h)
  left = slice(0, s_w)
  right = slice(w - s_w, w)

  Ls = [img_L[..., top, left], img_L[..., top, right], img_L[..., bottom, left], img_L[..., bottom, right]]

  Es = [model.forward(Ls[i]) for i in range(4)]

  b, c = Es[0].size()[:2]

  E = torch.zeros(b, c, h, w).type_as(img_L)

  E[..., : r_h, :r_w] = Es[0][..., :r_h, :r_w]
  E[..., : r_h, r_w:w] = Es[1][..., :r_h, (-w + r_w):]
  E[..., r_h:h, :r_w] = Es[2][..., (-h + r_h):, :r_w]
  E[..., r_h:h, r_w:w] = Es[3][..., (-h + r_h):, (-w + r_w):]

  return E


def upsample_convtranspose(in_channels, out_channels, device):
  return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False, device=device)


def downsample_strideconv(in_channels, out_channels, device):
  return conv2D(in_channels, out_channels, 2, 2, 0, False, device)


class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, device):
    super(ResBlock, self).__init__()

    self.res = torch.nn.Sequential(
        conv2D(in_channels, out_channels, device=device),
        nn.ReLU(inplace=True),
        conv2D(in_channels, out_channels, device=device)
    )

  def forward(self, x):
    return x + self.res(x)


class UNetRes(nn.Module):
  def __init__(self, device):
    super(UNetRes, self).__init__()

    in_nc, out_nc = 4, 3
    nc = [64, 128, 256, 512]

    self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, 1, 1, False, 'zeros', device)

    self.m_down1 = torch.nn.Sequential(ResBlock(nc[0], nc[0], device), ResBlock(nc[0], nc[0], device), ResBlock(nc[0], nc[0], device), ResBlock(nc[0], nc[0], device), downsample_strideconv(nc[0], nc[1], device))
    self.m_down2 = torch.nn.Sequential(ResBlock(nc[1], nc[1], device), ResBlock(nc[1], nc[1], device), ResBlock(nc[1], nc[1], device), ResBlock(nc[1], nc[1], device), downsample_strideconv(nc[1], nc[2], device))
    self.m_down3 = torch.nn.Sequential(ResBlock(nc[2], nc[2], device), ResBlock(nc[2], nc[2], device), ResBlock(nc[2], nc[2], device), ResBlock(nc[2], nc[2], device), downsample_strideconv(nc[2], nc[3], device))

    self.m_body = torch.nn.Sequential(ResBlock(nc[3], nc[3], device), ResBlock(nc[3], nc[3], device), ResBlock(nc[3], nc[3], device), ResBlock(nc[3], nc[3], device))

    self.m_up3 = torch.nn.Sequential(upsample_convtranspose(nc[3], nc[2], device), ResBlock(nc[2], nc[2], device), ResBlock(nc[2], nc[2], device), ResBlock(nc[2], nc[2], device), ResBlock(nc[2], nc[2], device))
    self.m_up2 = torch.nn.Sequential(upsample_convtranspose(nc[2], nc[1], device), ResBlock(nc[1], nc[1], device), ResBlock(nc[1], nc[1], device), ResBlock(nc[1], nc[1], device), ResBlock(nc[1], nc[1], device))
    self.m_up1 = torch.nn.Sequential(upsample_convtranspose(nc[1], nc[0], device), ResBlock(nc[0], nc[0], device), ResBlock(nc[0], nc[0], device), ResBlock(nc[0], nc[0], device), ResBlock(nc[0], nc[0], device))

    self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, 1, 1, False, 'zeros', device)

  def forward(self, x0):
    x1 = self.m_head(x0)
    x2 = self.m_down1(x1)
    x3 = self.m_down2(x2)
    x4 = self.m_down3(x3)
    x = self.m_body(x4)
    x = self.m_up3(x + x4)
    x = self.m_up2(x + x3)
    x = self.m_up1(x + x2)
    x = self.m_tail(x + x1)

    return x


def DPIR(clip: vs.VideoNode, strength: float = None, task: str = 'denoise', device_type: str = 'cuda', device_index: int = 0, contra_sharpening: bool = False, no_cache: Optional[bool] = None) -> vs.VideoNode:
  '''
  DPIR: Deep Plug-and-Play Image Restoration

  Parameters:
      clip: Clip to process. Only planar format with float sample type of 32 bit depth is supported.

      strength: Strength for deblocking or denoising. Must be greater than 0. Defaults to 50.0 for 'deblock' task, 5.0 for 'denoise' task.

      task: Task to perform. Must be 'deblock' or 'denoise'.

      device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

      device_index: Device ordinal for the device type.

      contra_sharpening: Whether to contra sharpen or not. Not recommended at high strength.
  '''
  if not isinstance(clip, vs.VideoNode):
    raise vs.Error('DPIR: This is not a clip')

  if clip.format.id != vs.RGBS:  # noqa
    raise vs.Error('DPIR: Only RGBS format is supported')

  if strength is not None and strength <= 0:
    raise vs.Error('DPIR: strength must be greater than 0')

  task = task.lower()
  device_type = device_type.lower()

  if task not in ['deblock', 'denoise']:
    raise vs.Error("DPIR: task must be 'deblock' or 'denoise'")

  if device_type not in ['cuda', 'cpu']:
    raise vs.Error("DPIR: device_type must be 'cuda' or 'cpu'")

  if device_type == 'cuda' and not torch.cuda.is_available():
    raise vs.Error('DPIR: CUDA is not available')

  if no_cache is None:
    no_cache = device_type == 'cuda'

  if device_type == 'cpu' and no_cache:
    raise vs.Error('DPIR: You can disable cache only with cuda')

  device = torch.device(device_type, device_index)

  if device_type == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True  # noqa
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

  if task == 'deblock':
    if strength is None:
      strength = 50.0
    strength /= 100
    model_name = 'drunet_deblocking_color.pth'
  else:
    if strength is None:
      strength = 5.0
    strength /= 255
    model_name = 'drunet_color.pth'

  noise_level_map = torch.empty((1, 1, clip.height, clip.width)).fill_(strength)

  model_path = os.path.join(os.path.dirname(__file__), model_name)

  weights = torch.load(model_path)

  model = UNetRes(device)
  model.load_state_dict(weights, strict=True)
  model.eval()

  for _, v in model.named_parameters():
    v.requires_grad = False

  for param in model.parameters():
    param.grad = None

  model = model.to(device, None, True)
  noise_level_map = noise_level_map.to(device, None, True)

  img_L = torch.empty((3, clip.height, clip.width), device=device)

  __process = model.forward if clip.height % 8 == 0 and clip.width % 8 == 0 else partial(refield_tensor, model=model)

  def _process(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
    with torch.no_grad():
      frame_to_tensor(f, device, img_L, noise_level_map)

      if no_cache:
        torch.cuda.empty_cache()

      # Uncomment to trace and save the model (just run one frame)
      # traced = torch.jit.trace(model, img_L)

      # traced.save(task + '_jit_' + device + '.pt')

      return tensor_to_frame(__process(img_L), f)

  processed = clip.std.ModifyFrame(clips=clip, selector=_process)

  if not contra_sharpening:
    return processed

  sharp = core.std.Expr([
      processed,
      processed.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]),
      processed.std.Median()
  ], 'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?')

  blur = sharp.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

  diff = core.std.Expr([
      core.std.MakeDiff(clip, processed).std.Convolution(matrix=[4, 4, 4, 2, 1, 2, 4, 4, 4]),
      core.std.MakeDiff(sharp, blur).std.Convolution(matrix=[1, 1, 1, 1, 4, 1, 1, 1, 1])
  ], expr='x y min')

  return core.std.MaskedMerge(
      core.std.MergeDiff(processed, diff),
      processed,
      core.std.BlankClip(
          processed, processed.width - 2,
          processed.height - 2,
          vs.GRAYS, 1, color=0
      ).std.AddBorders(1, 1, 1, 1, 1)
  )


def frame_to_tensor(f: vs.VideoFrame, device, img_L, noise_level_map) -> None:
  torch.cat((
      torch.tensor(
          np.asfarray([np.asfarray(f.get_read_array(i)) for i in np.arange(3)]),
          device=device, requires_grad=False
      ).unsqueeze(0),
      noise_level_map
  ), dim=1, out=img_L)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
  arr = t.data.squeeze().cpu().numpy()

  fout = f.copy()
  [np.copyto(np.asarray(fout.get_write_array(i)), arr[i, ...]) for i in np.arange(3)]
  return fout
