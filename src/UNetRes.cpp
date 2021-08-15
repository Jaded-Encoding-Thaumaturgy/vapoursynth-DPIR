#pragma warning(disable : 4624)
#pragma once

#include "vsdpir.h"

static torch::nn::ConvTranspose2d upsample_convtranspose(int in_channels, int out_channels) {
  return torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 2).padding(0).stride(2).bias(false));
}

static torch::nn::Conv2d downsample_strideconv(int in_channels, int out_channels) {
  return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 2).stride(2).padding(0).bias(false));
}

struct ResBlock : torch::nn::Module {
  ResBlock(int channels, torch::Device *device) {
    res = register_module("res", torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).stride(1).padding(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions(true)),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).stride(1).padding(1).bias(false))
    ));

    res->to(*device);
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::NoGradGuard no_grad;
    
    return x + res->forward(x);
  }

  torch::nn::Sequential res = {nullptr};
};

struct UNetRes : torch::nn::Module {
  UNetRes(torch::Device *device) {
    int NC[] = {64, 128, 256, 512};

    m_head = register_module("m_head", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, NC[0], 3).stride(1).padding(1).bias(false)));

    m_down1 = register_module("m_down1", torch::nn::Sequential(ResBlock(NC[0], device), ResBlock(NC[0], device), ResBlock(NC[0], device), ResBlock(NC[0], device), downsample_strideconv(NC[0], NC[1])));
    m_down2 = register_module("m_down2", torch::nn::Sequential(ResBlock(NC[1], device), ResBlock(NC[1], device), ResBlock(NC[1], device), ResBlock(NC[1], device), downsample_strideconv(NC[1], NC[2])));
    m_down3 = register_module("m_down3", torch::nn::Sequential(ResBlock(NC[2], device), ResBlock(NC[2], device), ResBlock(NC[2], device), ResBlock(NC[2], device), downsample_strideconv(NC[2], NC[3])));

    m_body = register_module("m_body", torch::nn::Sequential(ResBlock(NC[3], device), ResBlock(NC[3], device), ResBlock(NC[3], device), ResBlock(NC[3], device)));

    m_up3 = register_module("m_up3", torch::nn::Sequential(upsample_convtranspose(NC[3], NC[2]), ResBlock(NC[2], device), ResBlock(NC[2], device), ResBlock(NC[2], device), ResBlock(NC[2], device)));
    m_up2 = register_module("m_up2", torch::nn::Sequential(upsample_convtranspose(NC[2], NC[1]), ResBlock(NC[1], device), ResBlock(NC[1], device), ResBlock(NC[1], device), ResBlock(NC[1], device)));
    m_up1 = register_module("m_up1", torch::nn::Sequential(upsample_convtranspose(NC[1], NC[0]), ResBlock(NC[0], device), ResBlock(NC[0], device), ResBlock(NC[0], device), ResBlock(NC[0], device)));

    m_tail = register_module("m_tail", torch::nn::Conv2d(torch::nn::Conv2dOptions(NC[0], 3, 3).stride(1).padding(1).bias(false)));

    m_head->to(*device);
    m_down1->to(*device);
    m_down2->to(*device);
    m_down3->to(*device);
    m_body->to(*device);
    m_up3->to(*device);
    m_up2->to(*device);
    m_up1->to(*device);
    m_tail->to(*device);
  }

  torch::Tensor forward(torch::Tensor x0) {
    torch::NoGradGuard no_grad;
    torch::Tensor x, x1, x2, x3, x4;
    x1 = m_head->forward(x0);

    x2 = m_down1->forward(x1);
    x3 = m_down2->forward(x2);
    x4 = m_down3->forward(x3);

    x = m_body->forward(x4);

    x = m_up3->forward(x + x4);
    x = m_up2->forward(x + x3);
    x = m_up1->forward(x + x2);

    return m_tail->forward(x + x1);
  }

  torch::nn::Conv2d m_head{nullptr}, m_tail{nullptr};
  torch::nn::Sequential m_body{nullptr};
  torch::nn::Sequential m_down1{nullptr}, m_down2{nullptr}, m_down3{nullptr};
  torch::nn::Sequential m_up1{nullptr}, m_up2{nullptr}, m_up3{nullptr};
};

// def test_onesplit(model, L, refield=64, sf=1):
//   h, w = L.size()[-2:]

//   top = slice(0, (h // 2 // refield + 1) * refield)
//   bottom = slice(h - (h // 2 // refield + 1) * refield, h)
//   left = slice(0, (w // 2 // refield + 1) * refield)
//   right = slice(w - (w // 2 // refield + 1) * refield, w)
//   Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
//   Es = [model(Ls[i]) for i in range(4)]
//   b, c = Es[0].size()[:2]
//   E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
//   E[..., :h // 2 * sf, :w // 2 * sf] = Es[0][..., :h // 2 * sf, :w // 2 * sf]
//   E[..., :h // 2 * sf, w // 2 * sf:w * sf] = Es[1][..., :h // 2 * sf, (-w + w // 2) * sf:]
//   E[..., h // 2 * sf:h * sf, :w // 2 * sf] = Es[2][..., (-h + h // 2) * sf:, :w // 2 * sf]
//   E[..., h // 2 * sf:h * sf, w // 2 * sf:w * sf] = Es[3][..., (-h + h // 2) * sf:, (-w + w // 2) * sf:]
//   return E
