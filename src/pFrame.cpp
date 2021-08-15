#pragma warning(disable : 4624)
#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/script.h>

#include "UNetRes.cpp"
#include "commons.cpp"
#include "vsdpir.h"

static const VSFrameRef* VS_CC processFrame(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
  FrameData* data = static_cast<FrameData*>(*instanceData);
  torch::NoGradGuard no_grad;

  const VSFrameRef* src = vsapi->getFrameFilter(n, data->node, frameCtx);

  int width = vsapi->getFrameWidth(src, 0);
  int height = vsapi->getFrameHeight(src, 0);

  VSFrameRef* dest = vsapi->newVideoFrame(data->vinfo.format, width, height, src, core);

  auto* src_r_plane = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 0));
  auto* src_g_plane = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 1));
  auto* src_b_plane = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 2));

  auto* dest_r_plane = reinterpret_cast<float*>(vsapi->getWritePtr(dest, 0));
  auto* dest_g_plane = reinterpret_cast<float*>(vsapi->getWritePtr(dest, 1));
  auto* dest_b_plane = reinterpret_cast<float*>(vsapi->getWritePtr(dest, 2));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int k = width * i + j;

      data->frameArray[k] = src_r_plane[k];
      data->frameArray[(width * height) + k] = src_g_plane[k];
      data->frameArray[(width * height * 2) + k] = src_b_plane[k];
    }
  }

  auto value = torch::cat({torch::from_blob(data->frameArray.data(), {1, 3, height, width}), data->weights}, 1).to(*data->device);

  value = data->model->forward(value);

  value = value.squeeze().contiguous().cpu();

  std::vector<float> output(value.data_ptr<float>(), value.data_ptr<float>() + value.numel());

  auto out_start = output.begin();
  auto out_mid_0 = out_start + (width * height);
  auto out_mid_1 = out_start + (width * height * 2);

  std::copy(out_start, out_mid_0, dest_r_plane);
  std::copy(out_mid_0, out_mid_1, dest_g_plane);
  std::copy(out_mid_1, output.end(), dest_b_plane);

  if (data->noCache) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  vsapi->freeFrame(src);

  return dest;
}