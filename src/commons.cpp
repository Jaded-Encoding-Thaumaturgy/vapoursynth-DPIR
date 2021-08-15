#pragma warning(disable : 4624)
#pragma once

#include "UNetRes.cpp"
#include "vsdpir.h"

enum FilterType {
  DEBLOCK = 0,
  DENOISE = 1,
};

struct FrameData {
  bool noCache;
  VSNodeRef *node;
  VSVideoInfo vinfo;
  torch::Device *device;
  torch::Tensor weights;
  std::vector<float> frameArray;
  std::shared_ptr<UNetRes> model;
};

static bool checkClipFormat(FrameData data) {
  return isConstantFormat(&data.vinfo) && data.vinfo.format->colorFamily == cmRGB && data.vinfo.format->sampleType == stFloat && data.vinfo.format->bitsPerSample == 32;
}
