#pragma warning(disable : 4624)
#pragma once

#include "vsdpir.h"

#include "UNetRes.cpp"
#include "pFrame.cpp"
#include "commons.cpp"

static void VS_CC filterInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
  vsapi->setVideoInfo(&static_cast<FrameData *>(*instanceData)->vinfo, 1, node);
}

static void VS_CC filterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
  FrameData *data = static_cast<FrameData *>(instanceData);
  vsapi->freeNode(data->node);

  delete data;
}

static const VSFrameRef *VS_CC filterGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
  FrameData *data = static_cast<FrameData *>(*instanceData);

  if (activationReason == VSActivationReason::arInitial) {
    vsapi->requestFrameFilter(n, data->node, frameCtx);
  }

  if (activationReason == VSActivationReason::arAllFramesReady) {
    return processFrame(n, activationReason, instanceData, frameData, frameCtx, core, vsapi);
  }

  return nullptr;
}

static void VS_CC filterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi, FilterType filterType, float defaultStrength) {
  int error;
  VSFilterMode filterMode;
  std::string name = "VSDPIR.";
  FrameData data{}, *out_data;
  torch::NoGradGuard no_grad;

  data.node = vsapi->propGetNode(in, "clip", 0, 0);
  data.vinfo = *vsapi->getVideoInfo(data.node);

  try {
    if (!checkClipFormat(data)) {
      throw std::string{"'clip' has to be constant RGB format and 32 bit float"};
    }

    int gpu = vsapi->propGetInt(in, "gpu", 0, &error);

    bool is_gpu = (error || gpu == -1) ? torch::cuda::is_available() : !!gpu;

    int device_index = (is_gpu && (gpu > 1)) ? gpu - 1 : 0;

    int isParallel = vsapi->propGetInt(in, "parallel", 0, &error);

    filterMode = (isParallel || error) ? VSFilterMode::fmParallel : VSFilterMode::fmUnordered;

    auto strength = vsapi->propGetFloat(in, "strength", 0, &error);

    if (error) {
      strength = defaultStrength;
    } else if (strength > 4096) {
      throw std::string{"'strength' has to be <= 2 ** 12"};
    }

    int noCache = vsapi->propGetInt(in, "no_cache", 0, &error);

    data.noCache = (noCache == 0 || error) && is_gpu;

    data.device = new torch::Device(is_gpu ? torch::kCUDA : torch::kCPU, device_index);

    data.model = std::make_shared<UNetRes>(data.device);

    std::string pluginFilePath = std::string{vsapi->getPluginPath(vsapi->getPluginById("dev.setsugen.vsdpir", core))};

    std::string model_path = pluginFilePath.substr(0, pluginFilePath.find_last_of('/')).append("/dpir_models/");

    if (filterType == FilterType::DENOISE) {
      model_path.append("drunet_denoise_jit.pt");
      name.append("Denoise");
      strength /= 255.0;
    } else {
      model_path.append("drunet_deblock_jit.pt");
      name.append("Deblock");
      strength /= 100.0;
    }

    try {
      torch::load(data.model, model_path);
    } catch (const c10::Error &e) {
      throw std::string{name + ": Error loading the model" + e.msg() + e.what()};
    }

    data.model->eval();

    data.model->to(*data.device);

    data.weights = torch::full({1, 1, data.vinfo.height, data.vinfo.width}, strength);

    data.weights.to(*data.device);

    data.frameArray = std::vector<float>(3 * data.vinfo.width * data.vinfo.height);

    out_data = new FrameData{data};

  } catch (const std::string &error) {
    vsapi->setError(out, ("vsdpir: " + error).c_str());
    vsapi->freeNode(data.node);
    return;
  }

  vsapi->createFilter(in, out, name.c_str(), filterInit, filterGetFrame, filterFree, filterMode, 0, out_data, core);
}

static void VS_CC filterDeblockCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
  return filterCreate(in, out, userData, core, vsapi, FilterType::DEBLOCK, 50.0);
}

static void VS_CC filterDenoiseCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
  return filterCreate(in, out, userData, core, vsapi, FilterType::DENOISE, 5.0);
}

VS_EXTERNAL_API(void)

VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
  configFunc("dev.setsugen.vsdpir", "vsdpir", "VapourSynth DPIR Implementation", VAPOURSYNTH_API_VERSION, 1, plugin);
  registerFunc("Deblock", "clip:clip;strength:float:opt;gpu:int:opt;parallel:int:opt;no_cache:int:opt;", filterDeblockCreate, 0, plugin);
  registerFunc("Denoise", "clip:clip;strength:float:opt;gpu:int:opt;parallel:int:opt;no_cache:int:opt;", filterDenoiseCreate, 0, plugin);
}