# VapourSynth-DPIR

Ported from https://github.com/cszn/DPIR, https://github.com/HolyWu/vs-dpir

## Still in beta.

If you only work on CPU, you can set parallel and have a speed improvement at the cost of maxed RAM (probably)
If you're on GPU you will see the same speed.

# Dependencies

- Torch (Tested with 1.9)
  - Cuda 10 (Tested on Cuda 11.1)
- VapourSynth (duh)

VapourSynth DPIR Implementation

# Usage

```python
  core.DPIR.Deblock(clip clip, [float strength, int gpu, int parallel, int no_cache])

  core.DPIR.Denoise(clip clip, [float strength, int gpu, int parallel, int no_cache])
```

- clip: Clip to process. Only RGBS format is supported.
- strength: Strenght of the function.
- gpu:
  - -1: Auto
  - 0: Cpu
  - 1: Cuda device
  - \>1: Intended for multi GPU, will select device with index `gpu - 1`
- parallel: if to do parallel processing or not. 0 is recommended if you have a small amount of VRAM.
- no_cache: Disable cuda cache.

# Compilation

I prefer doing it in VSCode, but

```bash
cmake --build ./build --config Release --target vsdpir -j 26 --
```

Keep in mind you'll have to set VS_LIB_PATH, TORCH_LIB_PATH, BOOST_LIB_PATH manually.

# "Todo List"

Implement a semaphore with cuda api to limit concurrent threads so it won't try to allocate absurd amounts of VRAM.

Add Deblur/Demosaick/SuperResolution

If possible, OpenCL or ROCm optimization 