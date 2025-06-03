# Steps to reproduce error

root@3297547ee6e0:/home/tensorrt-test# python3 create_trt.py

  ONNX file created and saved: ./pad_layer.onnx

  Converting Model pad_layer.onnx to TensorRT - Version: 10.0.1

  [06/03/2025-20:36:52] [TRT] [I] [MemUsageChange] Init CUDA: CPU +17, GPU +0, now: CPU 27, GPU 225 (MiB)

  [06/03/2025-20:36:54] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1762, GPU +312, now: CPU 1925, GPU 537 (MiB)

  parsing model

  [06/03/2025-20:36:54] [TRT] [W] ModelImporter.cpp:420: Make sure input pad_sizes has Int64 binding.

  model parsed

  Network Description

  Input 'input_image' with shape (1, -1, -1, -1) and dtype DataType.FLOAT

  Input 'pad_sizes' with shape (8,) and dtype DataType.INT64

  Output 'paded_image' with shape (-1, -1, -1, -1) and dtype DataType.FLOAT

  Profile for Input 'input_image' shape (1, -1, -1, -1)

  Profile for Input 'pad_sizes' shape (8,)

  Serializing engine to file: /home/tensorrt-test/pad_layer.trt

  [06/03/2025-20:36:54] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.

  [06/03/2025-20:36:54] [TRT] [I] Detected 2 inputs and 1 output network tensors.

  [06/03/2025-20:36:54] [TRT] [I] Total Host Persistent Memory: 0

  [06/03/2025-20:36:54] [TRT] [I] Total Device Persistent Memory: 0

  [06/03/2025-20:36:54] [TRT] [I] Total Scratch Memory: 0

  [06/03/2025-20:36:54] [TRT] [I] Total Activation Memory: 0

  [06/03/2025-20:36:54] [TRT] [I] Total Weights Memory: 0

  [06/03/2025-20:36:54] [TRT] [I] Engine generation completed in 0.0057974 seconds.

  [06/03/2025-20:36:54] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 610 MiB

  [06/03/2025-20:36:54] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 3107 MiB


root@3297547ee6e0:/home/tensorrt-test# ./build/tensorrt_test pad_layer.trt input_image.jpg

  Load modelEngine file path: pad_layer.trt

  Loaded engine size: 0 MiB

  [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)

  Number of IO Tensors: 3

  Tensor name: input_image

  Tensor Dims: 1 -1 -1 -1

  Tensor name: pad_sizes

  Tensor Dims: 8

  Tensor name: paded_image

  Tensor Dims: 1 -1 -1 -1

  Segmentation fault (core dumped)
  
