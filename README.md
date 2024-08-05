# Fusing Depthwise and Pointwise Convolutions for Efficient Inference on GPUs

This repository contains an implementation of the paper [Fusing Depthwise and Pointwise Convolutions for Efficient Inference on GPUs](https://arxiv.org/pdf/2404.19331)

If you find this repository useful for your research, please use the following bibtex to cite us,

```
@article{qararyah2024fusing,
  title={Fusing Depthwise and Pointwise Convolutions for Efficient Inference on GPUs},
  author={Qararyah, Fareed and Azhar, Muhammad Waqar and Maleki, Mohammad Ali and Trancoso, Pedro},
  journal={arXiv preprint arXiv:2404.19331},
  year={2024}
}
```

## The core idea

Depthwise and pointwise convolutions have fewer parameters and perform fewer operations than standard convolutions. As a result, they have become increasingly used in various compact DNNs, including convolutional neural networks (CNNs) and vision transformers (ViTs). However, they have a lower compute-to-memory-access ratio than standard convolutions, making their memory accesses often the performance bottleneck.\\
Fusing depthwise and pointwise convolutions helps in overcoming their memory access bottleneck. This repository contains fused implementations of depthwise and pointwise convolutions on GPUs. We refer to fused pairs of pointwise and depthwise convolutiona as _Fused Convolutional Modules (FCMs)_. FCMs significantly reduce pointwise and depthwise convolutions memory accesses, improving execution time and energy efficiency. 

Nevertheless, fusion does not guarantee performance improvement in all cases. For more details please refer to the paper ([Fusing Depthwise and Pointwise Convolutions for Efficient Inference on GPUs](https://arxiv.org/pdf/2404.19331)). There is a need fo cost models to evaluate the trade-offs associated with fusion and determine which convolutions are beneficial to fuse and the optimal FCM parameters. _FusePlanner_ module in the repository contains implementations of such cost models. FusePlanner consists of cost models to estimate the memory accesses of depthwise, pointwise, and FCM kernels given GPU characteristics and suggest tiling parameters that are estimated to mainimize the global memory access.

## How to use:

### 1- Using the kernels in a custom implementation
The repository contains depthwise and pointwise convolution kernel implementations and kernels of fused depthwise and pointwise convolutions.
These kernels adopt a direct convolution implementation and use variants of Output Stationary (OS) dataflow. There are implementations using
both fp32 and int8. They are under the directory _fcm\_and\_lbl\_kernels_. _CNN\_segment.cpp_ shows an example of using the kernels to implement the sequence of convolutions constituting the body of CNNs.

### 2- Using _FusePlanner_

FusePlanner takes as inputs:
1) GPU specifications. An example: _./FusePlanner/hw\_configs/gtx1660.cfg_
2) DAG representing a model or set of layers, their weight and FM specifications, and the layers connectivity. An example: _/extract\_tflite\_model\_metadata/models\_archs/models/mob\_v1/model\_dag.json_

FusePlanner outputs:
1) which layers are to be fused and which are not
2) which FCMs to use
3) the tiling that minimizes the global memory access in each case.

Running FusePlanner: _./FusePlanner/main.py_

_./extract\_tflite\_model\_metadata/extract\_weights\_and\_fms.py_ script generates a FusePlanner DAG representation starting from Keras model.

## Acknowledgements

This work was supported by VEDLIoT project, which received funding from the European Union's Horizon 2020 research and innovation program under grant agreement No 957197. This work was also partly supported by the Swedish Foundation for Strategic Research (contract number CHI19-0048) under the PRIDE project and the European High Performance Computing Joint Undertaking (JU) under Framework Partnership Agreement No 800928 and Specific Grant Agreement No 101036168 (EPI SGA2).
The JU receives support from the European Union’s Horizon 2020 research and innovation programme and from Croatia, France, Germany, Greece, Italy, Netherlands, Portugal, Spain, Sweden, and Switzerland.