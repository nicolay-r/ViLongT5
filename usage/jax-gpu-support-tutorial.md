## JAXlibrary -- setup GPU Support Tutorial

`JAX`, rougly speaking, is a library, proposed and maintained by Google which combines 
[autograd](https://github.com/hips/autograd)
and [XLA](https://www.tensorflow.org/xla) 
compiler in order to bring neural networks  onto the computational devices in a most efficient way.
However, speaking about `GPU` and more nn oriented devices as `TPU`, it is pretty important to perform a 
proper library setup in order to make them available to use.
`JAX` is widely used for transformers, where [T5](https://github.com/google-research/t5x) founds its implementation, 
including other variations formed into another [flaxformer](https://github.com/google/flaxformer) project.

In this post we address on the issue you may encountered with once decided to apply `Jax` library for `GPU` calculations.
It is required to make library frienly and familiar with such toolkits: NVidia CUDA compiler (`ncdu`), CUDA DNN (`cudnn`).
Most of the steps were taken from the [JAX installation with Nvidia CUDA and cudNN support](https://www.youtube.com/watch?v=auksaSl8jlM) 
video by [Avkash Chauhan](https://twitter.com/prodramp).

Let's get started!

The problem that you may enconter first is that you got installed the ordinary version of the related library dubbed as `jaxlib`.
For example and in case of training T5 model, you may see the following logs:

```shell
[xla_bridge.py:356] Unable to initialize backend 'tpu_driver': NOT_FOUND: Unable to find driver in registry given worker:
[xla_bridge.py:356] Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attirbute 'GpuAllocatorConfig'
[xla_bridge.py:356] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attirbute 'GpuAllocatorConfig'
...
[xla_bdridge.py:363] No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
```

At first, let's say we have installed:
```shell
jax==0.3.21
jaxlib==0.3.15
```
Here is list of actions required to be performed in order to make it familiar with `GPU` devices:

1. Uninstall `jax` and `jaxlib` in order to perform its clean installation later:
```shell
pip uninstall jax jaxlib
```
2. `cudnn` version checkout:
```shell
cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN
```
3. Clean install from [list of laxlib-cudnn-cuda](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).
At this point is pretty important to checkout the version of `cuda` and `cudnn` pre-installed and to be used by `jaxlib`;
According to the list of the available `pip` packages we were down for `jaxlib==0.3.15+cuda11.cudnn82` and therefore 
perform the installation as follows:
```shell
pip install --upgrade jax==0.3.15 jaxlib==0.3.15+cuda11.cudnn82 \
  -f https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.15+cuda11.cudnn82-cp39-none-manylinux2014_x86_64.whl
```
4. check out:
```shell
>>> import jax
>>> jax.devices()
[GpuDevice(id=0, process_index=0), ... GpuDevice(id=3, process_index=0)]
```

## Reference

[JAX installation with Nvidia CUDA and cudNN support](https://www.youtube.com/watch?v=auksaSl8jlM) by [Avkash Chauhan](https://twitter.com/prodramp)