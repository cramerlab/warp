# Please don't tweet just yet!

# Using Warp

If you just want to use Warp, all tutorials and binaries can be found at http://www.warpem.com. Continue reading this README only if you're interested in compiling Warp from source.

# Compiling Warp

## Prerequisites

[CUDA SDK 9.0](https://developer.nvidia.com/cuda-downloads) with $PATH and $CUDA_PATH_V9_0 environment variables set correctly  
[cuDNN 7.0](https://developer.nvidia.com/cudnn)  
[Visual Studio 2017](https://www.visualstudio.com/) with the VC++ 2015 compiler and Windows SDK 10.0 included during installation  

[TensorFlow 1.5](https://github.com/tensorflow/tensorflow) source code  

[GTOM](https://github.com/dtegunov/gtom)  
[liblion](https://github.com/dtegunov/liblion)  

## Preparations

Make sure the Warp, GTOM and liblion folders are located in the same parent folder, or modify the include paths accordingly.  

Compile GTOM and liblion using the same configuration (Debug/Release) you intend to use for Warp's compilation.  

Compile TensorFlow 1.5. This is a tricky thing to do on Windows, and the exact steps depend highly on your environment. However, some googling has always solved this problem so far. Once compiled, create an environment variable $TENSORFLOW_LIBS that points to TF's 'build' folder. This will be required to configure all paths correctly in Warp's TFUtility project.  

## Compilation

Open Warp.sln in Visual Studio. If everything is configured correctly, you shouldn't see any error messages. Now build the 'Warp' project, which will also build all of its dependencies. That's it!  

## Other operating systems

Warp's GUI definitely won't build on Linux or OS X as there are no ongoing efforts to port WPF to these platforms. There is a good chance you will manage to build WarpLib, or any other WPF-free parts of the solution. However, we haven't tried it. We definitely intend to make a head-less, cross-platform version for processing in HPC clusters, but that is still a long way down the road.  

## Authorship

Warp is being developed by Dimitry Tegunov ([tegunov@gmail.com](mailto:tegunov@gmail.com)) in Patrick Cramer's lab at the Max Planck Institute for Biophysical Chemistry in Göttingen, Germany.
