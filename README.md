# jetsonnano-experiments

## Introducción

Info general sobre uso de Jetson Nano y experimentos realizados.

## Comandos útiles en monitoreo de trabajo en CPU & GPU

top

jtop <https://github.com/rbonghi/jetson_stats>

ps -aux

tegrastats

Para ejecutar en background:
tegrastats --interval <int> --logfile <out_file> &

## Cambios en modo de operacion (TX2)

Ver este vinculo:

<https://developer.ridgerun.com/wiki/index.php?title=NVIDIA_Jetson_TX2_NVP_model>

Comando en consola:

```console
$ sudo nvpmodel -m [modo]
...
```

### Links de utilidad

#### General

get-started-jetson-nano-devkit:
<https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro>

jetson-nano:
<https://developer.nvidia.com/embedded/jetson-nano-developer-kit>

jetson-tx2:
<https://developer.nvidia.com/embedded/jetson-tx2-developer-kit>

software:
<https://developer.nvidia.com/embedded/develop/software>

hardware:
<https://developer.nvidia.com/embedded/jetson-modules>

vnc-setup:
<https://developer.nvidia.com/embedded/learn/tutorials/vnc-setup>

instalar jetpack:
<https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html>

instalacion pytorch:
<https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048>

instalacion tensorflow:
<https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html> --> note: using version 2.5.0

instalacion tensorrt:
<https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#installing-pip>

insalacion pycuda:
<https://medium.com/dropout-analytics/pycuda-on-jetson-nano-7990decab299>

uso nvcc:
<https://medium.com/dropout-analytics/pycuda-on-jetson-nano-7990decab299>

utilizacion camara web:
<https://developer.nvidia.com/embedded/learn/tutorials/first-picture-csi-usb-camera>

#### Sobre la CPU

<https://developer.arm.com/documentation/ddi0488/h/programmers-model/armv8-a-architecture-concepts?lang=en>

#### Memoria

<https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management>

<https://developer.ridgerun.com/wiki/index.php?title=NVIDIA_CUDA_Memory_Management>

##### Agregar SWAP

Ver estado actual

```console
$ free -m
Mem:  ...
Swap: ...
```

Crear archivo swap

```console
$ sudo systemctl disable nvzramconfig.service
$ sudo fallocate -l 4GB /mnt/4GB.swap
$ sudo chmod 600 /mnt/4GB.swap
$ sudo mkswap /mnt/4GB.swap
Setting up ...
```

Editar archivo

```console
$ sudo vi /etc/fstab
# /etc/fstab
...
/dev/root...
/mnt/4GB.swap swap swap defaults 0 0  # This line must be added
```

Reiniciar y revisar.

```console
$ sudo reboot
...
$ free -m
Mem:  ...
Swap: ...
```
