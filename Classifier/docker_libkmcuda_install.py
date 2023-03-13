from torch.cuda import is_available, device_count
import os, sys

if is_available():
    if device_count() > 0:
        try:
            # not sure if this will actually install kmcuda correctly
            os.system('cd /kmcuda/src')
            os.system('cmake -D DISABLE_R=y -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCMAKE_BUILD_TYPE=Release . && make')
            os.system('CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda CUDA_ARCH=64 pip install libKMCUDA')
            # todo; might want to remove CUDA_ARCH or add ability to change to the device's CUDA_ARCH automatically
            sys.stdout.write("[KMcuda] Installation succeeded!")
        except:
            sys.stdout.write("[KMcuda] Failed to install libkmcuda, udpate docker_libkmcuda_install.py")
    else:
        sys.stdout.write("[KMcuda] Skipping installation because no CUDA-enabled gpu devices are available.")
else:
    sys.stdout.write("[KMcuda] Skipping installation because not torch.cuda.is_available().")
