import sys

import torch

if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA is not available")
        print("Ensure you have both a compatible Nvidia GPU and CUDA installed")
        sys.exit(0)

    print(f"Number of cuda devices available: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
