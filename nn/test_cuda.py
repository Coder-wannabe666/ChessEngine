import torch

# Check if PyTorch can see the GPU
is_cuda = torch.cuda.is_available()
print(f"CUDA available: {is_cuda}")

if is_cuda:
    # Print the exact name of the graphics card
    print(f"GPU found: {torch.cuda.get_device_name(0)}")
else:
    # Warning message if it fails
    print("CUDA is still not working. Double check your NVIDIA drivers.")