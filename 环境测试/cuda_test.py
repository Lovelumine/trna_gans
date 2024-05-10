import torch

def check_pytorch_and_cuda():
    # 检查PyTorch是否安装成功
    print(f"PyTorch version: {torch.__version__}")

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("CUDA is available. Here are the CUDA devices:")
        # 打印可用的CUDA设备数量和每个设备的详细信息
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)} (Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9} GB)")
    else:
        print("CUDA is not available. Running on CPU.")

if __name__ == "__main__":
    check_pytorch_and_cuda()
