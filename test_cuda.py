import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("torch.version.cuda:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())


# from transformers import pipeline
# pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar", device=0)  # 0 = أول GPU, -1 = CPU

# import torch, time
# x = torch.randn(4096,4096, device="cuda")
# t0 = time.time(); y = x @ x; torch.cuda.synchronize(); print("GEMM ms:", (time.time()-t0)*1e3)


import torch, transformers
print(torch.__version__, torch.version.cuda, transformers.__version__)
