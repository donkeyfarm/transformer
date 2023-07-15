# transformer
study the features of transformer

# development environment
- macbook air whith m1(mps)
- anaconda: 
    `curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh`
    `sh Miniconda3-latest-MacOSX-arm64.sh`
- python3.9
    `conda install pytorch torchvision torchaudio -c pytorch-nightly`
    ```
    import torch
        if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")
    ```

# refs
1. https://developer.apple.com/metal/pytorch/
2. https://towardsdatascience.com/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1
3. https://github.com/jamescalam/pytorch-mps