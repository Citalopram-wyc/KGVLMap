# Third-party dependencies and checkpoints

This repository imports third-party packages rather than redistributing their source trees or pretrained weights. Those materials retain their respective upstream licenses and terms.

- PyTorch and torchvision: https://github.com/pytorch/pytorch and https://github.com/pytorch/vision
- timm / PyTorch Image Models: https://github.com/huggingface/pytorch-image-models
- OpenCLIP: https://github.com/mlfoundations/open_clip
- RemoteCLIP: https://github.com/ChenDelong1999/RemoteCLIP
- RemoteCLIP checkpoint source: https://huggingface.co/chendelong/RemoteCLIP
- Swin-B checkpoint source: https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k

Other runtime dependencies are declared in `requirements.txt`. The local weight downloader records source URLs and SHA-256 hashes for downloaded files. Dataset usage and redistribution rights are separate from the source-code license.
