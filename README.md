# CoaT: Co-Scale Conv-Attentional Image Transformers
This repo contains PyTorch implementation of paper [Co-Scale Conv-Attentional Image Transformers](https://arxiv.org/abs/2104.06399)
and this is not the official implementation. For official implementation please visit [here](https://github.com/mlpc-ucsd/CoaT).

![](model.PNG)


## Usage:
```python
import numpy as np
from coat import CoaT
import torch

img = torch.ones([1, 3, 224, 224])

model = CoaT(3, 224, 1000)
out = model(img)

print("Shape of out :", out.shape)  # [B, num_classes]

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)
```
## Citation:
```
@misc{xu2021coscale,
      title={Co-Scale Conv-Attentional Image Transformers}, 
      author={Weijian Xu and Yifan Xu and Tyler Chang and Zhuowen Tu},
      year={2021},
      eprint={2104.06399},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
