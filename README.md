## Torch-gists

A collection of models and utilities that I frequently use in my ML projects

#### Importing models

Models included:
- ResNet (18, 34, 101, 152)
- VGG
- MobileNet-V2

To import models:
```python
from torch_gists.models import ResNet18
model = ResNet18(num_classes=10)
```

#### Utilities

##### wrapper for pytorch_lightning

A base class `pl_wrapper` that extens pytorch_lightning implementing training, validation, testing loops,
dataloader that's common to most models.
Users can extend the `pl_wrapper` class and implement their own `__ini__`, `forward` and data methods.

```python
from torch_gists.utils import pl_wrapper

class my_model(pl_wrapper):
    def __init__(self):
        super().__init__()
        ...
    
    def forward(self, x):
        ...


```