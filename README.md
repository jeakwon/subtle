# SUBTLE
SUBTLE (Spectrogram-UMAP-based Temporal Link Embedding) is a animal behavior mapper.

```
pip install git+https://github.com/jeakwon/subtle.git
```

## Reading example AVATAR data
```python
import subtle
import numpy as np
import pandas as pd

Xs = [
    pd.read_csv('https://raw.githubusercontent.com/jeakwon/avatarpy/main/avatarpy/data/freely_moving.csv', header=None).values,
    pd.read_csv('https://raw.githubusercontent.com/jeakwon/avatarpy/main/avatarpy/data/freely_moving2.csv', header=None).values
]

mapper = subtle.Mapper(20)
data = mapper.train(Xs)
```