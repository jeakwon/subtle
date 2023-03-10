# SUBTLE
SUBTLE (Spectrogram-UMAP-based Temporal Link Embedding) is a animal behavior mapper.

```
pip install -U git+https://github.com/jeakwon/subtle.git
```
or
```
pip install -U https://github.com/jeakwon/subtle/archive/refs/heads/main.zip
```

## Prepare dataset
```python
import subtle
import pandas as pd

# Dataset for training (5 young 5 adult mice)
y5a5 = [
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/adult_6112.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/adult_6115.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/adult_6116.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/adult_6127.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/adult_7678.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/young_7100.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/young_7678.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/young_8294.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/young_8296.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y5a5/coords/young_8301.csv',
]

# Dataset for mapping (3 young 6 adult mice)
y3a6 = [
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/adult_8294.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/adult_8296.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/adult_8301.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/adult_8765.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/adult_8767.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/adult_8789.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/young_8765.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/young_8767.csv',
    'https://raw.githubusercontent.com/jeakwon/subtle/main/dataset/y3a6/coords/young_8789.csv',
]

dataset1 = []
for csv in y5a5:
    X = pd.read_csv(csv, header=None).values
    X = subtle.avatar_preprocess(X) # subtract coordinates with global mean of (x, y, z)
    dataset1.append(X)

dataset2 = []
for csv in y3a6:
    X = pd.read_csv(csv, header=None).values
    X = subtle.avatar_preprocess(X) # subtract coordinates with global mean of (x, y, z)
    dataset2.append(X)
```

## Training and Mapping
```python
mapper = subtle.Mapper(fs=20) # fs, sampling frequency
output1 = mapper.fit(dataset1)
output2 = mapper.run(dataset2)
```

## Save and Load trained model
```python
mapper.save('trained_model.pkl')
mapper = subtle.load('trained_model.pkl')
```

### Save output result into csv file
```python
output = output1[0]

# export embeddings
df = pd.DataFrame(output.Z)
df.to_csv('Z.csv', header=None, index=None)

# export subclusters
df = pd.DataFrame(output.y)
df.to_csv('y.csv', header=None, index=None)

# export superclusters
df = pd.DataFrame(output.Y)
df.to_csv('Y.csv', header=None, index=None)

# export transition probabilities
df = pd.DataFrame(output.TP)
df.to_csv('TP.csv', header=None, index=None)

# export retention rate
df = pd.DataFrame(output.R)
df.to_csv('TP.csv', header=None, index=None)

```

## Visualize trained result
```python

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(mapper.Z[:, 0], mapper.Z[:, 1], s=1, c=mapper.y) # subclusters
ax[1].scatter(mapper.Z[:, 0], mapper.Z[:, 1], s=1, c=mapper.Y[:, -1]) # superclusters
```
