# 0. Import libraries
```python
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
import subtle

# 1. Train model and save results from local data directory
```python
DATA_DIR = "<data_dir>" # replace with yours
SAVE_DIR = "<save_dir>" # replace with yours
os.makedirs(SAVE_DIR, exist_ok=True) # create if not exist

# Prepare datasets
csv_paths = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.csv')]

names, datasets = [], []
for csv_path in csv_paths:
    name = csv_path.split('/')[-1].replace('.csv', '')
    names.append(name)

    X = pd.read_csv(csv_path, header=None).values
    X = subtle.avatar_preprocess(X) # subtract coordinates with global mean of (x, y, z)
    datasets.append(X)

# Fit model
model = subtle.Mapper(fs=20, include_coordinates=False)
outputs = model.fit(datasets)
results = dict(names=names, outputs=outputs)

# Save model
model_save_path = os.path.join(SAVE_DIR, 'model.pkl')
model.save(model_save_path)

# Save results
results_save_path = os.path.join(SAVE_DIR, 'results.pkl')
with open(results_save_path, 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
```

# 2. Load model and results from local
```python

DATA_DIR = "<data_dir>" # replace with yours
SAVE_DIR = "<save_dir>" # replace with yours

model_save_path = os.path.join(SAVE_DIR, 'model.pkl')
results_save_path = os.path.join(SAVE_DIR, 'results.pkl')

model = subtle.load(model_save_path)

# load
with open(results_save_path, 'rb') as f:
    results = pickle.load(f)

names = results['names']
outputs = results['outputs']
```

# 3. Visualize trained model
```python
# get subcluster coordinates
df_umap = pd.DataFrame(model.Z, columns=['dim1', 'dim2'])
df_sub = pd.DataFrame(model.y, columns=['subcluster'])
df = pd.concat([df_umap, df_sub], axis=1)
subcluster_center = df.groupby('subcluster').mean()
x, y = subcluster_center['dim1'], subcluster_center['dim2']

# plot 
sns.set('notebook')
sns.set_style('white')

n_superclusters = [2,3,4,5,6,7,8]
fig, ax = plt.subplots(1, 1+len(n_superclusters), figsize=(40, 5), sharex=True, sharey=True, dpi=100)
sns.scatterplot(x=model.Z[:, 0], y=model.Z[:, 1], s=1, hue=model.y, palette='viridis', ax=ax[0], legend=False)
for i in range(len(subcluster_center.index)):
    ax[0].text(x[i], y[i], subcluster_center.index[i], ha='center', va='center', fontsize=8)
    ax[0].set_title(f'Subclusters (k={k})')

for n in n_superclusters:
    idx = n-1
    sns.scatterplot(x=model.Z[:, 0], y=model.Z[:, 1], s=1, hue=model.Y[:, idx], palette='tab10', ax=ax[idx])
    for i in range(len(subcluster_center.index)):
        ax[idx].text(x[i], y[i], subcluster_center.index[i], ha='center', va='center', fontsize=8)
        ax[idx].set_title(f'Superclusters (n={n})')
```

