# 0. Import libraries
```python
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subtle
```


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

# how many superclusters to include for visualization?)
n_superclusters = [2,3,4,5,6,7,8]
# the resoludation of figure
dpi = 300

# get subcluster coordinates
df_umap = pd.DataFrame(model.Z, columns=['dim1', 'dim2'])
df_sub = pd.DataFrame(model.y, columns=['subcluster'])
df = pd.concat([df_umap, df_sub], axis=1)
subcluster_center = df.groupby('subcluster').mean()
k = len(subcluster_center)
x = subcluster_center['dim1']
y = subcluster_center['dim2']

# plot 
sns.set('notebook')
sns.set_style('white')

num_axes = 1+len(n_superclusters)
fig, ax = plt.subplots(1, num_axes, figsize=(5*num_axes, 5), dpi=dpi)
sns.scatterplot(x=model.Z[:, 0], y=model.Z[:, 1], s=1, hue=model.y, palette='viridis', ax=ax[0], legend=False)
for i in range(len(subcluster_center.index)):
    ax[0].text(x[i], y[i], subcluster_center.index[i], ha='center', va='center', fontsize=8)
    ax[0].set_title(f'Subclusters (k={k})')

for idx, n in enumerate(n_superclusters, start=1):
    sns.scatterplot(x=model.Z[:, 0], y=model.Z[:, 1], s=1, hue=model.Y[:, n-1], palette='tab10', ax=ax[idx])
    for i in range(len(subcluster_center.index)):
        ax[idx].text(x[i], y[i], subcluster_center.index[i], ha='center', va='center', fontsize=8)
        ax[idx].set_title(f'Superclusters (n={n})')
```

# 4. Save data and visualize individual output 
```python
for name, output in zip(names, outputs):
    dirname = os.path.join(SAVE_DIR, name)
    os.makedirs(dirname, exist_ok=True)

    # export embeddings
    embeddings = pd.DataFrame(output.Z)
    embeddings.to_csv(os.path.join(dirname, 'embeddings.csv'), header=None, index=None)

    # export subclusters
    subclusters = pd.DataFrame(output.y)
    subclusters.to_csv(os.path.join(dirname, 'subclusters.csv'), header=None, index=None)

    # export superclusters
    superclusters = pd.DataFrame(output.Y)
    superclusters.to_csv(os.path.join(dirname, 'superclusters.csv'), header=None, index=None)

    # export transition probabilities
    transition_probabilities = pd.DataFrame(output.TP)
    transition_probabilities.to_csv(os.path.join(dirname, 'transition_probabilities.csv'), header=None, index=None)

    # export retention rate
    retention_rate = pd.DataFrame(output.R)
    retention_rate.to_csv(os.path.join(dirname, 'retention_rate.csv'), header=None, index=None)

    # export visualization file [TODO: this is just for temporary usage]
    visualization_sup = pd.concat([embeddings, superclusters[6], superclusters[6]], axis=1)
    visualization_sup.to_csv(os.path.join(dirname, 'visualization_sup.csv'), header=None)
    visualization_sub = pd.concat([embeddings, subclusters, subclusters], axis=1)
    visualization_sub.to_csv(os.path.join(dirname, 'visualization_sub.csv'), header=None)

    # visualize activity
    sns.set('notebook')
    sns.set_style('white')
    
    num_axes = 1+len(n_superclusters)
    fig, ax = plt.subplots(1, num_axes, figsize=(5*num_axes, 5), sharex=True, sharey=True, dpi=dpi)
    ax[0].scatter(model.Z[:, 0], model.Z[:, 1], s=1, c='#AAAAAA') # backgroud map
    sns.scatterplot(x=output.Z[:, 0], y=output.Z[:, 1], s=1, hue=output.y, palette='viridis', ax=ax[0], legend=False)
    for i in range(len(subcluster_center.index)):
        ax[0].text(x[i], y[i], subcluster_center.index[i], ha='center', va='center', fontsize=8)
        ax[0].set_title(f'Subclusters (k={k})')

    for idx, n in enumerate(n_superclusters, start=1):
        ax[idx].scatter(model.Z[:, 0], model.Z[:, 1], s=1, c='#AAAAAA') # backgroud map
        sns.scatterplot(x=output.Z[:, 0], y=output.Z[:, 1], s=1, hue=output.Y[:, n-1], palette='tab10', ax=ax[idx])
        for i in range(len(subcluster_center.index)):
            ax[idx].text(x[i], y[i], subcluster_center.index[i], ha='center', va='center', fontsize=8)
            ax[idx].set_title(f'Superclusters (n={n})')

    fig.suptitle(name)
    fig.savefig(os.path.join(dirname, 'embedding_visualize.png'))
    plt.show()
```

# 5. Extract subcluster stay rate 
```python
results = []
for name, output in zip(names, outputs):
    subcusters=output.y
    unique_values, counts = np.unique(subcusters, return_counts=True)
    result = dict(zip(unique_values, counts))
    result['name'] = name
    results.append(result)
df = pd.DataFrame(results).fillna(0).set_index('name').apply(lambda x:x/x.sum(), axis=1).T
df.to_csv(os.path.join(SAVE_DIR, 'subcluster_stay_rate.csv'))
```
