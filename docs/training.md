# Train model and save results from local data directory
```python
import os
import subtle
import pandas as pd
import pickle

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

# Load model and results from local
```python
import os
import subtle
import pandas as pd
import pickle

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
