import numpy as np
import pandas as pd

def from_avatar_csv(csv: str) -> np.array:
    X = pd.read_csv(csv, header=None).values
    return X.reshape(X.shape[0], X.shape[1]//3, 3) # num_frames, num_joints, num_channels