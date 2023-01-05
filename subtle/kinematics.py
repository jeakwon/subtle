import numpy as np
import pandas as pd

class Kinematics:
    def __init__(self, fs, nodes={}, edges={}, angles={}):
        self.fs = fs
        self.nodes = nodes
        self.edges = edges
        self.angles = angles
    
    def __call__(self, X):
        self.X = X
        self.dt = 1/self.fs
        T, V, D = X.shape
        if not self.nodes:
            self.nodes = {v:v for v in range(V)}

        labels = []
        features = []

        for n in range(3):
            for name, i in self.nodes.items():
                labels.append( f'V{n}({i})' )
                features.append( self.V(n, i) )

            for name, (i, j) in self.edges.items():
                labels.append( f'E{n}({i},{j})' )
                features.append( self.E(n, i, j) )

            for name, (i, j, k) in self.angles.items():
                labels.append( f'A{n}({i},{j},{k})' )
                features.append( self.A(n, i, j, k) )
        return pd.DataFrame(
          data=np.vstack(features).T, 
          index=np.arange(len(X))*self.dt, 
          columns=labels)
        
    def v(self, i):
        return self.X[:, i, :]

    def e(self, i, j):
        return self.v(j) - self.v(i)

    def a(self, i, j, k):
        e_ji, e_jk = self.e(i, j), self.e(k, j)
        a = np.einsum('ij,ij->i', e_ji, e_jk)
        b = np.linalg.norm(e_ji, axis=1) * np.linalg.norm(e_jk, axis=1) + 1e-12
        return np.arccos( a / b )

    def V(self, n, i):
        v = self.v(i)
        dv = np.diff(v, n=n, axis=0, prepend=v[:n])
        dt = self.dt**n
        return np.linalg.norm(dv/dt, axis=1)
         
    def E(self, n, i, j):
        e = self.e(i, j)
        de = np.diff(e, n=n, axis=0, prepend=e[:n])
        dt = self.dt**n
        return np.linalg.norm(de/dt, axis=1)

    def A(self, n, i, j, k):
        a = self.a(i, j, k)
        da = np.diff(a, n=n, axis=0, prepend=a[:n])
        dt = self.dt**n
        return np.linalg.norm(da/dt, axis=1)