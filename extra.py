# extra.py

import numpy as np

def normalize(v):
    return v / np.sqrt(v.dot(v))

def generate_phi(x, y):
    np.random.seed(333)
    phi = np.random.normal(size=(x, y))
    n = len(phi)
    
    # Perform Gram-Schmidt orthonormalization
    phi[0, :] = normalize(phi[0, :])
    
    for i in range(1, n):
        Ai = phi[i, :]
        for j in range(0, i):
            Aj = phi[j, :]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        phi[i, :] = normalize(Ai)
        
    return phi
