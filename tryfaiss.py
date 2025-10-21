import faiss
import numpy as np

# Suppose we have these vectors (float32)
xb = np.random.randn(1000, 768).astype('float32')  # database vectors
xq = np.random.randn(10, 768).astype('float32')    # query vectors

# 1. Build an index (L2 distance)
index = faiss.IndexFlatL2(768)  # dimension must match

# 2. Add the database vectors
index.add(xb)

# 3. Search: get 5 nearest neighbors for each query
k = 5
D, I = index.search(xq, k)  # D = distances, I = indices of nearest neighbors
print(I)  # indices of nearest neighbors for first query
print(D)  # corresponding distances
