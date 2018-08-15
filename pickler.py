import numpy as np
import png
import pickle
import load_means as lm

print("loading png...")
means_matrix_1k = lm.png_to_matrix("sparse_means_1k.png")

print("preparing to dump...")
with open("means_matrix_dump",mode='wb') as mmd:
    pickle.dump(means_matrix_1k,mmd)
print("dumped")