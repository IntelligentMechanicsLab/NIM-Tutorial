import numpy as np
import tensorflow as tf

class matrix_oper:
    def _vectorize(vec):
        """
        Transform a list of arrary into a 1D arrary (num,)
        """
        vec_new = []
        for gI in vec:
            gI = gI.flatten()
            # grads_1.append(gI)
            vec_new = np.concatenate([vec_new,gI],axis=0)
        return vec_new