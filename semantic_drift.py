import numpy as np
from numpy import linalg as LA

def l1_distance(model1, model2):
    w = zip(model1.get_weights(), model2.get_weights())
    lw = list(w)
    sums = 0
    i = 0
    for ww in lw:
        i += 1
        sums += np.average(np.absolute(ww[0] - ww[1]))
    return sums / i

# This is not normalized, unlike l1_distance()!
# @TODO normalize
def l2_distance(model1, model2):
    if model1.count_params() != model2.count_params():
        raise ValueError("two models have different number of parameters")
    lw = list(zip(model1.get_weights(), model2.get_weights()))
    sums = []
    for ww in lw:
        norm = LA.norm(ww[0] - ww[1])
        sums.append(norm)
    return LA.norm(np.array(sums))