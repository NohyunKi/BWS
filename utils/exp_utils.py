import numpy as np
from scipy import stats

def make_corr(x, y):
    def corrcoef(x, y):
        return np.corrcoef(x, y)[0, 1]
    
    def spearman(x, y):
        return stats.spearmanr(x, y)[0]
    
    return corrcoef(x, y), spearman(x, y)