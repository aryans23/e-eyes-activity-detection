#!/usr/bin/env python3

from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from nltk.metrics.distance import edit_distance
from matplotlib import pyplot as plt

class DynamicTimeWarping(object):
    """ 
        DynamicTimeWarping class
        Input: amplitude histograms
        Output: E matrix
    """
    def __init__(self, amplitude_histograms):
        super(DynamicTimeWarping, self).__init__()
        self.amplitude_histograms = amplitude_histograms
        self.E = []
        self.closest = [] # can be calculated from E as well 
        print("DynamicTimeWarping classifier initialized")

    def dtw(self, x, y, dist):
        assert len(x)
        assert len(y)
        r, c = len(x), len(y)
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
        D1 = D0[1:, 1:] # view
        for i in range(r):
            for j in range(c):
                D1[i, j] = dist(x[i], y[j])
        C = D1.copy()
        for i in range(r):
            for j in range(c):
                D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
        if len(x)==1:
            path = zeros(len(y)), range(len(y))
        elif len(y) == 1:
            path = range(len(x)), zeros(len(x))
        else:
            path = self._traceback(D0)
        return D1[-1, -1] / sum(D1.shape), C, D1, path

    def fastdtw(self, x, y, dist):
        assert len(x)
        assert len(y)
        if ndim(x)==1:
            x = x.reshape(-1,1)
        if ndim(y)==1:
            y = y.reshape(-1,1)
        r, c = len(x), len(y)
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
        D1 = D0[1:, 1:]
        D0[1:,1:] = cdist(x,y,dist)
        C = D1.copy()
        for i in range(r):
            for j in range(c):
                D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
        if len(x)==1:
            path = zeros(len(y)), range(len(y))
        elif len(y) == 1:
            path = range(len(x)), zeros(len(x))
        else:
            path = self._traceback(D0)
        return D1[-1, -1] / sum(D1.shape), C, D1, path

    def _traceback(self, D):
        i, j = array(D.shape) - 2
        p, q = [i], [j]
        while ((i > 0) or (j > 0)):
            tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
            if (tb == 0):
                i -= 1
                j -= 1
            elif (tb == 1):
                i -= 1
            else: # (tb == 2):
                j -= 1
            p.insert(0, i)
            q.insert(0, j)
        return array(p), array(q)

    def get_DTW_matrix(self):
        """
            Input: cached amplitude_histograms
            Output: cached DTW matrix (R x R)
        """
        print("Calculating DTW matrix for %d histograms" %(len(self.amplitude_histograms))) 
        for C_r in self.amplitude_histograms:
            E_r = []
            for C_i in self.amplitude_histograms:
                # euclidean_distances as calculating dependent DTW
                dist, cost, acc, path = self.dtw(C_r,C_i,euclidean_distances)
                E_r.append(dist)
                # print("DTW calculated in current iteration as ", str(dist))
            E_r_np = np.array(E_r)
            # calculating the second smallest dist as smallest would be 0
            closest_activity = E_r_np.argsort()[1] 
            self.closest.append(closest_activity)
            self.E.append(np.array(E_r))
        print("DTW matrix calculated!")
        return np.array(self.E)

    def get_closest_activity(self):
        print("Calculating closest_activity")
        if not self.E:
            self.get_EMD_matrix()
        print("closest_activity calculated")
        return np.array(self.closest)

def test():
    if 0: # 1-D numeric
        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
        dist_fun = manhattan_distances
    elif 0: # 2-D numeric
        x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
        y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
        dist_fun = euclidean_distances
    else: # 1-D list of strings
        #x = ['we', 'shelled', 'clams', 'for', 'the', 'chowder']
        #y = ['class', 'too']
        x = ['i', 'soon', 'found', 'myself', 'muttering', 'to', 'the', 'walls']
        y = ['see', 'drown', 'himself']
        #x = 'we talked about the situation'.split()
        #y = 'we talked about the situation'.split()
        dist_fun = edit_distance
    DTW = DynamicTimeWarping([])
    dist, cost, acc, path = DTW.dtw(x, y, dist_fun)

    # vizualize
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    plt.plot(path[0], path[1], '-o') # relation
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')
    plt.title('Minimum distance: {}'.format(dist))
    plt.show()

### Testing ###
if __name__ == '__main__':
    print("Starting testing for DTW")
    test()