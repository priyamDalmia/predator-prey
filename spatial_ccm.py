# IMPORTS
import pandas as pd 
import numpy as np
from causal_ccm.causal_ccm import ccm
from scipy.spatial import distance

class spatial_ccm(ccm):
    """
    We're checking causality X -> Y       
    Args
        X: timeseries for variable X that could cause Y
        Y: timeseries for variable Y that could be caused by X
        tau: time lag. default = 1
        E: shadow manifold embedding dimension. default = 2
        L: time period/duration to consider (longer = more data). default = length of X
    """
    def __init__(self, X, Y, sample_index, tau=1, E=2, L=None):
        '''
        X: timeseries for variable X that could cause Y
        Y: timeseries for variable Y that could be caused by X
        tau: time lag
        E: shadow manifold embedding dimension
        L: time period/duration to consider (longer = more data)
        We're checking for X -> Y
        '''
        super().__init__(X, Y, tau, E, L)
        self.My = self.dewdrop_shadow_manifold(Y, sample_index) # shadow manifold for Y (we want to know if info from X is in Y)
        self.t_steps, self.dists = self.get_distances(self.My) # for distances between points in manifold

    def dewdrop_shadow_manifold(self, V, sample_index):
        """
        Given
            V: some time series vector
            tau: lag step
            E: shadow manifold embedding dimension
            L: max time step to consider - 1 (starts from 0)
        Returns
            {t:[t, t-tau, t-2*tau ... t-(E-1)*tau]} = Shadow attractor manifold, dictionary of vectors
        """
        V = V[:self.L] # make sure we cut at L
        sample_index = sample_index[:self.L]
        grouped_df = pd.DataFrame({"V": V, "sample_index": sample_index}).groupby("sample_index")

        M = {t:[] for t in range((self.E-1) * self.tau, self.L)} # shadow manifold
        for name, group in grouped_df:
            L = len(group)
            start_index = group.index[0]
            for t in range((self.E-1) * self.tau, L):
                v_lag = [] # lagged values
                a = t + start_index
                for t2 in range(0, self.E-1 + 1): # get lags, we add 1 to E-1 because we want to include E
                    v_lag.append(V[a-t2*self.tau])
                M[a] = v_lag
        filter_M = {k:v for k,v in M.items() if len(v) != 0} # filter out empty vectors
        return filter_M
    
    # get pairwise distances between vectors in the time series
    def get_distances(self, M):
        """
        Args
            M: The shadow manifold from the time series
        Returns
            t_steps: timesteps
            dists: n x n matrix showing distances of each vector at t_step (rows) from other vectors (columns)
        """

        # we extract the time indices and vectors from the manifold M
        # we just want to be safe and convert the dictionary to a tuple (time, vector)
        # to preserve the time inds when we separate them
        t_vec = [(k, v) for k,v in M.items() if len(v) != 0]
        t_steps = np.array([i[0] for i in t_vec])
        vecs = np.array([i[1] for i in t_vec])
        dists = distance.cdist(vecs, vecs)
        return t_steps, dists

    def get_nearest_distances(self, t, t_steps, dists):
        """
        Args:
            t: timestep of vector whose nearest neighbors we want to compute
            t_teps: time steps of all vectors in the manifold M, output of get_distances()
            dists: distance matrix showing distance of each vector (row) from other vectors (columns). output of get_distances()
            E: embedding dimension of shadow manifold M
        Returns:
            nearest_timesteps: array of timesteps of E+1 vectors that are nearest to vector at time t
            nearest_distances: array of distances corresponding to vectors closest to vector at time t
        """
        t_ind = np.where(t_steps == t) # get the index of time t
        dist_t = dists[t_ind].squeeze() # distances from vector at time t (this is one row)

        # get top closest vectors
        nearest_inds = np.argsort(dist_t)[1:self.E+1 + 1] # get indices sorted, we exclude 0 which is distance from itself
        nearest_timesteps = t_steps[nearest_inds] # index column-wise, t_steps are same column and row-wise
        nearest_distances = dist_t[nearest_inds]

        return nearest_timesteps, nearest_distances