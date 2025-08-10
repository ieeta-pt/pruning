import numpy as np

from scipy import signal

import cProfile
import pstats
import io

import numba


@numba.njit
def compute_histogram(data, grid_min, grid_max, grid_size):
    dx = (grid_max - grid_min) / (grid_size - 1)
    hist = np.zeros(grid_size, dtype=np.float64)
    for x in data:
        idx = int((x - grid_min) / dx + 0.5)
        if 0 <= idx < grid_size:
            hist[idx] += 1
    hist /= (len(data) * dx)
    return hist


class KDEFFT:
    def __init__(self, data, bandwidth=None, grid_size=1024, grid_range=None, method = 'scott'):

        """
        Performs FFT-based Kernel Density Estimation (KDE) on a 1D dataset.
        """
        self.data = np.asarray(data)
        self.n = len(self.data)

        # Choose bandwidth if not provided (Methods only implemented for 1D Data)
        if bandwidth is None and method == 'scott':
            self.bandwidth = np.std(self.data) * self.n**(-1/5)

        elif bandwidth is None and method == 'silverman':
            self.bandwidth = 1.06 * np.std(self.data) * self.n**(-1/5)
        else:
            self.bandwidth = bandwidth

        # Set up a uniform grid covering the data range
        if grid_range is None:
            buffer = 3 * self.bandwidth  # Extend grid beyond min/max data points
            self.grid_min = self.data.min() - buffer
            self.grid_max = self.data.max() + buffer
        else:
            self.grid_min, self.grid_max = grid_range

        self.grid_size = grid_size  # Number of grid points
        self.dx = (self.grid_max - self.grid_min) / (self.grid_size - 1)
        self.grid = np.linspace(self.grid_min, self.grid_max, self.grid_size)

        # Compute FFT-based KDE
        self.kde_grid_values = self._compute_fft_kde()

        

    def _compute_fft_kde(self):
        """ Compute the FFT-based KDE on a uniform grid """
        

        data_hist = compute_histogram(self.data, self.grid_min, self.grid_max, self.grid_size)#np.histogram(self.data, bins=self.grid_size, range=(self.grid_min, self.grid_max))

        # Compute Gaussian kernel on the same grid
        kernel_extent = 3 * self.bandwidth  # Limit kernel size
        kernel_x = np.arange(-kernel_extent, kernel_extent + self.dx, self.dx)
        kernel_vals = (1 / self.bandwidth) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (kernel_x / self.bandwidth) ** 2)
        


        
        # Perform FFT convolution
        density_full = signal.fftconvolve(data_hist, kernel_vals, mode='same')

        density_full /= np.sum(density_full) * self.dx

        return density_full

    def evaluate(self, points):
        """ Evaluate the KDE at arbitrary points using interpolation """
        return np.interp(points, self.grid, self.kde_grid_values)#self.interp_function(points)

    __call__ = evaluate
    
    def profile(self, points):
        """ Profile the execution of FFT-based KDE """
        pr = cProfile.Profile()
        pr.enable()
        self.__call__(points)  # Call KDE evaluation
        pr.disable()

        # Print sorted profiling stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
        ps.print_stats(20)  # Print top 20 slowest functions
        print(s.getvalue())
    