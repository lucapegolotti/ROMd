import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt

class ResampledGeometry:
    def __init__(self, geometry, coeff):
        self.geometry = geometry
        self.resample(coeff)
        self.construct_interpolation_matrices()

    def resample(self, coeff):
        portions = self.geometry.portions
        self.p_portions = []
        for portion in portions:
            p_portion = self.geometry.points[portion[0]:portion[1]+1,:]

            # compute h of the portion
            alength = 0
            for i in range(0, p_portion.shape[0] - 1):
                alength += np.linalg.norm(p_portion[i+1,:] - p_portion[i,:])

            N = int(np.floor(alength / (coeff * self.geometry.h)))


            tck, u = scipy.interpolate.splprep([p_portion[:,0],
                                                p_portion[:,1],
                                                p_portion[:,2]], s=0, k = 3)
            u_fine = np.linspace(0, 1, N)
            x, y, z = interpolate.splev(u_fine, tck)
            p_portion = np.vstack((x,y,z)).transpose()
            self.p_portions.append(p_portion)

    def construct_interpolation_matrices(self):
        p_matrices = []
        i_matrices = []
        portions = self.geometry.portions
        stdev = self.geometry.h * 50

        def kernel(nnorm, stdev):
            # 99% of the gaussian distribution is within 3 stdev from the mean
            return np.exp(-(nnorm / (2 * stdev**2)))

        for ipor in range(0,len(portions)):
            p_portion = self.geometry.points[portions[ipor][0]:portions[ipor][1]+1,:]
            N = self.p_portions[ipor].shape[0]
            M = p_portion.shape[0]
            new_matrix = np.zeros((N,M))
            for i in range(0,N):
                for j in range(0,M):
                    n = np.linalg.norm(self.p_portions[ipor][i,:] -
                                       p_portion[j,:])
                    # we consider 4 stdev to be safe
                    if n < 4 * stdev:
                        new_matrix[i,j] = kernel(n, stdev = stdev)
            p_matrices.append(new_matrix)

            N = p_portion.shape[0]
            M = N
            new_matrix = np.zeros((N,M))
            for i in range(0,N):
                for j in range(0,M):
                    n = np.linalg.norm(p_portion[i,:] -  p_portion[j,:])
                    if n < 4 * stdev:
                        new_matrix[i,j] = kernel(n, stdev = stdev)
            i_matrices.append(new_matrix)

        self.projection_matrices    = p_matrices
        self.interpolation_matrices = i_matrices

    def plot(self, title = "", field = np.zeros((0))):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        indices = self.geometry.inlet + self.geometry.outlets

        # plot inlet
        ax.scatter3D(self.geometry.points[self.geometry.inlet,0],
                     self.geometry.points[self.geometry.inlet,1],
                     self.geometry.points[self.geometry.inlet,2], color = 'blue')

        # plot outlets
        ax.scatter3D(self.geometry.points[self.geometry.outlets,0],
                     self.geometry.points[self.geometry.outlets,1],
                     self.geometry.points[self.geometry.outlets,2], color = 'red')

        if field.size == 0:
            for portion in self.p_portions:
                # ax.plot3D(portion[:,0], portion[:,1], portion[:,2])
                ax.scatter(portion[:,0], portion[:,1], portion[:,2], color = 'black',
                           s = 0.1)
        else:
            nportions = len(self.p_portions)

            fmin = np.min(field)
            fmax = np.max(field)

            for ipor in range(0, nportions):
                proj_values = self.compute_proj_field(ipor, field)

                portion = self.p_portions[ipor]
                ax.scatter(portion[:,0], portion[:,1], portion[:,2], s = 2,
                           c = proj_values, vmin = fmin, vmax = fmax)

        # plot bifurcations
        ax.scatter3D(self.geometry.points[self.geometry.bifurcations,0],
                     self.geometry.points[self.geometry.bifurcations,1],
                     self.geometry.points[self.geometry.bifurcations,2], color = 'green')
        plt.title(title)

    def compute_proj_field(self, index_portion, field):
        values = field[self.geometry.portions[index_portion][0]:
                       self.geometry.portions[index_portion][1]+1]
        weights = np.linalg.solve(self.interpolation_matrices[index_portion], values)
        proj_values = np.matmul(self.projection_matrices[index_portion], weights)
        return proj_values
