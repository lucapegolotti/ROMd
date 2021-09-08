import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt

class ResampledGeometry:
    def __init__(self, geometry, coeff):
        self.geometry = geometry
        self.resample(coeff)
        self.construct_interpolation_matrices()

    def assign_area(self, area):
        self.areas = []
        nportions = len(self.p_portions)
        for ipor in range(0, nportions):
            self.areas.append(self.compute_proj_field(ipor, area))

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
        stdevcoeff = 40

        def kernel(nnorm, h):
            # 99% of the gaussian distribution is within 3 stdev from the mean
            return np.exp(-(nnorm / (2 * (h * stdevcoeff)**2)))

        for ipor in range(0,len(portions)):
            p_portion = self.geometry.points[portions[ipor][0]:portions[ipor][1]+1,:]
            N = self.p_portions[ipor].shape[0]
            M = p_portion.shape[0]
            new_matrix = np.zeros((N,M))

            hs = []
            for j in range(0,M):
                h1 = -1
                h2 = -1
                if j != M-1:
                    h1 = np.linalg.norm(p_portion[j+1,:] - p_portion[j,:])
                if j != 0:
                    h2 = np.linalg.norm(p_portion[j,:] - p_portion[j-1,:])
                h = np.max((h1, h2))
                hs.append(h)
            for i in range(0,N):
                for j in range(0,M):
                    n = np.linalg.norm(self.p_portions[ipor][i,:] -
                                       p_portion[j,:])
                    # we consider 4 stdev to be safe
                    # if n < 4 * hs[j] * stdevcoeff:
                    new_matrix[i,j] = kernel(n,hs[j])

            p_matrices.append(new_matrix)

            N = p_portion.shape[0]
            M = N
            new_matrix = np.zeros((N,M))
            for i in range(0,N):
                for j in range(0,M):
                    n = np.linalg.norm(p_portion[i,:] -  p_portion[j,:])
                    # if n < 4 * hs[j] * stdevcoeff:
                    new_matrix[i,j] = kernel(n,hs[j])

            i_matrices.append(new_matrix)

        self.projection_matrices    = p_matrices
        self.interpolation_matrices = i_matrices

    def plot(self, title = "", field = np.zeros((0))):
        fig = plt.figure(10)
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

        for portion in self.geometry.portions:
            ax.plot3D(self.geometry.points[portion[0]:portion[1]+1,0],
                      self.geometry.points[portion[0]:portion[1]+1,1],
                      self.geometry.points[portion[0]:portion[1]+1,2], 'g--')

        if field.size == 0:
            index = 0
            for portion in self.p_portions:
                # ax.plot3D(portion[:,0], portion[:,1], portion[:,2])
                ax.scatter(portion[:,0], portion[:,1], portion[:,2], color = 'black',
                           s = 0.1)
                N = portion.shape[0]
                ax.text(portion[int(N/2),0],
                        portion[int(N/2),1],
                        portion[int(N/2),2],
                        str(index),
                        color='black',
                        fontsize = 7)
                index = index + 1
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

    def compare_field_along_centerlines(self, field):
        nportions = len(self.p_portions)

        for ipor in range(0, nportions):
            # plot original one
            fig = plt.figure(ipor)
            ax = plt.axes()

            x = [0]
            iin = self.geometry.portions[ipor][0]
            iend = self.geometry.portions[ipor][1]
            points = self.geometry.points[iin:iend+1]

            for ip in range(1, points.shape[0]):
                x.append(np.linalg.norm(points[ip,:] - points[ip-1,:]) + x[ip-1])

            ax.plot(np.array(x), field[iin:iend+1], 'k--o')

            x = [0]
            points = self.p_portions[ipor]

            for ip in range(1, points.shape[0]):
                x.append(np.linalg.norm(points[ip,:] - points[ip-1,:]) + x[ip-1])

            r_field = self.compute_proj_field(ipor, field)
            ax.plot(np.array(x), r_field, 'r-o')
