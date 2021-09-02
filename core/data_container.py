import numpy as np

class DataContainer:
    def __init__(self, resampled_geometry, max_stencil_size, pressures, velocities, areas):
        self.datasets = {}
        self.mins     = {}
        self.maxs     = {}

        for stencil_size in range(int(np.floor((max_stencil_size - 1) / 2)) + 1, max_stencil_size + 1):
            self.generate_dataset(resampled_geometry, stencil_size,
                                  pressures, velocities, areas)


    def generate_dataset(self, resampled_geometry, stencil_size, pressures, velocities, areas):
        dataset = []
        for ipor in range(0,len(resampled_geometry.p_portions)):
            times = [ptime for ptime in pressures]
            times.sort()
            area = resampled_geometry.compute_proj_field(ipor, areas)
            for itime in range(0, len(times) - 1):
                pressure = resampled_geometry.compute_proj_field(ipor, pressures[times[itime + 1]])
                velocity = resampled_geometry.compute_proj_field(ipor, velocities[times[itime + 1]])
                p_pressure = resampled_geometry.compute_proj_field(ipor, pressures[times[itime]])
                p_velocity = resampled_geometry.compute_proj_field(ipor, velocities[times[itime]])
                nnodes = pressure.size

                for inode in range(0, nnodes - stencil_size):
                    # current velocities and pressures
                    ps = pressure[inode:inode + stencil_size]
                    qs = velocity[inode:inode + stencil_size]
                    # previous velocities and pressures
                    p_ps = p_pressure[inode:inode + stencil_size]
                    p_qs = p_velocity[inode:inode + stencil_size]
                    ar   = areas[inode:inode + stencil_size]
                    xs = resampled_geometry.p_portions[ipor][inode:inode + stencil_size,0]
                    ys = resampled_geometry.p_portions[ipor][inode:inode + stencil_size,1]
                    zs = resampled_geometry.p_portions[ipor][inode:inode + stencil_size,2]

                    minxs = np.min(xs)
                    maxxs = np.max(xs)
                    minys = np.min(ys)
                    maxys = np.max(ys)
                    minzs = np.min(zs)
                    maxzs = np.max(zs)

                    xs = (xs - minxs) / (maxxs - minxs)
                    ys = (ys - minys) / (maxys - minys)
                    zs = (zs - minzs) / (maxzs - minzs)

                    new_data = ps
                    new_data = np.hstack((new_data,qs))
                    new_data = np.hstack((new_data,p_ps))
                    new_data = np.hstack((new_data,p_qs))
                    new_data = np.hstack((new_data,ar))
                    new_data = np.hstack((new_data,xs))
                    new_data = np.hstack((new_data,ys))
                    new_data = np.hstack((new_data,zs))
                    new_data = np.hstack((new_data,times[itime + 1] - times[itime]))
                    new_data = np.hstack((new_data,maxxs - minxs))
                    new_data = np.hstack((new_data,maxys - minys))
                    new_data = np.hstack((new_data,maxzs - minzs))
                    dataset.append(new_data)

        dataset = np.array(dataset)

        mins = []
        maxs = []

        for j in range(0, dataset.shape[1]):
            m = np.min(dataset[:,j])
            M = np.max(dataset[:,j])
            dataset[:,j] = (dataset[:,j] - m) / (M - m)
            mins.append(m)
            maxs.append(M)

        self.datasets[stencil_size] = dataset
        self.mins[stencil_size]     = mins
        self.maxs[stencil_size]     = maxs

    def get(self, stsize, center, var):
        X = np.copy(self.datasets[stsize])
        mins = np.copy(self.mins[stsize])
        maxs = np.copy(self.maxs[stsize])

        if var == 'pressure':
            offset = 0
        else:
            offset = stsize

        Y = X[:,offset + center]
        my = X[offset + center]
        My = X[offset + center]

        np.delete(X, offset + center, axis = 1)
        np.delete(my, offset + center)
        np.delete(My, offset + center)

        return X, Y, mins, maxs, my, My
