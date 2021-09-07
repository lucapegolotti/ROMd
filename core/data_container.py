import numpy as np
import random
from itertools import permutations

class DataContainer:
    def __init__(self, resampled_geometry, max_stencil_size, pressures, velocities, areas):
        self.sequential_datasets = {}
        self.sequential_mins     = {}
        self.sequential_maxs     = {}

        for stencil_size in range(int(np.floor((max_stencil_size - 1) / 2)) + 1, max_stencil_size + 1):
            self.generate_sequential_dataset(resampled_geometry, stencil_size,
                                             pressures, velocities, areas)

        self.junctions_datasets = {}
        self.junctions_mins     = {}
        self.junctions_maxs     = {}
        self.generate_junctions_dataset(resampled_geometry, stencil_size, pressures,
                                        velocities, areas)



    # The linear part of the dataset is structrued as follows:
    # [pressure, flowrate, previous_pressure, previous_flowrate, areas,
    # xs, ys, zs, dt, scale_x, scale_y, scale_z]
    def generate_sequential_dataset(self, resampled_geometry, stencil_size, pressures, velocities, areas):
        dataset = []
        times = [ptime for ptime in pressures]
        times.sort()
        for ipor in range(0,len(resampled_geometry.p_portions)):
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
                    ar   = area[inode:inode + stencil_size]
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

        random.shuffle(dataset)
        dataset = np.array(dataset)

        mins = []
        maxs = []

        for j in range(0, dataset.shape[1]):
            m = np.min(dataset[:,j])
            M = np.max(dataset[:,j])
            dataset[:,j] = (dataset[:,j] - m) / (M - m)
            mins.append(m)
            maxs.append(M)

        self.sequential_datasets[stencil_size] = dataset
        self.sequential_mins[stencil_size]     = mins
        self.sequential_maxs[stencil_size]     = maxs

    def generate_junctions_dataset(self, resampled_geometry, stencil_size,
                                   pressures, velocities, areas):
        datasets = {}
        half = int(np.floor((stencil_size-1)/2))
        times = [ptime for ptime in pressures]
        times.sort()
        connectivity = resampled_geometry.geometry.connectivity
        sc = np.sum(np.abs(connectivity), axis = 1)
        nportions = connectivity.shape[1]
        for ijun in range(0, sc.size):
            numbif = int(sc[ijun])
            dataset_junction = []

            inlet = np.where(connectivity[ijun,:] == -1)[0][0]
            outlets = np.where(connectivity[ijun,:] == 1)[0]

            xs_inlet = resampled_geometry.p_portions[inlet][-half-1:,0]
            ys_inlet = resampled_geometry.p_portions[inlet][-half-1:,1]
            zs_inlet = resampled_geometry.p_portions[inlet][-half-1:,2]

            xs_outlets = resampled_geometry.p_portions[outlets[0]][1:half+1:,0]
            ys_outlets = resampled_geometry.p_portions[outlets[0]][1:half+1:,1]
            zs_outlets = resampled_geometry.p_portions[outlets[0]][1:half+1:,2]

            for outlet in outlets[1:]:
                xs_outlets = np.vstack((xs_outlets,resampled_geometry.p_portions[outlet][1:half+1:,0]))
                ys_outlets = np.vstack((ys_outlets,resampled_geometry.p_portions[outlet][1:half+1:,1]))
                zs_outlets = np.vstack((zs_outlets,resampled_geometry.p_portions[outlet][1:half+1:,2]))

            minxs_inlet = np.min(xs_inlet)
            maxxs_inlet = np.max(xs_inlet)
            minys_inlet = np.min(ys_inlet)
            maxys_inlet = np.max(ys_inlet)
            minzs_inlet = np.min(zs_inlet)
            maxzs_inlet = np.max(zs_inlet)

            minxs_outlets = np.min(np.min(xs_outlets))
            maxxs_outlets = np.max(np.max(xs_outlets))
            minys_outlets = np.min(np.min(ys_outlets))
            maxys_outlets = np.max(np.max(ys_outlets))
            minzs_outlets = np.min(np.min(zs_outlets))
            maxzs_outlets = np.max(np.max(zs_outlets))

            minxs = np.min((minxs_inlet, minxs_outlets))
            maxxs = np.max((maxxs_inlet, maxxs_outlets))
            minys = np.min((minys_inlet, minys_outlets))
            maxys = np.max((maxys_inlet, maxys_outlets))
            minzs = np.min((minzs_inlet, minzs_outlets))
            maxzs = np.max((maxzs_inlet, maxzs_outlets))

            xs_inlet = (xs_inlet - minxs) / (maxxs - minxs)
            ys_inlet = (ys_inlet - minys) / (maxys - minys)
            zs_inlet = (zs_inlet - minzs) / (maxzs - minzs)

            xs_outlets = (xs_outlets - minxs) / (maxxs - minxs)
            ys_outlets = (ys_outlets - minys) / (maxys - minys)
            zs_outlets = (zs_outlets - minzs) / (maxzs - minzs)

            area_inlet = resampled_geometry.compute_proj_field(inlet, areas)[-half-1:]

            area_outlets = resampled_geometry.compute_proj_field(outlets[0], areas)[1:half+1]
            for outlet in outlets[1:]:
                area_outlets = np.vstack((area_outlets,
                               resampled_geometry.compute_proj_field(outlet, areas)[1:half+1]))

            perm_out = list(permutations(range(0, len(outlets))))

            for itime in range(0, len(times) - 1):
                p_in = resampled_geometry.compute_proj_field(inlet, pressures[times[itime + 1]])[-half-1:]
                q_in = resampled_geometry.compute_proj_field(inlet, velocities[times[itime + 1]])[-half-1:]
                prev_p_in = resampled_geometry.compute_proj_field(inlet, pressures[times[itime]])[-half-1:]
                prev_q_in = resampled_geometry.compute_proj_field(inlet, velocities[times[itime]])[-half-1:]

                p_out = []
                q_out = []
                prev_p_out = []
                prev_q_out = []

                for outlet in outlets:
                    p_out.append(resampled_geometry.compute_proj_field(outlet, pressures[times[itime + 1]])[1:half+1:])
                    q_out.append(resampled_geometry.compute_proj_field(outlet, velocities[times[itime + 1]])[1:half+1:])
                    prev_p_out.append(resampled_geometry.compute_proj_field(outlet, pressures[times[itime]])[1:half+1:])
                    prev_q_out.append(resampled_geometry.compute_proj_field(outlet, velocities[times[itime]])[1:half+1:])

                for p in perm_out:
                    new_data = p_in
                    for oind in p:
                        new_data = np.hstack((new_data,p_out[oind]))
                    new_data = np.hstack((new_data,q_in))
                    for oint in p:
                        new_data = np.hstack((new_data,q_out[oind]))
                    new_data = np.hstack((new_data,prev_p_in))
                    for oint in p:
                        new_data = np.hstack((new_data,prev_p_out[oind]))
                    new_data = np.hstack((new_data,prev_q_in))
                    for oint in p:
                        new_data = np.hstack((new_data,prev_q_out[oind]))
                    new_data = np.hstack((new_data,area_inlet))
                    for oint in p:
                        new_data = np.hstack((new_data,area_outlets[oind,:]))
                    new_data = np.hstack((new_data,xs_inlet))
                    for oint in p:
                        new_data = np.hstack((new_data,xs_outlets[oind,:]))
                    new_data = np.hstack((new_data,ys_inlet))
                    for oint in p:
                        new_data = np.hstack((new_data,ys_outlets[oind,:]))
                    new_data = np.hstack((new_data,zs_inlet))
                    for oint in p:
                        new_data = np.hstack((new_data,zs_outlets[oind,:]))
                    new_data = np.hstack((new_data,times[itime + 1] - times[itime]))
                    new_data = np.hstack((new_data,maxxs - minxs))
                    new_data = np.hstack((new_data,maxys - minys))
                    new_data = np.hstack((new_data,maxzs - minzs))
                    dataset_junction.append(new_data)

        random.shuffle(dataset_junction)
        if sc[ijun] in datasets:
            datasets[sc[ijun]] = np.vstack((datasets[sc[ijun]], np.array(dataset_junction)))
        else:
            datasets[sc[ijun]] = np.array(dataset_junction)

        for nbif in datasets:
            mins = []
            maxs = []

            dataset = datasets[nbif]
            for j in range(0, dataset.shape[1]):
                m = np.min(dataset[:,j])
                M = np.max(dataset[:,j])
                if M - m > 1e-15:
                    dataset[:,j] = (dataset[:,j] - m) / (M - m)
                else:
                    dataset[:,j] = dataset[:,j] * 0
                mins.append(m)
                maxs.append(M)

            self.junctions_datasets[int(nbif)] = dataset
            self.junctions_mins[int(nbif)]     = mins
            self.junctions_maxs[int(nbif)]     = maxs

    def get_sequential_dataset(self, stsize, center, var):
        X = np.copy(self.sequential_datasets[stsize])
        mins = np.copy(self.sequential_mins[stsize])
        maxs = np.copy(self.sequential_maxs[stsize])

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

    def get_junction_dataset(self, number_bifurcations, stencil_size, var):
        half = int(np.floor((stencil_size-1)/2))
        X = np.copy(self.junctions_datasets[number_bifurcations])
        mins = np.copy(self.junctions_mins[number_bifurcations])
        maxs = np.copy(self.junctions_maxs[number_bifurcations])

        if var == 'pressure':
            offset = 0
        else:
            offset = number_bifurcations * half + 1

        Y = X[:,offset + half]
        my = X[offset + half]
        My = X[offset + half]

        np.delete(X, offset + half, axis = 1)
        np.delete(my, offset + half)
        np.delete(My, offset + half)

        return X, Y, mins, maxs, my, My
