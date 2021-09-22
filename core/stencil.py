import numpy as np
import tensorflow as tf
from keras.models import Model

class Stencil:
    def __init__(self, points, stsize, areas, center):
        self.points = points
        self.areas = areas
        self.center = center
        self.stsize = stsize

        self.scaled_points = np.copy(self.points)
        self.points_mins = []
        self.points_maxs = []
        for j in range(0, self.points.shape[1]):
            cmin = np.min(self.scaled_points[:,j])
            cmax = np.max(self.scaled_points[:,j])
            self.scaled_points[:,j] = (self.scaled_points[:,j] - cmin) / (cmax - cmin)
            self.points_mins.append(cmin)
            self.points_maxs.append(cmax)

        self.points_mins = np.array(self.points_mins)
        self.points_maxs = np.array(self.points_maxs)
        
    def plot(self, fig, ax):
        ax.scatter3D(self.points[:,0],
                     self.points[:,1],
                     self.points[:,2], color = 'blue')

    def set_type_sequential(self, global_range):
        self.type = 'sequential'
        self.global_range = global_range

    def set_type_junction(self, njunctions, global_range_inlet, global_ranges_outlets):
        self.type = 'junction'
        self.njunctions = njunctions
        self.global_range_inlet = global_range_inlet
        self.global_ranges_outlets = global_ranges_outlets

    def evaluate_model(self, vector, prev_vector, dt):
        offset = int(np.floor(vector.size / 2))

        if self.type == 'sequential':
            values_p = vector[self.global_range]
            values_q = vector[self.global_range + offset]
            prev_values_p = prev_vector[self.global_range]
            prev_values_q = prev_vector[self.global_range + offset]

            x = np.copy(values_p)
            x = np.hstack((x, values_q))
            x = np.hstack((x, prev_values_p))
            x = np.hstack((x, prev_values_q))
            x = np.hstack((x, self.areas))
            x = np.hstack((x, self.scaled_points[:,0]))
            x = np.hstack((x, self.scaled_points[:,1]))
            x = np.hstack((x, self.scaled_points[:,2]))
            x = np.hstack((x, dt))
            x = np.hstack((x, self.points_maxs - self.points_mins))

            x_p = np.copy(x)
            # x_p = np.delete(x_p, self.center)

            # scale
            x_p = (x_p - self.mins_x[self.model_name_pressure])

            for i in range(0, self.maxs_x[self.model_name_pressure].size):
                if self.maxs_x[self.model_name_pressure][i] - self.mins_x[self.model_name_pressure][i] > 1e-12:
                    x_p[i] = x_p[i] / (self.maxs_x[self.model_name_pressure][i] - self.mins_x[self.model_name_pressure][i])
                else:
                    x_p[i] = 0
            
            diff = self.X[self.model_name_pressure] - x_p
            ii = np.where(np.linalg.norm(diff, axis = 1) < 1e-3)[0]
            if len(ii) == 0:
                print('hello')
            truep = self.Y[self.model_name_pressure][ii]
            
            p = self.models[self.model_name_pressure](np.expand_dims(x_p,axis=0)).numpy()
            print(str(p) + ' ' + str(truep))
            # p = self.mins_y[self.model_name_pressure] + p * (self.maxs_y[self.model_name_pressure] - self.mins_y[self.model_name_pressure])
            pres = (p - values_p[self.center]) 

            x_q = np.copy(x)
           #  x_q = np.delete(x_q, self.center + values_p.size)

            # scale
            x_q = (x_q - self.mins_x[self.model_name_velocity])
            for i in range(0, self.maxs_x[self.model_name_velocity].size):
                if self.maxs_x[self.model_name_velocity][i] - self.mins_x[self.model_name_velocity][i] > 1e-12:
                    x_q[i] = x_q[i] / (self.maxs_x[self.model_name_velocity][i] - self.mins_x[self.model_name_velocity][i])
                else:
                    x_q[i] = 0
                    
            diff = self.X[self.model_name_velocity] - x_q
            ii = np.where(np.linalg.norm(diff, axis = 1) < 1e-3)[0]
            trueq = self.Y[self.model_name_velocity][ii]

            q = self.models[self.model_name_velocity](np.expand_dims(x_q,axis=0)).numpy()
            print(str(q) + ' ' + str(trueq))
            # q = self.mins_y[self.model_name_velocity] + q * (self.maxs_y[self.model_name_velocity] - self.mins_y[self.model_name_velocity])
            qres = (q - values_q[self.center])

        if self.type == 'junction':
            values_p_inlet = vector[self.global_range_inlet]
            values_q_inlet = vector[self.global_range_inlet + offset]
            prev_values_p_inlet = prev_vector[self.global_range_inlet]
            prev_values_q_inlet = prev_vector[self.global_range_inlet + offset]

            values_p_outlet = []
            values_q_outlet = []
            prev_values_p_outlet = []
            prev_values_q_outlet = []
            for range_outlets in self.global_ranges_outlets:
                values_p_outlet.append(vector[range_outlets])
                values_q_outlet.append(vector[range_outlets + offset])
                prev_values_p_outlet.append(prev_vector[range_outlets])
                prev_values_q_outlet.append(prev_vector[range_outlets + offset])

            x = np.copy(values_p_inlet)
            for p_outlet in values_p_outlet:
                x = np.hstack((x, p_outlet))
            totp = x.size
            x = np.hstack((x, values_q_inlet))
            for q_outlet in values_q_outlet:
                x = np.hstack((x, q_outlet))
            x = np.hstack((x, prev_values_p_inlet))
            for prev_p in prev_values_p_outlet:
                x = np.hstack((x, prev_p))
            x = np.hstack((x, prev_values_q_inlet))
            for prev_q in prev_values_q_outlet:
                x = np.hstack((x, prev_q))
            x = np.hstack((x, self.areas))
            x = np.hstack((x, self.scaled_points[:,0]))
            x = np.hstack((x, self.scaled_points[:,1]))
            x = np.hstack((x, self.scaled_points[:,2]))
            x = np.hstack((x, dt))
            x = np.hstack((x, self.points_maxs - self.points_mins))

            x_p = np.copy(x)
            x_p = np.delete(x_p, self.center)

            # scale
            x_p = (x_p - self.mins_x[self.model_name_pressure])

            for i in range(0, self.maxs_x[self.model_name_pressure].size):
                if self.maxs_x[self.model_name_pressure][i] - self.mins_x[self.model_name_pressure][i] > 1e-12:
                    x_p[i] = x_p[i] / (self.maxs_x[self.model_name_pressure][i] - self.mins_x[self.model_name_pressure][i])
                else:
                    x_p[i] = 0
                    
            diff = self.X[self.model_name_pressure] - x_p
            ii = np.where(np.linalg.norm(diff, axis = 1) < 1e-12)[0]
            truep = self.Y[self.model_name_pressure][ii]

            p = self.models[self.model_name_pressure](np.expand_dims(x_p,axis=0)).numpy()
            p = self.mins_y[self.model_name_pressure] + p * (self.maxs_y[self.model_name_pressure] - self.mins_y[self.model_name_pressure])
            pres = (p - values_p_inlet[self.center]) 

            x_q = np.copy(x)
            x_q = np.delete(x_q, self.center + totp)

            # scale
            x_q = (x_q - self.mins_x[self.model_name_velocity])
            for i in range(0, self.maxs_x[self.model_name_velocity].size):
                if self.maxs_x[self.model_name_velocity][i] != self.mins_x[self.model_name_velocity][i]:
                    x_q[i] = x_q[i] / (self.maxs_x[self.model_name_velocity][i] - self.mins_x[self.model_name_velocity][i])
                else:
                    x_q[i] = 0

            q = self.models[self.model_name_velocity](np.expand_dims(x_q,axis=0)).numpy()
            q = self.mins_y[self.model_name_velocity] + q * (self.maxs_y[self.model_name_velocity] - self.mins_y[self.model_name_velocity])
            qres = (q - values_q_inlet[self.center])
            
        # print('---')
        # print(pres)
        # print(qres)

        return pres, qres

    def evaluate_model_jacobian(self, vector, prev_vector, dt):
        offset = int(np.floor(vector.size / 2))

        if self.type == 'sequential':
            values_p = vector[self.global_range]
            values_q = vector[self.global_range + offset]
            prev_values_p = prev_vector[self.global_range]
            prev_values_q = prev_vector[self.global_range + offset]

            x = np.copy(values_p)
            x = np.hstack((x, values_q))
            x = np.hstack((x, prev_values_p))
            x = np.hstack((x, prev_values_q))
            x = np.hstack((x, self.areas))
            x = np.hstack((x, self.scaled_points[:,0]))
            x = np.hstack((x, self.scaled_points[:,1]))
            x = np.hstack((x, self.scaled_points[:,2]))
            x = np.hstack((x, dt))
            x = np.hstack((x, self.points_maxs - self.points_mins))

            x_p = np.copy(x)
            # x_p = np.delete(x_p, self.center)

            # scale
            x_p = (x_p - self.mins_x[self.model_name_pressure])

            for i in range(0, self.maxs_x[self.model_name_pressure].size):
                if self.maxs_x[self.model_name_pressure][i] - self.mins_x[self.model_name_pressure][i] > 1e-12:
                    x_p[i] = x_p[i] / (self.maxs_x[self.model_name_pressure][i] - self.mins_x[self.model_name_pressure][i])
                else:
                    x_p[i] = 0

            xt = tf.convert_to_tensor(np.expand_dims(x_p,axis=0))
            with tf.GradientTape() as g:
                g.watch(xt)
                p = self.models[self.model_name_pressure](xt)
            grads_p = g.batch_jacobian(p, xt).numpy().squeeze()
            # grads_p = np.divide(grads_p, self.maxs_x[self.model_name_pressure] - self.mins_x[self.model_name_pressure])
            # grads_p = grads_p * (self.maxs_y[self.model_name_pressure] - self.mins_y[self.model_name_pressure])
            grads_p = grads_p[:values_p.size + values_q.size]
            # grads_p = grads_p[:values_p.size + values_q.size - 1]
            # grads_p = np.insert(grads_p, self.center, -1)
            grads_p[self.center] = grads_p[self.center] - 1

            x_q = np.copy(x)
            # x_q = np.delete(x_q, self.center + values_p.size)

            # scale
            x_q = (x_q - self.mins_x[self.model_name_velocity])
            for i in range(0, self.maxs_x[self.model_name_velocity].size):
                if self.maxs_x[self.model_name_velocity][i] - self.mins_x[self.model_name_velocity][i] > 1e-12:
                    x_q[i] = x_q[i] / (self.maxs_x[self.model_name_velocity][i] - self.mins_x[self.model_name_velocity][i])
                else:
                    x_q[i] = 0
            
            xt = tf.convert_to_tensor(np.expand_dims(x_q,axis=0))
            with tf.GradientTape() as g:
                g.watch(xt)
                q = self.models[self.model_name_velocity](xt)
            grads_q = g.batch_jacobian(q, xt).numpy().squeeze()
            # grads_q = np.divide(grads_q, self.maxs_x[self.model_name_velocity] - self.mins_x[self.model_name_velocity])
            # grads_q = grads_q * (self.maxs_y[self.model_name_velocity] - self.mins_y[self.model_name_velocity])
            grads_q = grads_q[:values_p.size + values_q.size]
            # grads_q = grads_q[:values_p.size + values_q.size - 1]
            # grads_q = np.insert(grads_q, self.center + values_p.size, -1)
            grads_q[self.center + values_p.size] = grads_q[self.center + values_p.size] - 1

            gr1 = np.copy(self.global_range)
            gr2 = np.copy(self.global_range) + offset
            
            grp = np.hstack((gr1, gr2))
            grq = np.hstack((gr1, gr2))

        if self.type == 'junction':
            values_p_inlet = vector[self.global_range_inlet]
            values_q_inlet = vector[self.global_range_inlet + offset]
            prev_values_p_inlet = prev_vector[self.global_range_inlet]
            prev_values_q_inlet = prev_vector[self.global_range_inlet + offset]

            values_p_outlet = []
            values_q_outlet = []
            prev_values_p_outlet = []
            prev_values_q_outlet = []
            for range_outlets in self.global_ranges_outlets:
                values_p_outlet.append(vector[range_outlets])
                values_q_outlet.append(vector[range_outlets + offset])
                prev_values_p_outlet.append(prev_vector[range_outlets])
                prev_values_q_outlet.append(prev_vector[range_outlets + offset])

            x = np.copy(values_p_inlet)
            for p_outlet in values_p_outlet:
                x = np.hstack((x, p_outlet))
            totp = x.size
            x = np.hstack((x, values_q_inlet))
            for q_outlet in values_q_outlet:
                x = np.hstack((x, q_outlet))
            tot = x.size
            x = np.hstack((x, prev_values_p_inlet))
            for prev_p in prev_values_p_outlet:
                x = np.hstack((x, prev_p))
            x = np.hstack((x, prev_values_q_inlet))
            for prev_q in prev_values_q_outlet:
                x = np.hstack((x, prev_q))
            x = np.hstack((x, self.areas))
            x = np.hstack((x, self.scaled_points[:,0]))
            x = np.hstack((x, self.scaled_points[:,1]))
            x = np.hstack((x, self.scaled_points[:,2]))
            x = np.hstack((x, dt))
            x = np.hstack((x, self.points_maxs - self.points_mins))

            x_p = np.copy(x)
            x_p = np.delete(x_p, self.center)
            
            x_p = (x_p - self.mins_x[self.model_name_pressure])
            for i in range(0, self.maxs_x[self.model_name_pressure].size):
                if self.maxs_x[self.model_name_pressure][i] - self.mins_x[self.model_name_pressure][i] > 1e-12:
                    x_p[i] = x_p[i] / (self.maxs_x[self.model_name_pressure][i] - self.mins_x[self.model_name_pressure][i])
                else:
                    x_p[i] = 0

            xt = tf.convert_to_tensor(np.expand_dims(x_p,axis=0))
            with tf.GradientTape() as g:
                g.watch(xt)
                p = self.models[self.model_name_pressure](xt)
            grads_p = g.batch_jacobian(p, xt).numpy().squeeze()
            grads_p = np.divide(grads_p, self.maxs_x[self.model_name_pressure] - self.mins_x[self.model_name_pressure])
            grads_p = grads_p * (self.maxs_y[self.model_name_pressure] - self.mins_y[self.model_name_pressure])
            grads_p = grads_p[:tot - 1]
            grads_p = np.insert(grads_p, self.center, -1)

            x_q = np.copy(x)
            x_q = np.delete(x_q, self.center + totp)

            # scale
            x_q = (x_q - self.mins_x[self.model_name_velocity])
            for i in range(0, self.maxs_x[self.model_name_velocity].size):
                if self.maxs_x[self.model_name_velocity][i] - self.mins_x[self.model_name_velocity][i] > 1e-12:
                    x_q[i] = x_q[i] / (self.maxs_x[self.model_name_velocity][i] - self.mins_x[self.model_name_velocity][i])
                else:
                    x_q[i] = 0

            xt = tf.convert_to_tensor(np.expand_dims(x_q,axis=0))
            with tf.GradientTape() as g:
                g.watch(xt)
                q = self.models[self.model_name_velocity](xt)
            grads_q = g.batch_jacobian(q, xt).numpy().squeeze()
            grads_q = np.divide(grads_q, self.maxs_x[self.model_name_velocity] - self.mins_x[self.model_name_velocity])
            grads_q = grads_q * (self.maxs_y[self.model_name_velocity] - self.mins_y[self.model_name_velocity])
            grads_q = grads_q[:tot - 1]
            grads_q = np.insert(grads_q, totp + self.center, -1)


            gr1 = np.copy(self.global_range_inlet)
            gr2 = np.copy(self.global_range_inlet) + offset

            for range_outlets in self.global_ranges_outlets:
                gr1 = np.hstack((gr1, range_outlets))
                gr2 = np.hstack((gr2, range_outlets + offset))

            offset = gr1.size

            grp = np.hstack((gr1, gr2))
            grq = np.hstack((gr1, gr2))

        return grads_p, grads_q, grp, grq


    def load_model(self, fdr, models, mins_x, maxs_x, mins_y, maxs_y, X, Y):
        if self.type == 'sequential':
            self.model_name_pressure = fdr + '/S_pressurest' + str(self.stsize) + 'c' + str(self.center)
            self.model_name_velocity = fdr + '/S_velocityst' + str(self.stsize) + 'c' + str(self.center)
        if self.type == 'junction':
            self.model_name_pressure = fdr + '/J_pressurest' + str(self.stsize) + 'nj' + str(self.njunctions)
            self.model_name_velocity = fdr + '/J_velocityst' + str(self.stsize) + 'nj' + str(self.njunctions)
        if self.model_name_pressure not in models:
            models[self.model_name_pressure] = tf.keras.models.load_model(self.model_name_pressure)
            mins_x[self.model_name_pressure] = np.load(self.model_name_pressure + '/mins_x.npy')
            maxs_x[self.model_name_pressure] = np.load(self.model_name_pressure + '/maxs_x.npy')
            mins_y[self.model_name_pressure] = np.load(self.model_name_pressure + '/mins_y.npy')
            maxs_y[self.model_name_pressure] = np.load(self.model_name_pressure + '/maxs_y.npy')
            X[self.model_name_pressure] = np.load(self.model_name_pressure + '/X.npy')
            Y[self.model_name_pressure] = np.load(self.model_name_pressure + '/Y.npy')
        if self.model_name_velocity not in models:
            models[self.model_name_velocity] = tf.keras.models.load_model(self.model_name_velocity)
            mins_x[self.model_name_velocity] = np.load(self.model_name_velocity + '/mins_x.npy')
            maxs_x[self.model_name_velocity] = np.load(self.model_name_velocity + '/maxs_x.npy')
            mins_y[self.model_name_velocity] = np.load(self.model_name_velocity + '/mins_y.npy')
            maxs_y[self.model_name_velocity] = np.load(self.model_name_velocity + '/maxs_y.npy')
            X[self.model_name_velocity] = np.load(self.model_name_velocity + '/X.npy')
            Y[self.model_name_velocity] = np.load(self.model_name_velocity + '/Y.npy')
        self.models = models
        self.mins_x = mins_x
        self.maxs_x = maxs_x
        self.mins_y = mins_y
        self.maxs_y = maxs_y
        # these are just for debugging
        self.X = X
        self.Y = Y


class StencilsArray:
    def __init__(self, resampled_geometry, max_stencil_size):
        self.resampled_geometry = resampled_geometry
        self.max_stencil_size = max_stencil_size
        self.generate()

    def generate(self):
        half = int(np.floor((self.max_stencil_size - 1) / 2))
        stencils = []

        offset = 0
        nportions = len(self.resampled_geometry.p_portions)
        connectivity = self.resampled_geometry.geometry.connectivity

        inlets = []
        outlets = []
        # we look for inlets and outlets
        njuns = connectivity.shape[0]
        for ijun in range(0, njuns):
            inlets.append(np.where(connectivity[ijun,:] == -1)[0][0])
            outlets.append(np.where(connectivity[ijun,:] == 1)[0])
            
        # compute npoints for every portion
        npointss = []
        for ipor in range(0, nportions):
           isoutlet = False
           npoints = self.resampled_geometry.p_portions[ipor].shape[0]

           for ijun in range(0, njuns):
               if ipor in outlets[ijun]:
                   isoutlet = True

           # if it's an outlet, we don't count the first node (junction)
           if isoutlet:
               npoints = npoints - 1

           npointss.append(npoints)
        self.npoints = npointss

        offsets = [0]
        lasts = []
        # compute offset in global vector for every portion
        for ipor in range(0, nportions - 1):
            offsets.append(offsets[-1] + self.npoints[ipor])

        self.offsets = offsets

        for ipor in range(0, nportions):
            points = self.resampled_geometry.p_portions[ipor]
            npoints = points.shape[0]

            # see if current portion is an inlet or an outlet
            isinlet = False
            isoutlet = False
            for ijun in range(0, njuns):
                if inlets[ijun] == ipor:
                    isinlet = True
                    corroutlets = outlets[ijun]
                if ipor in outlets[ijun]:
                    isoutlet = True
                    
            shift = 0
            if isoutlet:
                shift = -1

            for inode in range(0, npoints):
                if inode == 0 and isoutlet:
                    continue

                # if I am here, then I have to add a bifurcation model
                if inode == npoints - 1 and isinlet:
                    points = self.resampled_geometry.p_portions[ipor][-half-1:,:]
                    areas = self.resampled_geometry.areas[ipor][-half-1:]

                    range_inlet = np.arange(self.resampled_geometry.p_portions[ipor].shape[0]-half-1 + offsets[ipor] + shift,
                                        self.resampled_geometry.p_portions[ipor].shape[0] + offsets[ipor] + shift).astype(int)

                    ranges_outlets = []
                    for outlet in corroutlets:
                        # note that we exclude the first point (junction) for the outlets
                        # as in the data generation
                        points = np.vstack((points, self.resampled_geometry.p_portions[outlet][1:half+1,:]))
                        curnpoints = self.resampled_geometry.p_portions[outlet].shape[0]
                        ranges_outlets.append(np.arange(offsets[outlet],
                                              half + offsets[outlet]).astype(int))
                        areas = np.hstack((areas, self.resampled_geometry.areas[outlet][1:half+1]))

                    stencils.append(Stencil(points, self.max_stencil_size,
                                            areas, half))

                    stencils[-1].set_type_junction(len(corroutlets) + 1,
                                                   range_inlet,
                                                   ranges_outlets)

                # if I am here, then I have to add a sequential model
                else:
                    istart = np.max((0, inode - half))
                    iend = np.min((npoints, inode + half + 1))

                    points = self.resampled_geometry.p_portions[ipor][istart:iend,:]
                    areas = self.resampled_geometry.areas[ipor][istart:iend]

                    stencils.append(Stencil(points,
                                            iend - istart,
                                            areas,
                                            inode - istart))
                    stencils[-1].set_type_sequential(np.arange(istart + offsets[ipor] + shift, iend + offsets[ipor] + shift).astype(int))
        self.stencils = stencils

    def generate_global_vector(self, pressure, velocity):
        portions = self.resampled_geometry.p_portions
        connectivity = self.resampled_geometry.geometry.connectivity

        outlets = []
        # we look for inlets and outlets
        njuns = connectivity.shape[0]
        for ijun in range(0, njuns):
            outlets.append(np.where(connectivity[ijun,:] == 1)[0])

        global_pressure = np.zeros((0))
        global_velocity = np.zeros((0))
        for ipor in range(0, len(portions)):
            proj_pressure = self.resampled_geometry.compute_proj_field(ipor, pressure)
            proj_velocity = self.resampled_geometry.compute_proj_field(ipor, velocity)

            isoutlet = False
            for ijun in range(0, njuns):
                if ipor in outlets[ijun]:
                    isoutlet = True

            if isoutlet:
                global_pressure = np.hstack((global_pressure, proj_pressure[1:]))
                global_velocity = np.hstack((global_velocity, proj_velocity[1:]))
            else:
                global_pressure = np.hstack((global_pressure, proj_pressure))
                global_velocity = np.hstack((global_velocity, proj_velocity))

        global_vector = np.hstack((global_pressure, global_velocity))
        return global_vector

    def load_models(self, training_fdr):
        models = {}
        mins_x = {}
        maxs_x = {}
        mins_y = {}
        maxs_y = {}
        X      = {}
        Y      = {}
        for stencil in self.stencils:
            stencil.load_model(training_fdr,
                               models,
                               mins_x, maxs_x, mins_y, maxs_y, X, Y)

    def evaluate_models(self, vector, prev_vector, dt):
        N = len(self.stencils)
        res = np.zeros((2 * N))
        index = 0
        for stencil in self.stencils:
            # print(index)
            if index == 23:
                    print('hi')
            p, q = stencil.evaluate_model(vector, prev_vector, dt)
            res[index] = res[index] + p
            res[index + N] = res[index + N] + q
            index = index + 1

        return res

    def evaluate_models_jacobian(self, vector, prev_vector, dt, exact = True, otherjac = None):
        N = len(self.stencils)
        jacobian = np.zeros((2 * N, 2 * N))

        if exact:
            indexrow = 0
            for stencil in self.stencils:
                if indexrow == 34 or indexrow == 43:
                    print('hi')
                pjac, qjac, pcols, qcols = stencil.evaluate_model_jacobian(vector, prev_vector, dt)

                jacobian[indexrow, pcols] = jacobian[indexrow, pcols] + pjac
                jacobian[indexrow + N, qcols] = jacobian[indexrow + N, qcols] + qjac
                indexrow = indexrow + 1
        else:
            M = self.evaluate_models(vector, prev_vector, dt)
            for i in range(0, 2 * N):
                e = np.zeros((2 * N))
                eps = 1e-4
                e[i] = eps

                Md = self.evaluate_models(vector + e, prev_vector, dt)
                grad = (Md - M) / eps
                jacobian[:,i] = grad
                r1 = jacobian[np.where(np.abs(jacobian[:,i]) > 0)[0],i]
                r2 = otherjac[np.where(np.abs(otherjac[:,i]) > 0)[0],i]
                if (r1.size != r2.size):
                    print('hello')
                    M = self.evaluate_models(vector, prev_vector, dt)
                else:
                    print('----------')
                    r = np.linalg.norm(r1 - r2)
                    print(r)
                    print(r1)
                    print(r2)
                    print(np.where(np.abs(jacobian[:,i]) > 0)[0])
                    print(np.where(np.abs(otherjac[:,i]) > 0)[0])
                    types = []
                    for st in np.where(np.abs(jacobian[:,i]) > 0)[0]:
                        if st < len(self.stencils):
                            types.append(self.stencils[st].type)
                    print(types)
                    if (r > 0.001):
                        print('hello')
                        M = self.evaluate_models(vector, prev_vector, dt)

        return jacobian

    def find_bcs_indices(self):
        N = len(self.stencils)

        connectivity = self.resampled_geometry.geometry.connectivity

        inlet = 0
        outlets = []
        for j in range(0, connectivity.shape[1]):
            if np.sum(connectivity[:,j]) == -1:
                inlet = j
            if np.sum(connectivity[:,j]) == 1:
                outlets.append(j)
        if len(outlets) == 0:
            outlets.append(0)

        selector = np.zeros((2 * N,1))

        # bc on flowrate
        selector[self.offsets[inlet] + N] = 1
        selector[self.offsets[inlet]] = 1

        for outlet in outlets:
            # bc on pressure
            selector[self.offsets[inlet] + self.npoints[inlet] - 1 + N] = 1
            selector[self.offsets[inlet] + self.npoints[inlet] - 1] = 1

        return selector
