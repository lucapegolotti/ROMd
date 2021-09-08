import numpy as np
import tensorflow as tf
from keras.models import Model

class Stencil:
    def __init__(self, points, stsize, areas, center):
        self.points = points
        self.areas  = areas
        self.center = center
        self.stsize = stsize

    def set_type_sequential(self, global_range):
        self.type = 'sequential'
        self.global_range = global_range

    def set_type_junction(self, njunctions, global_range_inlet, global_ranges_outlets):
        self.type = 'junction'
        self.njunctions = njunctions
        self.global_range_inlet = global_range_inlet
        self.global_ranges_outlets = global_ranges_outlets

    def load_model(self, fdr, models):
        if self.type == 'sequential':
            name_pressure = fdr + '/S_pressurest' + str(self.stsize) + 'c' + str(self.center)
            name_velocity = fdr + '/S_velocityst' + str(self.stsize) + 'c' + str(self.center)
        if self.type == 'junction':
            name_pressure = fdr + '/J_pressurest' + str(self.stsize) + 'nj' + str(self.njunctions)
            name_velocity = fdr + '/J_velocityst' + str(self.stsize) + 'nj' + str(self.njunctions)
        if name_pressure not in models:
            models[name_pressure] = tf.keras.models.load_model(name_pressure)
        if name_velocity not in models:
            models[name_velocity] = tf.keras.models.load_model(name_velocity)
        model_pressure = models[name_pressure]
        model_velocity = models[name_velocity]

class StencilsArray:
    def __init__(self, resampled_geometry, max_stencil_size):
        self.generate(resampled_geometry, max_stencil_size)

    def generate(self, resampled_geometry, max_stencil_size):
        half = int(np.floor((max_stencil_size - 1) / 2))
        stencils = []

        offset = 0
        nportions = len(resampled_geometry.p_portions)
        connectivity = resampled_geometry.geometry.connectivity

        inlets = []
        outlets = []
        # we look for inlets and outlets
        njuns = connectivity.shape[0]
        for ijun in range(0, njuns):
            inlets.append(np.where(connectivity[ijun,:] == -1)[0][0])
            outlets.append(np.where(connectivity[ijun,:] == 1)[0])

        offsets = [0]
        # compute offset in global vector for every portion
        for ipor in range(0, nportions - 1):
            isoutlet = False
            npoints = resampled_geometry.p_portions[ipor].shape[0]

            for ijun in range(0, njuns):
                if ipor in outlets[ijun]:
                    isoutlet = True

            # if it's an outlet, we don't count the first node (junction)
            if isoutlet:
                npoints = npoints - 1

            offsets.append(offsets[-1] + npoints)

        for ipor in range(0, nportions):
            points = resampled_geometry.p_portions[ipor]
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

            for inode in range(0, npoints):
                if inode == 0 and isoutlet:
                    continue

                # if I am here, then I have to add a bifurcation model
                if inode == npoints - 1 and isinlet:
                    points = resampled_geometry.p_portions[ipor][-half-1:,:]
                    areas = resampled_geometry.areas[ipor][-half-1:]

                    range_inlet = range(points.shape[0]-half-1 + offsets[ipor],
                                        points.shape[0] + offsets[ipor])

                    ranges_outlets = []
                    for outlet in corroutlets:
                        # note that we exclude the first point (junction) for the outlets
                        # as in the data generation
                        points = np.vstack((points, resampled_geometry.p_portions[outlet][1:half+1,:]))
                        curnpoints = resampled_geometry.p_portions[outlet].shape[0]
                        ranges_outlets.append(range(1 + offsets[outlet],
                                                    1 + half + offsets[outlet]))
                        areas = np.hstack((areas, resampled_geometry.areas[outlet][1:half+1]))

                    stencils.append(Stencil(points, max_stencil_size,
                                            areas, half))

                    stencils[-1].set_type_junction(len(corroutlets) + 1,
                                                   range_inlet,
                                                   ranges_outlets)

                # if I am here, then I have to add a sequential model
                else:
                    istart = np.max((0, inode - half))
                    iend = np.min((npoints, inode + half + 1))

                    points = resampled_geometry.p_portions[ipor][istart:iend,:]
                    areas = resampled_geometry.areas[ipor][istart:iend]

                    stencils.append(Stencil(points,
                                            iend - istart,
                                            areas,
                                            inode - istart))
                    stencils[-1].set_type_sequential(range(istart + offsets[ipor], iend + offsets[ipor]))
        self.stencils = stencils

    def load_models(self, training_fdr):
        models = {}
        for stencil in self.stencils:
            print('------')
            print(stencil.center)
            print(stencil.areas)
            stencil.load_model(training_fdr, models)
