import numpy as np

class Assembler:
    def __init__(self, resampled_geometry, stencil_size, training_data_fdr):
        self.resampled_geometry = resampled_geometry
        self.stencil_size       = stencil_size
        self.training_data_fdr  = training_data_fdr

    def create_model_maps(self):
        portions = self.resampled_geometry.p_portions
        nportions = len(portions)
        model_maps
        for ipor in range(0, nportions):
            isinlet = 0
            isoutlet = 0

            aux = np.sum(self.resampled_geometry.geometry.connectivity[:,ipor])

            if aux == -1:
                isinlet = True
            elif aux == 1:
                isoutlet = True

            # treat inlet
            if isinlet:
