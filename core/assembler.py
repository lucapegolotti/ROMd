import numpy as np
from stencil import *

class Assembler:
    def __init__(self, resampled_geometry, stencil_size, training_data_fdr):
        self.resampled_geometry = resampled_geometry
        self.stencil_size = stencil_size
        self.training_data_fdr = training_data_fdr
        self.stencils_array = StencilsArray(resampled_geometry, stencil_size)
        self.stencils_array.load_models(training_data_fdr)
