import sys

# add path to core
sys.path.append("core/")

import matplotlib.pyplot as plt
import io_utils as io
import nn_utils as nn
# from assembler import Assembler
from geometry import Geometry
from resampled_geometry import ResampledGeometry
from data_container import DataContainer
from stencil import *

def main():
    model = 'data/3d_flow_repository/0111_0001.vtp'
    soln = io.read_geo(model).GetOutput()
    soln_array, _, p_array = io.get_all_arrays(soln)
    pressures, velocities = io.gather_pressures_velocities(soln_array)
    geometry = Geometry(p_array)
    rgeo = ResampledGeometry(geometry, 10)
    rgeo.assign_area(soln_array['area'])
    stencil_size = 13
    tdata_directory = 'training_data'
    stencils_array = StencilsArray(rgeo, stencil_size)
    # assembler = Assembler(rgeo, stencil_size, tdata_directory)

if __name__ == "__main__":
    main()
