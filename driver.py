import sys

# add path to core
sys.path.append("core/")

import matplotlib.pyplot as plt
import io_utils as io
import nn_utils as nn
from geometry import Geometry
from resampled_geometry import ResampledGeometry
from data_container import DataContainer

def main():
    model = 'data/3d_flow_repository/0111_0001.vtp'
    soln = io.read_geo(model).GetOutput()
    soln_array, a, p_array = io.get_all_arrays(soln)
    pressures, velocities = io.gather_pressures_velocities_areas(soln_array)
    geometry = Geometry(p_array)
    rgeo = ResampledGeometry(geometry, 10)
    stencil_size = 13
    data = DataContainer(rgeo, stencil_size, pressures, velocities, soln_array['area'])
    nn.train_and_save_all_networks(data, stencil_size)
    geometry.plot(field = soln_array['pressure_0.28000'])
    rgeo.plot(field = soln_array['pressure_0.28000'])
    plt.show()

if __name__ == "__main__":
    main()
