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
    model = 'data/3d_flow_ini_1d_quad/0111_0001.vtp'
    soln = io.read_geo(model).GetOutput()
    soln_array, _, p_array = io.get_all_arrays(soln)
    pressures, velocities = io.gather_pressures_velocities(soln_array)
    geometry = Geometry(p_array)
    # geometry.plot()
    rgeo = ResampledGeometry(geometry, 5)
    # # rgeo.plot()
    rgeo.compare_field_along_centerlines(velocities[700])
    # stencil_size = 5
    # data = DataContainer(rgeo, stencil_size, pressures, velocities, soln_array['area'])
    # nn.train_and_save_all_networks(data, stencil_size)
    # geometry.plot(field = soln_array['velocity_0.01400'])
    # rgeo.plot(field = soln_array['velocity_0.01400'])
    plt.show()

if __name__ == "__main__":
    main()
