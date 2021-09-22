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
from assembler import Assembler

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
    assembler = Assembler(rgeo, stencil_size, tdata_directory)
    assembler.set_initial_conditions(pressures[0.014], velocities[0.014])
    deltat = 0.014
    assembler.evaluate_residual(assembler.initial_condition,
                                assembler.initial_condition,
                                deltat)

# if __name__ == "__main__":
#     main()

model = 'data/3d_flow_repository/0111_0001_recomputed.vtp'
soln = io.read_geo(model).GetOutput()
soln_array, _, p_array = io.get_all_arrays(soln, 160)
pressures, velocities = io.gather_pressures_velocities(soln_array)
geometry = Geometry(p_array)
rgeo = ResampledGeometry(geometry, 5)
rgeo.assign_area(soln_array['area'])
stencil_size = 5
tdata_directory = 'training_data'
assembler = Assembler(rgeo, stencil_size, tdata_directory)
t0 = 10
dt = 10
assembler.set_initial_conditions(pressures[t0], velocities[t0])
assembler.set_exact_solutions(pressures, velocities)
assembler.solve(t0 = t0, T = 1200, dt = dt)
