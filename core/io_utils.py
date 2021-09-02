import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

def save_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var):
    try:
        os.mkdir('training_data/')
    except OSError as error:
        print('training_data directory exists')
    directory = 'training_data/' + var + 'st' + str(stencil_size) + 'c' + str(center) + '/'
    os.mkdir(directory)
    np.save(directory + 'X.npy', X)
    np.save(directory + 'Y.npy', Y)
    np.save(directory + 'mins.npy', mins)
    np.save(directory + 'maxs.npy', maxs)
    np.save(directory + 'my.npy', my)
    np.save(directory + 'My.npy', My)
    model.save(directory)

def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res

def collect_points(output):
    return v2n(output.GetData())

def get_all_arrays(geo):
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())
    points = collect_points(geo.GetPoints())
    return point_data, cell_data, points

def read_geo(fname):
    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader

def gather_pressures_velocities(arrays):
    pressures   = {}
    velocities  = {}
    for array in arrays:
        if array[0:8] == 'pressure':
            time = float(array[9:])
            pressures[time] = arrays[array]
        if array[0:8] == 'velocity':
            time = float(array[9:])
            velocities[time] = arrays[array]

    return pressures, velocities
