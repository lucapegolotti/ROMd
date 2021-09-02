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

def generate_dataset(resampled_geometry, stencil_size, pressures, velocities, areas):
    dataset = []
    for ipor in range(0,len(resampled_geometry.p_portions)):
        times = [ptime for ptime in pressures]
        times.sort()
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
                ar   = areas[inode:inode + stencil_size]
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

    dataset = np.array(dataset)

    mins = []
    maxs = []

    for j in range(0, dataset.shape[1]):
        m = np.min(dataset[:,j])
        M = np.max(dataset[:,j])
        dataset[:,j] = (dataset[:,j] - m) / (M - m)
        mins.append(m)
        maxs.append(M)
    return dataset, mins, maxs

def gather_pressures_velocities_areas(arrays):
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

def train_and_save_all_networks(dataset, max_stencil_size):

    half = int(np.floor((max_stencil_size - 1) / 2))

    for stencil_size in range(half + 1, max_stencil_size + 1):
        # pressure
        if stencil_size != max_stencil_size:
            var = 'pressure'
            center = half
            X, Y, mins, maxs, my, My = dataset.get(stencil_size, center, var)
            model = train_network(X, Y)
            save_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

            var = 'pressure'
            center = stencil_size - half - 1
            X, Y, mins, maxs, my, My = dataset.get(stencil_size, center, var)
            model = train_network(X, Y)
            save_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

            var = 'velocity'
            center = half
            X, Y, mins, maxs, my, My = dataset.get(stencil_size, center, var)
            model = train_network(X, Y)
            save_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

            var = 'velocity'
            center = stencil_size - half - 1
            X, Y, mins, maxs, my, My = dataset.get(stencil_size, center, var)
            model = train_network(X, Y)
            save_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

        else:
            var = 'pressure'
            center = half
            X, Y, mins, maxs, my, My = dataset.get(stencil_size, center, var)
            model = train_network(X, Y)
            save_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)

            var = 'velocity'
            center = half
            X, Y, mins, maxs, my, My = dataset.get(stencil_size, center, var)
            model = train_network(X, Y)
            save_data(X, Y, mins, maxs, my, My, model, stencil_size, center, var)


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

def main():
    # l = os.listdir('data/3d_flow_repository/')

    #%% Loop through models
    # count = 0
    # for model in  :
    #     print(model)
    #     #%% Extract Relevant Data from VTK files
    #     soln = read_geo("data/3d_flow_repository/" +
    #                     model).GetOutput()  # get 3d flow data
    #     soln_array, a, p_array = get_all_arrays(soln)
    #     points = Geometry(p_array)
    #     points.plot(model)
    #     count = count + 1

    model = 'data/3d_flow_repository/0111_0001.vtp'
    soln = read_geo(model).GetOutput()
    soln_array, a, p_array = get_all_arrays(soln)
    pressures, velocities = gather_pressures_velocities_areas(soln_array)
    geometry = Geometry(p_array)
    geometry.plot(field = soln_array['pressure_0.28000'])
    rgeo = ResampledGeometry(geometry, 10)
    stencil_size = 13
    data = generate_dataset(rgeo, stencil_size, pressures,
                            velocities, soln_array['area'])
    rgeo.plot(field = soln_array['pressure_0.28000'])
    plt.show()

# if __name__ == "__main__":
#     main()

model = 'data/3d_flow_repository/0111_0001.vtp'
soln = read_geo(model).GetOutput()
soln_array, a, p_array = get_all_arrays(soln)
pressures, velocities = gather_pressures_velocities_areas(soln_array)
geometry = Geometry(p_array)
rgeo = ResampledGeometry(geometry, 10)
stencil_size = 13
data = Dataset(rgeo, stencil_size, pressures, velocities, soln_array['area'])
train_and_save_all_networks(data, stencil_size)
geometry.plot(field = soln_array['pressure_0.28000'])
rgeo.plot(field = soln_array['pressure_0.28000'])
plt.show()
