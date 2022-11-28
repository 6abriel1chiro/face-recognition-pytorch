import numpy as np
import h5py
def convert_file(input_dir, filename, output_dir):
    filepath = input_dir + '/' + filename
    fin = open(filepath, 'rb')
    binary_data = fin.read()
    new_filepath = output_dir + '/' + filename[:-4] + '.hdf5'
    f = h5py.File(new_filepath)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset = f.create_dataset('binary_data', (100, ), dtype=dt)
    dset[0] = np.fromstring(binary_data, dtype='uint8')


convert_file()