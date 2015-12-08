import numpy as np
import glob
import os.path
import utils

src_data_dir = "/Users/kohb9m/tmp/cdd"
dst_data_dir = "/Users/kohb9m/tmp/cdd_decode/data"
window_size = 15

converted_files = glob.glob('{}/*.smp.npy'.format(dst_data_dir))

for file_name in glob.glob('{}/*.smp'.format(src_data_dir)):
    print ("parsing {} ... ".format(file_name), end="")
    
    file_basename = os.path.basename(file_name)

    if "{}/{}.npy".format(dst_data_dir, file_basename) in converted_files:
        print("found existing file. skipping.")
        continue

    pssm = utils.smp_to_pssm(file_name)

    if pssm.shape[0] < window_size:
        print ("sequence shorter than window size, skipping.")
        continue

    # Window the PSSM into 3d tensors
    pssm_tensor = utils.window(pssm, window_size)

    np.save('{}/{}'.format(dst_data_dir, file_basename), pssm_tensor)

    print("done.".format(file_name))