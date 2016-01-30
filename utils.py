import numpy as np

def window(matrix, ws):
    """
    Scan a matrix returning window size 3d tensors.
    """
    data_list = []
    for i in range(matrix.shape[0] - ws + 1):
        data_list.append(np.expand_dims(matrix[i:i+ws,:], axis=0))

    return np.vstack(data_list)

def window_recurrent(sequence, ws):
    data_list = []
    
    for i in range(0, len(sequence) - ws + 1, 2):
        data_list.append(sequence[i:i+ws])

    return np.vstack(data_list)


def smp_to_pssm(smpfile, full=False, order='ncbistdaa'):
    """
    Convert an smp file to a PSSM matrix.
    """
    f = open(smpfile, 'r')

    content = f.read()
    f.close()

    start = content.find("scores {") + 8
    end = content.find("}", start)

    # Create list of all PSSM elements
    pssm_elements = content[start:end].split(',')

    # Strip spaces and return carriages, convert to integer.
    pssm_column_wise = list(map(int,list(map(str.strip, pssm_elements))))

    # Convert to numpy array. After conversion columns 1,3-20,22 represent the amino acids
    # refer to http://www.ncbi.nlm.nih.gov/IEB/ToolBox/SDKDOCS/BIOSEQ.HTML under
    # `NCBIstdaa: A Simple Sequential Code for Amino Acids` section for full coding
    if order == 'ncbistdaa':
        amino_acid_column_indices = [1,]+list(range(3,21))+[22,]
    else:
        raise Exception("{} ordering not supported".format(order))

    full_pssm_matrix = np.reshape(pssm_column_wise, (-1, 28))

    if full:
        return full_pssm_matrix
    else:
        # retrieve columns that refer to amino acids
        return full_pssm_matrix[:, amino_acid_column_indices]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
