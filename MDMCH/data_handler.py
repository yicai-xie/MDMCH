import h5py
import scipy.io as scio


def load_data(path):
    file = h5py.File(path)
    images = file['images'][:].astype('float')
    labels = file['LAll'][:].transpose(1,0)
    tags = file['YAll'][:].transpose(1,0)
    file.close()
    return images, tags, labels

def load_PR_curve_data(path):
    file = h5py.File(path)
    pi2t = file['pi2t'][:].transpose(1, 0)
    ri2t = file['ri2t'][:].transpose(1, 0)
    file.close()
    return pi2t, ri2t

def load_hashcode_data(path):
    file = h5py.File(path)
    hashcodeX = file['hashcodeX'][:].transpose(1, 0)
    hashcodeY = file['hashcodeY'][:].transpose(1, 0)
    file.close()
    return hashcodeX, hashcodeY

def load_W_sematic_distance(path_W_sematic_distance):
    file_W = h5py.File(path_W_sematic_distance,'r')
    W_sematic_distance = file_W['W_sematic_distance'][:].astype('float')
    file_W.close()
    return W_sematic_distance

def load_S_Multi_value(path_S_Multi_value):
    file_S = h5py.File(path_S_Multi_value,'r')
    S_Multi_value = file_S['S_distance'][:].astype('float')
    file_S.close()
    return S_Multi_value

def load_pretrain_model(path):
    return scio.loadmat(path)


if __name__ == '__main__':
    a = {'s': [12, 33, 44],
         's': 0.111}
    import os
    with open('result.txt', 'w') as f:
        for k, v in a.items():
            f.write(k, v)