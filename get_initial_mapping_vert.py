import numpy as np

def T_mapping_vert(T_map, expl_r, T_ref):
    mapping=np.full(T_map.shape, np.nan, dtype='f,f')
    for i in range(T_map.shape[0]):
        for j in range(T_map.shape[1]):
            i_prime, j_prime = vertical_search(i,j,expl_r, T_map.shape, T_ref, T_map)
            mapping[i,j] = (i_prime, j_prime)
    return mapping
            
def vertical_search(i,j,expl_r, shape, T_ref, T_map):
    T_target = T_map[i,j]
    i_prime = i
    j_prime = j
    v_path = get_v_path(i,j,expl_r,shape)
    dist_before = np.inf
    k,l,min_dist = get_best_from_path(v_path, T_target, shape, T_ref)
    if min_dist<dist_before:
        i_prime = k
        j_prime = l
        dist_before = min_dist
    return i_prime, j_prime

def get_v_path(i,j,expl_r,shape):
    v_path=[(i,j)]
    upper_path = get_upper_path(i,j,expl_r,shape)
    lower_path = get_lower_path(i,j,expl_r,shape)
    v_path.extend(upper_path)
    v_path.extend(lower_path)      
    return v_path

def get_upper_path(i,j,expl_r,shape):
    path=[]
    count=0
    while i>0 and count<expl_r:
        i,j = step_up(i,j)
        path.append((i,j))
        count+=1
    return path

def get_lower_path(i,j,expl_r,shape):
    path=[]
    count=0
    while i<shape[0]-1 and count<expl_r:
        i,j = step_down(i,j)
        path.append((i,j))
        count+=1
    return path


def get_best_from_path(path, T_target, shape, T_ref):
    min_e = np.inf
    for i,j in path:
        j = j % shape[1]
        T_r = T_ref[i, j]
        if np.abs(T_r - T_target)<min_e:
            min_inds = (i, j)
            min_e = np.abs(T_r - T_target)
    return min_inds[0], min_inds[1], min_e



def step_up(i,j):
    return i-1, j

def step_down(i,j):
    return i+1, j