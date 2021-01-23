import numpy as np

def T_mapping(T_map, expl_r, T_ref):
    mapping = np.zeros(T_map.shape)
    for i in T_map.shape[0]:
        for j in T_map.shape[1]:
            i_prime, j_prime = vicinity_search(i,j,expl_r, T_map.shape, T_ref)
            mapping[i,j] = (i_prime, j_prime)
            
def vicinity_search(i,j,expl_r, shape):
    T_target = T_map[i,j]
    i_prime = i
    j_prime = j
    for round_number in range(expl_r):
        path = get_path(i,j,round_number, shape)
        dist_before = np.inf
        k,l,min_dist = get_best_from_path(path, T_target, shape, T_ref)
        if min_dist<dist_before:
            i_prime = k
            j_prime = l
            dist_before = min_dist
    return i_prime, j_prime

def get_best_from_path(path, T_target, shape, T_ref):
    min_e = np.inf
    for i,j in path:
        j = j % shape[0]
        T_r = T_ref[i_prime, j_prime]
        if np.abs(T_r - T_target)<min_e:
            min_inds = (i_prime, j_prime)
            min_e = np.abs(T_r - T_target)
    return min_inds, min_e

def get_path(i,j,round_number, shape):
    pixels=[(i,j)]
    i,j = step_right(i,j)
    pixels.append((i,j))
    for times in range(2*round_number-1):
        if not i==0:
            i,j = step_up(i,j)
        pixels.append((i,j))
    for times in range(2*round_number):
        i,j = step_left(i,j)
        pixels.append((i,j))
    for times in range(2*round_number):
        if not i==shape[0]-1:
            i,j = step_down(i,j)
        pixels.append((i,j))
    for times in range(2*round_number-1):
        i,j = step_right(i,j)
        pixels.append((i,j))
    return pixels
        
        
def step_right(i,j):
    return i, j+1

def step_left(i,j):
    return i, j-1

def step_up(i,j):
    return i-1, j

def step_down(i,j):
    return i+1, j
