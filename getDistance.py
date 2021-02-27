import numpy as np, numba as nb, os
from tempfile import NamedTemporaryFile
import SharedArray as sa

def getDistance(data, func_name, pool, start=0, allowed_missing=0.0):
    with NamedTemporaryFile(dir='.', prefix='HCC_') as file :
        prefix = 'file://{0}'.format(file.name)
        func = eval(func_name)
        mat_buf = '{0}.mat.sa'.format(prefix)
        mat = sa.create(mat_buf, shape = data.shape, dtype = data.dtype)
        mat[:] = data[:]
        dist_buf = '{0}.dist.sa'.format(prefix)
        dist = sa.create(dist_buf, shape = [mat.shape[0] - start, mat.shape[0], 2], dtype = np.int32)
        dist[:] = 0
        __parallel_dist(mat_buf, func, dist_buf, mat.shape, pool, start, allowed_missing)
        sa.delete(mat_buf)
        sa.delete(dist_buf)
        #os.unlink(dist_buf[7:])
    return dist



def __parallel_dist(mat_buf, func, dist_buf, mat_shape, pool, start=0, allowed_missing=0.0) :
    n_pool = len(pool._pool)
    tot_cmp = (mat_shape[0] * mat_shape[0] - start * start)/n_pool
    s, indices = start, []
    for _ in np.arange(n_pool) :
        e = np.sqrt(s * s + tot_cmp)
        indices.append([s, e])
        s = e
    indices = (np.array(indices)+0.5).astype(int)
    for _ in pool.imap_unordered(__dist_wrapper, [[func, mat_buf, dist_buf, s, e, start, allowed_missing] for s, e in indices ]) :
        pass
    return

def __dist_wrapper(data) :
    func, mat_buf, dist_buf, s, e, start, allowed_missing = data
    mat = sa.attach(mat_buf)
    dist = sa.attach(dist_buf)
    if e > s :
        d = func(mat[:, 1:], s, e, allowed_missing)
        dist[s:e] = d
    del mat, dist

@nb.jit(nopython=True)
def dual_dist(mat, s, e, allowed_missing=0.03):
    dist = np.zeros((e-s, mat.shape[0], 2), dtype=np.int32 )
    n_loci = mat.shape[1]
    for i in range(s, e) :
        ql = np.sum(mat[i] > 0)
        for j in range(i) :
            rl, ad, al = 0., 1e-4, 1e-4
            for k in range(n_loci) :
                if mat[j, k] > 0 :
                    rl += 1
                    if mat[i, k] > 0 :
                        al += 1
                        if mat[i, k] != mat[j, k] :
                            ad += 1
            ll = max(ql, rl) - allowed_missing * n_loci
            ll2 = ql - allowed_missing * n_loci

            if ll2 > al :
                ad += ll2 - al
                al = ll2
            dist[i-s, j, 1] = int(ad/al * n_loci + 0.5)

            if ll > al :
                ad += ll - al
                al = ll
            dist[i-s, j, 0] = int(ad/al * n_loci + 0.5)
    return dist

@nb.jit(nopython=True)
def p_dist(mat, s, e, allowed_missing=0.0):
    dist = np.zeros((e-s, mat.shape[0], 2), dtype=np.int32 )
    n_loci = mat.shape[1]
    for i in range(s, e+1) :
        for j in range(i) :
            ad, al = 0., 0.
            for k in range(n_loci) :
                if mat[j, k] > 0 :
                    if mat[i, k] > 0 :
                        al += 1
                        if mat[i, k] != mat[j, k] :
                            ad += 1
            dist[i-s, j, 0] = int( -np.log(1.-(ad+0.5)/(al+1.0)) * n_loci * 100. + 0.5)
    return dist
