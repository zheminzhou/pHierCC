#!/usr/bin/env python

# HierCC.py
# Hierarchical Clustering Complex of MLST allelic profiles
#
# Author: Zhemin Zhou
# Lisence: GPLv3
#
# New assignment: hierCC.py -p <allelic_profile> -o <output_prefix>
# Incremental assignment: hierCC.py -p <allelic_profile> -o <output_prefix> -i <old_cluster_npy>
# Input format:
# ST_id gene1 gene2
# 1 1 1
# 2 1 2
# ...

import sys, gzip, logging, click
import pandas as pd, numpy as np
from multiprocessing import Pool, set_start_method
from scipy.spatial import distance as ssd
from scipy.cluster.hierarchy import linkage
try :
    from getDistance import getDistance
except :
    from .getDistance import getDistance

logging.basicConfig(format='%(asctime)s | %(message)s',stream=sys.stdout, level=logging.INFO)

def prepare_mat(profile_file) :
    mat = pd.read_csv(profile_file, sep='\t', header=None, dtype=str).values
    allele_columns = np.array([i == 0 or (not h.startswith('#')) for i, h in enumerate(mat[0])])
    mat = mat[1:, allele_columns].astype(int)
    mat = mat[mat.T[0]>0]
    return mat

@click.command()
@click.option('-p', '--profile', help='[INPUT; REQUIRED] name of the profile file. Can be GZIPed.',
                        required=True)
@click.option('-o', '--output',
                        help='[OUTPUT; REQUIRED] Prefix for the output files. These include a NUMPY and TEXT verions of the same clustering result',
                        required=True)
@click.option('-a', '--append', help='[INPUT; optional] The NUMPY version of an existing HierCC result',
                        default='')
@click.option('-n', '--n_proc', help='[DEFAULT: 4] Number of processors.', default=4, type=int)
def hierCC(profile, output, append, n_proc):
    '''HierCC takes allelic profile (as in https://pubmlst.org/data/) and
    work out hierarchical clusters of all the profiles based on a minimum-spanning tree.'''
    pool = Pool(n_proc)

    profile_file, cluster_file, old_cluster = profile, output + '.npz', append

    mat = prepare_mat(profile_file)
    n_loci = mat.shape[1] - 1

    logging.info(
        'Loaded in allelic profiles with dimension: {0} and {1}. The first column is assumed to be type id.'.format(
            *mat.shape))
    logging.info('Start hierCC assignments')

    # prepare existing clusters
    if not append:
        absence = np.sum(mat <= 0, 1)
        mat[:] = mat[np.argsort(absence, kind='mergesort')]
        typed = {}
    else :
        od = np.load(old_cluster, allow_pickle=True)
        cls = od['hierCC']
        typed = {c: id for id, c in enumerate(cls.T[0]) if c > 0}
    if len(typed) > 0:
        logging.info('Loaded in {0} old hierCC assignments.'.format(len(typed)))
        mat_idx = np.array([t in typed for t in mat.T[0]])
        mat[:] = np.vstack([mat[mat_idx], mat[(mat_idx) == False]])
        start = np.sum(mat_idx)
    else :
        start = 0

    res = np.repeat(mat.T[0], int(mat.shape[1]) + 1).reshape(mat.shape[0], -1)
    res[res < 0] = np.max(mat.T[0]) + 100
    res.T[0] = mat.T[0]
    logging.info('Calculate distance matrix')
    # prepare existing tree
    with getDistance(mat, 'dual_dist', pool, start) as dist:
        if append :
            for r in res :
                if r[0] in typed :
                    r[:] = cls[typed[r[0]]]
        else :
            dist.dist[:, :, 0] += dist.dist[:, :, 0].T
            logging.info('Start Single linkage clustering')
            slc = linkage(ssd.squareform(dist.dist[:, :, 0]), method='single')

            index = { s:i for i, s in enumerate(mat.T[0]) }
            descendents = [ [m] for m in mat.T[0] ] + [None for _ in np.arange(mat.shape[0]-1)]
            for idx, c in enumerate(slc.astype(int)) :
                n_id = idx + mat.shape[0]
                d = sorted([int(c[0]), int(c[1])], key=lambda x:descendents[x][0])
                min_id = min(descendents[d[0]])
                descendents[n_id] = descendents[d[0]] + descendents[d[1]]
                for tgt in descendents[d[1]] :
                    res[index[tgt], c[2]+1:] = res[index[min_id], c[2]+1:]

        logging.info('Attach genomes onto the tree.')
        for id, (r, d) in enumerate(zip(res[start:], dist.dist[:, :, 1])):
            if id + start > 0 :
                i = np.argmin(d[:id+start])
                min_d = d[i]
                if r[min_d + 1] > res[i, min_d + 1]:
                    r[min_d + 1:] = res[i, min_d + 1:]
    res.T[0] = mat.T[0]
    np.savez_compressed(cluster_file, hierCC=res)

    with gzip.open(output + '.hierCC.gz', 'wt') as fout:
        fout.write('#ST_id\t{0}\n'.format('\t'.join(['HC' + str(id) for id in np.arange(n_loci+1)])))
        for r in res[np.argsort(res.T[0])]:
            fout.write('\t'.join([str(rr) for rr in r]) + '\n')

    logging.info('NUMPY clustering result (for incremental hierCC): {0}.npz'.format(output))
    logging.info('TEXT  clustering result (for visual inspection): {0}.hierCC.gz'.format(output))
    pool.close()

if __name__ == '__main__':
    set_start_method('spawn')
    hierCC(sys.argv[1:])

