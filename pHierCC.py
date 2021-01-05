#!/usr/bin/env python

# pHierCC.py
# pipeline for Hierarchical Clustering of cgMLST
#
# Author: Zhemin Zhou
# Lisence: GPLv3
#
# New assignment: pHierCC -p <allelic_profile> -o <output_prefix>
# Incremental assignment: pHierCC -p <allelic_profile> -o <output_prefix> -i <old_cluster_npz>
# Input format (tab delimited):
# ST_id gene1 gene2
# 1 1 1
# 2 1 2
# ...

import sys, gzip, logging, click
import pandas as pd, numpy as np
from multiprocessing import Pool #, set_start_method
from scipy.spatial import distance as ssd
from scipy.cluster.hierarchy import linkage
try :
    from getDistance import getDistance
except :
    from .getDistance import getDistance

logging.basicConfig(format='%(asctime)s | %(message)s', stream=sys.stdout, level=logging.INFO)

def prepare_mat(profile_file) :
    mat = pd.read_csv(profile_file, sep='\t', header=None, dtype=str).values
    allele_columns = np.array([i == 0 or (not h.startswith('#')) for i, h in enumerate(mat[0])])
    mat = mat[1:, allele_columns].astype(int)
    mat = mat[mat.T[0]>0]
    return mat

@click.command()
@click.option('-p', '--profile', help='[INPUT] name of a profile file consisting of a table of columns of the ST numbers and the allelic numbers, separated by tabs. Can be GZIPped.',
                        required=True)
@click.option('-o', '--output',
                        help='[OUTPUT] Prefix for the output files consisting of a  NUMPY and a TEXT version of the clustering result. ',
                        required=True)
@click.option('-a', '--append', help='[INPUT; optional] The NPZ output of a previous pHierCC run (Default: None). ',
                        default='')
@click.option('-m', '--allowed_missing', help='[INPUT; optional] Allowed proportion of missing genes in pairwise comparisons (Default: 0.03). ',
                        default=0.03, type=float)
@click.option('-n', '--n_proc', help='[INPUT; optional] Number of processes (CPUs) to use (Default: 4).', default=4, type=int)
def phierCC(profile, output, append, n_proc, allowed_missing):
    '''pHierCC takes a file containing allelic profiles (as in https://pubmlst.org/data/) and works
    out hierarchical clusters of the full dataset based on a minimum-spanning tree.'''
    pool = Pool(n_proc)

    profile_file, cluster_file, old_cluster = profile, output + '.npz', append

    mat = prepare_mat(profile_file)
    n_loci = mat.shape[1] - 1

    logging.info(
        'Loaded in allelic profiles with dimension: {0} and {1}. The first column is assumed to be type id.'.format(
            *mat.shape))
    logging.info('Start HierCC assignments')

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
        logging.info('Loaded in {0} old HierCC assignments.'.format(len(typed)))
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
    dist = getDistance(mat, 'dual_dist', pool, start, allowed_missing)
    if append :
        for r in res :
            if r[0] in typed :
                r[:] = cls[typed[r[0]]]
    else :
        dist[:, :, 0] += dist[:, :, 0].T
        logging.info('Start Single linkage clustering')
        slc = linkage(ssd.squareform(dist[:, :, 0]), method='single')

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
    for id, (r, d) in enumerate(zip(res[start:], dist[:, :, 1])):
        if id + start > 0 :
            i = np.argmin(d[:id+start])
            min_d = d[i]
            if r[min_d + 1] > res[i, min_d + 1]:
                r[min_d + 1:] = res[i, min_d + 1:]
    res.T[0] = mat.T[0]
    np.savez_compressed(cluster_file, hierCC=res)

    with gzip.open(output + '.HierCC.gz', 'wt') as fout:
        fout.write('#ST_id\t{0}\n'.format('\t'.join(['HC' + str(id) for id in np.arange(n_loci+1)])))
        for r in res[np.argsort(res.T[0])]:
            fout.write('\t'.join([str(rr) for rr in r]) + '\n')

    logging.info('NPZ  clustering result (for production mode): {0}.npz'.format(output))
    logging.info('TEXT clustering result (for visual inspection and HCCeval): {0}.HierCC.gz'.format(output))
    pool.close()

if __name__ == '__main__':
    phierCC(sys.argv[1:])

