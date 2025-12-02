#!/bin/bash
set -e

NPROC=64 # modify as needed

mkdir -p mdx2
cd mdx2
for SUB in split_{00..16}; do
    mkdir -p $SUB
    cd $SUB
    EXPTFILE="../../dials/${SUB}.expt" # relative to subdir
    mdx2.import_geometry $EXPTFILE
    mdx2.import_data $EXPTFILE --nproc $NPROC 
    mdx2.find_peaks geometry.nxs data.nxs --count_threshold 20 --nproc $NPROC 
    mdx2.mask_peaks geometry.nxs data.nxs peaks.nxs --sigma_cutoff 3 --nproc $NPROC 
    mdx2.integrate geometry.nxs data.nxs --mask mask.nxs --subdivide 3 3 3 --nproc $NPROC 
    cd ..
done