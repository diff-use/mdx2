#!/bin/bash
set -e

NPROC=64 # modify as needed

# process background files
cd mdx2
for sub in {1..2}_bkg; do
    mkdir -p $sub
    cd $sub
    EXPTFILE="../../dials/${sub}/imported.expt"
    mdx2.import_data $EXPTFILE --nproc $NPROC
    mdx2.bin_image_series data.nxs 10 20 20 --valid_range 0 200 --outfile binned.nxs --nproc $NPROC
    cd ..
done

# apply the corrections
for sub in split_{00..08}; do
    cd $sub
    mdx2.correct geometry.nxs integrated.nxs --background ../1_bkg/binned.nxs
    cd ..
done

for sub in split_{09..16}; do
    cd $sub
    mdx2.correct geometry.nxs integrated.nxs --background ../2_bkg/binned.nxs
    cd ..
done