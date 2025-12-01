# Multi-crystal example

This example reproduces the data processing and analysis described here:

> Meisburger SP & Ando N. Scaling and merging macromolecular diffuse scattering with *mdx2*. Acta Cryst. D**80**, 299-313. [DOI](https://doi.org/10.1107/S2059798324002705)

## Download the raw data

First, download the raw diffraction images from Zenodo (doi:10.5281/zenodo.10515006) and extract the `*.tgz` files.

## Data processing

Data processing is orchestrated by chaining together *DIALS* and *mdx2* command-line programs using the Bourne Again Shell (BASH) scripting language. The scripts `1-*.sh` through `6-*.sh` should be executed sequentially. Scripts `1-dials_all.sh` and `1-dials_background.sh` require a `dials` installation. The `DATADIR` variable appearing in these scripts must be modified before running. The remaining scripts should be run after activating the `mdx2` environment.  If fewer than 64 cores are avilable, the parameter `--nproc` should be reduced. All scripts are run from the same base processing directory, and they produce the following output directory structure containing ~54 Gb of processed data:

```
. (project root)
├── dials
│   ├── 1_1
│   ├── 1_2
│   ├── 1_3
│   ├── 1_4
│   ├── 1_5
│   ├── 1_6
│   ├── 1_7
│   ├── 1_8
│   ├── 1_9
│   ├── 1_bkg
│   ├── 2_1
│   ├── 2_2
│   ├── 2_3
│   ├── 2_4
│   ├── 2_5
│   ├── 2_6
│   ├── 2_7
│   ├── 2_8
│   └── 2_bkg
└── mdx2
    ├── 1_bkg
    ├── 2_bkg
    ├── partial_merge
    ├── split_00
    ├── split_01
    ├── split_02
    ├── split_03
    ├── split_04
    ├── split_05
    ├── split_06
    ├── split_07
    ├── split_08
    ├── split_09
    ├── split_10
    ├── split_11
    ├── split_12
    ├── split_13
    ├── split_14
    ├── split_15
    └── split_16
```

## Data analysis

Statistical analysis of scattering and visualization tasks are performed Python using the *mdx2* package and standard tools such as *pandas* and *numpy*. The included Jupyter notebooks reproduce all pre-processing and plotting steps to generate the figures in Meisburger & Ando 2024.

- Figure 2 -- [dials_scale_factors.ipynb](dials_scale_factors.ipynb)
- Figure 3 -- [scaling_model.ipynb](scaling_model.ipynb)
- Figure 4 -- [map_statistics.ipynb](map_statistics.ipynb)
- Figure 5 -- [visualization.ipynb](visualization.ipynb)
- Figure S1 -- [scale_factor_mdx2_vs_dials.ipynb](scale_factor_mdx2_vs_dials.ipynb)
- Figure S2 -- [cchalf_vs_redundancy.ipynb](cchalf_vs_redundancy.ipynb)
