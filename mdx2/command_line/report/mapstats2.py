import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mdx2.mdx2.command_line.report1._functions import load_refl, calc_map_stats


def calc_cc(df,*cols):
    s_bin = pd.cut(df.s,np.linspace(0.03,.8,78))
    corr_binned = df.groupby(s_bin)[list(cols)].corr(method='pearson').unstack()
    x = df.groupby(s_bin)['s'].mean().values
    y = corr_binned[cols].values
    return x,y

def calc_cc2(df1,df2,col):
    df12 = df1.merge(df2,on=('h','k','l'),suffixes=('_1','_2'))
    df12.head()
    s_bin = pd.cut(df12.s_1,np.linspace(0.03,.8,78))
    df_tmp = df12.groupby(s_bin)[[col+'_1',col+'_2']].corr(method='pearson').unstack()
    x = df12.groupby(s_bin)['s_1'].mean().values
    y = df_tmp[(col+'_1',col+'_2')].values
    return x,y

def run():

    df0, df0h = load_refl('mdx2/merged_all.nxs')
    df0f, df0fh = load_refl('mdx2/merged_all_Friedel.nxs')
    df1, df1h = load_refl('mdx2/merged_crystal1.nxs')
    df2, df2h = load_refl('mdx2/merged_crystal2.nxs')

    df0_stats = calc_map_stats(df0)
    df0h_stats = calc_map_stats(df0h)
    df1_stats = calc_map_stats(df1)
    df1h_stats = calc_map_stats(df1h)
    df2_stats = calc_map_stats(df2)
    df2h_stats = calc_map_stats(df2h)

    Inorm0 = df0_stats['mean'].max()
    Inorm1 = df1_stats['mean'].max()
    Inorm2 = df2_stats['mean'].max()

    df0_stats /= Inorm0
    df0h_stats /= Inorm0
    df1_stats /= Inorm1
    df1h_stats /= Inorm1
    df2_stats /= Inorm2
    df2h_stats /= Inorm2

    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(4.2, 4.5))

    # add zoom insets
    axin1 = ax1.inset_axes([0.12,0.4,0.5,0.45], transform=ax1.transData)
    axin1.set_ylim(0.99,1.001)
    axin1.set_xlim(0.12,0.62)
    ax1.indicate_inset_zoom(axin1)

    # add zoom insets
    axin2 = ax2.inset_axes([0.12,0.1,0.4,0.75], transform=ax2.transData)
    axin2.set_ylim(0.95,1.005)
    axin2.set_xlim(0.12,0.52)
    ax2.indicate_inset_zoom(axin2)

    x,y = calc_cc(df0h,'group_0_intensity','group_1_intensity')
    ax1.plot(x,y,label='CC$_{1/2}$')
    axin1.plot(x,y)

    x,y = calc_cc(df0,'group_0_intensity','group_1_intensity')
    ax2.plot(x,y)
    axin2.plot(x,y)

    x,y = calc_cc2(df1h,df2h,'intensity')
    ax1.plot(x,y,label='CC$_{rep.}$')
    axin1.plot(x,y)

    x,y = calc_cc2(df1,df2,'intensity')
    ax2.plot(x,y)
    axin2.plot(x,y)

    x,y = calc_cc(df0fh,'group_0_intensity','group_1_intensity')
    ax1.plot(x,y,label='CC$_{Friedel}$')
    axin1.plot(x,y)

    x,y = calc_cc(df0f,'group_0_intensity','group_1_intensity')
    ax2.plot(x,y)
    axin2.plot(x,y)

    ax2.set_xlabel('s (Å$^{-1}$)')
    [ax.set_ylabel('Correlation coefficient') for ax in [ax1,ax2]]

    ax1.set_ylim(0,1.05)
    ax2.set_ylim(0,1.05)

    axin1.xaxis.set_ticklabels([])
    axin2.xaxis.set_ticklabels([])

    ax1.legend(ncol=3)

    plt.tight_layout()

    plt.savefig('figures/fig4b.png',transparent=True)


if __name__ == "__main__":
    run()