"""Generate map statistics plots for merged reflection data."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from mdx2.mdx2.command_line.report._functions import load_refl, calc_map_stats
import sys

from mdx2.command_line import make_argument_parser, with_logging, with_parsing

# prevent local mdx2/ directory from shadowing mdx2 package import
if "" in sys.path:
    sys.path.remove("")


def generate_mapstats_figure():
    # read merged data and normalize
    df0, df0h = load_refl("mdx2/merged_all.nxs")
    df1, df1h = load_refl("mdx2/merged_crystal1.nxs")
    df2, df2h = load_refl("mdx2/merged_crystal2.nxs")

    df0_stats = calc_map_stats(df0)
    df0h_stats = calc_map_stats(df0h)
    df1_stats = calc_map_stats(df1)
    df1h_stats = calc_map_stats(df1h)
    df2_stats = calc_map_stats(df2)
    df2h_stats = calc_map_stats(df2h)

    Inorm0 = df0_stats["mean"].max()
    Inorm1 = df1_stats["mean"].max()
    Inorm2 = df2_stats["mean"].max()

    df0_stats /= Inorm0
    df0h_stats /= Inorm0
    df1_stats /= Inorm1
    df1h_stats /= Inorm1
    df2_stats /= Inorm2
    df2h_stats /= Inorm2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, sharey="row", figsize=(6.5, 4.5)
    )

    for df, args in zip(
        [df1_stats, df1h_stats, df2_stats, df2h_stats],
        [
            dict(color="lightgreen"),
            dict(color="lightblue"),
            dict(color="darkgreen", marker="+", linestyle="none"),
            dict(color="darkblue", marker="+", linestyle="none"),
        ],
    ):
        df["mean"].plot(ax=ax2, **args)
        df["standard deviation"].plot(ax=ax4, **args)

    df0_stats["mean"].plot(ax=ax1, color="green")
    df0h_stats["mean"].plot(ax=ax1, color="blue")
    df0_stats["standard deviation"].plot(ax=ax3, color="green")
    df0h_stats["standard deviation"].plot(ax=ax3, color="blue")
    ax3.legend(["non-halo", "halo"])

    ax1.set_title("Crystals 1,2 scaled together")
    ax2.set_title("Crystals 1,2 scaled separately")
    ax1.set_ylabel("Mean intensity\n(normalized)")
    ax3.set_ylabel("Standard deviation of\nintensity (normalized)")
    ax4.legend(["#1, non-halo", "#1, halo", "#2, non-halo", "#2, halo"])
    [ax.set_xlabel("s (Å$^{-1}$)") for ax in (ax3, ax4)]
    plt.tight_layout()

    plt.savefig("figures/fig4a.png", transparent=True)


@dataclass
class Parameters:
    """Options for generating the map statistics figure."""

    # Currently no command-line options; this dataclass exists for consistency.
    pass


def run_mapstats1(params: Parameters):
    """Entry point used by the command-line wrapper."""
    generate_mapstats_figure()


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_mapstats1))


if __name__ == "__main__":
    run()