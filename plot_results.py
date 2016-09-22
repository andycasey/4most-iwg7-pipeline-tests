
""" Plot results from 4MOST tests. """

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.ticker import MaxNLocator


def plot_label_precision(optimized_labels, label_names, reference_labels,
    latex_labels=None, legend_labels=None, colors="rk", star_column="star", 
    snr_column="snr", fig=None, pixels_per_angstrom=1.0, common_y_axes=None,
    common_x_limits=None, **kwargs):
    """
    Plot the precision in labels as a function of S/N.

    :param optimized_labels:
        A table containing results for stars at many S/N ratios.

    :parma label_names:
        A tuple containing the label names.

    :param reference_labels: [optional]
        A table containing the expected labels for each star. If `None` is given
        then the set of labels with the highest S/N ratio in `optimized_labels`
        will be used.

    :param latex_labels: [optional]
        A tuple containing LaTeX labels to use on the figure axes.

    :param legend_labels: [optional]
        A list-like containing labels that will appear in the legend.

    :param fig: [optional]
        An optional `matplotlib` figure that has at least as many axes as the
        number of `label_names`.

    :param pixels_per_angstrom: [optional]
        The number of pixels per Angstrom. If this is set to anything other than
        1, the S/N on the x-axis will be shown per Angstrom, not per pixel.

    :param common_y_axes: [optional]
        Specify axes indices that should have the same y-axis limits and
        markers.

    :param common_x_limits: [optional]
        A two-length tuple containing the lower and upper limits to set on all
        x axes.

    :returns:
        A figure showing the precision as a function of S/N for each label.
    """

    if latex_labels is None:
        latex_labels = {
            "Teff": r"$T_{\rm eff}$ $[{\rm K}]$",
            "logg": r"$\log{g}$ $[{\rm dex}]$",
            "[Fe/H]": r"$[{\rm Fe}/{\rm H}]$ $[{\rm dex}]$",
            "[C/H]": r"$[{\rm C}/{\rm H}]$ $[{\rm dex}]$",
            "[N/H]": r"$[{\rm N}/{\rm H}]$ $[{\rm dex}]$",
            "[O/H]": r"$[{\rm O}/{\rm H}]$ $[{\rm dex}]$",
            "[Na/H]": r"$[{\rm Na}/{\rm H}]$ $[{\rm dex}]$",
            "[Mg/H]": r"$[{\rm Mg}/{\rm H}]$ $[{\rm dex}]$",
            "[Al/H]": r"$[{\rm Al}/{\rm H}]$ $[{\rm dex}]$",
            "[Si/H]": r"$[{\rm Si}/{\rm H}]$ $[{\rm dex}]$",
            "[Ca/H]": r"$[{\rm Ca}/{\rm H}]$ $[{\rm dex}]$",
            "[Ti/H]": r"$[{\rm Ti}/{\rm H}]$ $[{\rm dex}]$",
            "[Mn/H]": r"$[{\rm Mn}/{\rm H}]$ $[{\rm dex}]$",
            "[Co/H]": r"$[{\rm Co}/{\rm H}]$ $[{\rm dex}]$",
            "[Ni/H]": r"$[{\rm Ni}/{\rm H}]$ $[{\rm dex}]$",
            "[Ba/H]": r"$[{\rm Ba}/{\rm H}]$ $[{\rm dex}]$",
            "[Sr/H]": r"$[{\rm Sr}/{\rm H}]$ $[{\rm dex}]$",
        }
        latex_labels = [latex_labels.get(ln, ln) for ln in label_names]


    N = len(label_names)
    snr_values = np.sort(np.unique(optimized_labels[snr_column]))

    if not isinstance(reference_labels, (list, tuple)):
        reference_labels = [reference_labels]

    # Construct the dictionary for the label differences.
    label_diff = {}
    for snr in snr_values:
        label_diff[snr] = {}
        for label_name in label_names:
            label_diff[snr][label_name] = {}
            for j in range(len(reference_labels)):
                label_diff[snr][label_name][j] = []

    optimized_labels = optimized_labels.group_by(star_column)
    group_indices = optimized_labels.groups.indices
    for i, si in enumerate(group_indices[:-1]):
        ei = group_indices[i + 1]

        for j in range(len(reference_labels)):

            reference = reference_labels[j]

            # Get the reference labels
            idx = np.where(reference[star_column] \
                == optimized_labels[star_column][si])[0][0]

            for result in optimized_labels[si:ei]:
                for label_name in label_names:
                    try:
                        ref = reference[label_name][idx]

                    except KeyError:
                        ref = np.nan

                    label_diff[result[snr_column]][label_name][j].append(
                        result[label_name] - ref)

    metric = kwargs.pop("metric", np.std)

    if fig is None:
        M = kwargs.pop("M", N)
        fig, _ = plt.subplots(M, 1, figsize=(6, M * 4))

    for i, (ax, label_name, latex_label) \
    in enumerate(zip(np.array(fig.axes).flatten(), label_names, latex_labels)):

        for j, color in enumerate(colors):

            legend_label = None if legend_labels is None else legend_labels[j]

            y = np.nan * np.ones_like(snr_values)
            for k, xk in enumerate(snr_values):
                diffs = label_diff[xk][label_name][j]
                y[k] = metric(diffs)
                

            scale = np.sqrt(pixels_per_angstrom)
            ax.plot(snr_values * scale, y,
                c=color, label=legend_label)
            ax.scatter(snr_values * scale, y, 
                s=100, facecolor=color, zorder=10, alpha=0.75, label=None)

        ax.set_ylabel(latex_label)

        if ax.is_last_row():
            if pixels_per_angstrom != 1:
                ax.set_xlabel(r"$S/N$ $[{\rm \AA}^{-1}]$")
            else:
                ax.set_xlabel(r"$S/N$ $[{\rm pixel}^{-1}]$")
        else:
            ax.set_xticklabels([])

        # Set limits and ticks.
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.yaxis.set_major_locator(MaxNLocator(5))

        ax.set_xticks([0, 10, 20, 30, 40, 50, 100, 200])

        if common_x_limits is not None:
            ax.set_xlim(common_x_limits)


    if legend_labels is not None:
        fig.axes[0].legend(frameon=False)

    if common_y_axes is not None:
        limit = max([ax.get_ylim()[1] for i, ax in enumerate(fig.axes) \
                                                if i in common_y_axes])
        for i, ax in enumerate(fig.axes):
            if i not in common_y_axes: continue
            ax.set_ylim(0, limit)

            ax.axhline(0.1, linestyle=":", c="#666666", zorder=-1)
            ax.axhline(0.2, linestyle="--", c="#666666", zorder=-1)



    fig.tight_layout()
    
    return fig



if __name__ == "__main__":

    import cPickle as pickle

    N_labels = 17

    expected_path = "testset_param.tab"
    lrs_dispersion_path = "lrs_training_disp.pkl"
    hrs_dispersion_path = "hrs_training_disp.pkl"

    lrs_results_path = "lrs_results_{:.0f}L.fits".format(N_labels)
    hrs_results_path = "hrs_results_{:.0f}L.fits".format(N_labels)
    
    output_external_path = "precision-{:.0f}L-external-mr.pdf".format(N_labels)
    output_internal_path = "precision-{:.0f}L-internal-mr.pdf".format(N_labels)

    # 17 labels.
    label_names = ("Teff", "logg", "[Fe/H]", 
        "[C/H]", "[N/H]", "[O/H]", "[Na/H]", "[Mg/H]", "[Al/H]", "[Si/H]",
        "[Ca/H]", "[Ti/H]", "[Mn/H]", "[Co/H]", "[Ni/H]", "[Ba/H]", "[Sr/H]")
    label_names = label_names[:N_labels]


    with open(lrs_dispersion_path, "rb") as fp:
        lrs_disp = pickle.load(fp)

    with open(hrs_dispersion_path, "rb") as fp:
        hrs_disp = pickle.load(fp)


    expected = Table.read(expected_path, format="ascii")
    # Keep columns the same:
    expected["star"] = [each[4:] for each in expected["Starname"]]

    lrs_results = Table.read(lrs_results_path)
    hrs_results = Table.read(hrs_results_path)

    mr = hrs_results["[Fe/H]"] >= -0.5
    lrs_results = lrs_results[mr]
    hrs_results = hrs_results[mr]
    
    fig = plot_label_precision(hrs_results, label_names,
        reference_labels=(expected, ), colors=("#2980b9", ), 
        legend_labels=("HRS (external)", ), fig=None,
        common_x_limits=(0, 250),
        common_y_axes=range(2, 18), pixels_per_angstrom=np.median(1.0/np.diff(hrs_disp)))

    fig = plot_label_precision(lrs_results, label_names,
        reference_labels=(expected, ), colors=("#e67e22", ),
        legend_labels=("LRS (external)", ), fig=fig,
        common_y_axes=range(2, 18), pixels_per_angstrom=np.median(1.0/np.diff(lrs_disp)))

    fig.savefig(output_external_path)


    # Include the highest S/N values as an internal reference value.
    indices = []
    for unique_star in np.unique(hrs_results["star"]):
        match = hrs_results["star"] == unique_star
        index = np.argmax(hrs_results["snr"][match])
        indices.append(np.where(match)[0][index])
    indices = np.array(indices)
    hrs_internal_high_snr = hrs_results[indices]

    indices = []
    for unique_star in np.unique(lrs_results["star"]):
        match = lrs_results["star"] == unique_star
        index = np.argmax(lrs_results["snr"][match])
        indices.append(np.where(match)[0][index])
    indices = np.array(indices)
    lrs_internal_high_snr = lrs_results[indices]

    fig = plot_label_precision(hrs_results, label_names,
        reference_labels=(hrs_internal_high_snr, ), colors=("#2980b9", ), 
        legend_labels=("HRS (internal)", ), fig=None,
        common_x_limits=(0, 250),
        common_y_axes=range(2, 18), pixels_per_angstrom=np.median(1.0/np.diff(hrs_disp)))

    fig = plot_label_precision(lrs_results, label_names,
        reference_labels=(lrs_internal_high_snr, ), colors=("#e67e22", ), 
        legend_labels=("LRS (internal)", ), fig=fig,
        common_x_limits=(0, 250),
        common_y_axes=range(2, 18), pixels_per_angstrom=np.median(1.0/np.diff(lrs_disp)))

    fig.savefig(output_internal_path)

