
""" Plot results from 4MOST tests. """

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.ticker import MaxNLocator


def plot_label_precision(optimized_labels, label_names, reference_labels,
    latex_labels=None, legend_labels=None, colors="rk", star_column="star", 
    snr_column="snr", fig=None, **kwargs):
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

    :returns:
        A figure showing the precision as a function of S/N for each label.
    """

    if latex_labels is None:
        latex_labels = {
            "Teff": r"$T_{\rm eff}$",
            "logg": r"$\log{g}$",
            "[Fe/H]": r"$[{\rm Fe}/{\rm H}]$"
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
                    label_diff[result[snr_column]][label_name][j].append(
                        result[label_name] - reference[label_name][idx])

    metric = kwargs.pop("metric", np.std)

    if fig is None:
        M = N if N < 4 else (int(np.ceil(N**0.5)), int(np.ceil(N**0.5)))
        fig, _ = plt.subplots(M)

    for i, (ax, label_name, latex_label) \
    in enumerate(zip(np.array(fig.axes).flatten(), label_names, latex_labels)):

        for j, color in enumerate(colors):

            legend_label = None if legend_labels is None else legend_labels[j]

            y = np.nan * np.ones_like(snr_values)
            for k, xk in enumerate(snr_values):
                diffs = label_diff[xk][label_name][j]
                y[k] = metric(diffs)
                
            ax.plot(snr_values, y, c=color, label=legend_label)
            ax.scatter(snr_values, y, s=100, facecolor=color, zorder=10,
                alpha=0.75, label=None)


        ax.set_ylabel(latex_label)

        if ax.is_last_row():
            ax.set_xlabel(r"$S/N$")
        else:
            ax.set_xticklabels([])

        # Set limits and ticks.
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.yaxis.set_major_locator(MaxNLocator(5))


    if legend_labels is not None:
        fig.axes[0].legend(frameon=False)

    fig.tight_layout()
    
    return fig

if __name__ == "__main__":



    expected = Table.read("testset_param.tab", format="ascii")
    # Keep columns the same:
    expected["star"] = [each[4:] for each in expected["Starname"]]

    lrs_results = Table.read("lrs_results.fits")
    hrs_results = Table.read("hrs_results.fits")
    
    
    fig = plot_label_precision(hrs_results, ("Teff", "logg", "[Fe/H]"),
        reference_labels=(expected, ), colors=("#2980b9", ), 
        legend_labels=("HRS (external)", ), fig=None)

    fig = plot_label_precision(lrs_results, ("Teff", "logg", "[Fe/H]"),
        reference_labels=(expected, ), colors=("#e67e22", ),
        legend_labels=("LRS (external)", ), fig=fig)

    fig.savefig("precision-3L-external.pdf")



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

    fig = plot_label_precision(hrs_results, ("Teff", "logg", "[Fe/H]"),
        reference_labels=(hrs_internal_high_snr, ), colors=("#2980b9", ), 
        legend_labels=("HRS (internal)", ), fig=None)

    fig = plot_label_precision(lrs_results, ("Teff", "logg", "[Fe/H]"),
        reference_labels=(lrs_internal_high_snr, ), colors=("#e67e22", ), 
        legend_labels=("LRS (internal)", ), fig=fig)

    fig.savefig("precision-3L-internal.pdf")


