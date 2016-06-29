
import cPickle as pickle
import os
import numpy as np
from astropy.table import Table
from glob import glob

import AnniesLasso as tc


label_names = ("Teff", "logg", "[Fe/H]", )
# Some abundances in the training set are NaNs
#    "[C/H]", "[N/H]", "[O/H]", "[Na/H]", "[Mg/H]", "[Al/H]", "[Si/H]", "[P/H]", 
#    "[S/H]", "[K/H]", "[Ca/H]", "[Ti/H]", "[V/H]", "[Cr/H]", "[Mn/H]", "[Co/H]",
#    "[Ni/H]", "[Cu/H]", "[Ba/H]", "[Sr/H]")
training_set = Table.read("trainingset_param.tab", format="ascii")

training_set_dirname = "APOKASC_trainingset/LRS/"
test_set_dirname = "testset/LRS/"

output_path = "lrs_results.fits"


# Load spectra.
training_disp_path = "lrs_training_disp.pkl"
training_flux_path = "lrs_training_flux.pkl"
training_ivar_path = "lrs_training_ivar.pkl"

if not os.path.exists(training_flux_path) \
or not os.path.exists(training_ivar_path) \
or not os.path.exists(training_disp_path):

    training_flux = []
    training_ivar = []
    N, common_dispersion = len(training_set), None
    for i, star in enumerate(training_set):
        print(i, N)

        filename = os.path.join(
            training_set_dirname, "{}_SNR250.txt".format(star["Starname"]))

        # wavelength, norm_flux, norm_flux_err
        dispersion, normalized_flux, normalized_flux_err = np.loadtxt(filename).T
        if i == 0:
            common_dispersion = dispersion
        else:
            assert np.all(dispersion == common_dispersion), "Common dispersion?"

        # Create ivar
        normalized_ivar = normalized_flux_err**(-2)
        ignore = (normalized_flux < 0) + ~np.isfinite(normalized_ivar)

        normalized_ivar[ignore] = 0
        normalized_flux[ignore] = 1

        training_flux.append(normalized_flux)
        training_ivar.append(normalized_ivar)

    training_flux = np.array(training_flux)
    training_ivar = np.array(training_ivar)

    # Pickle it for faster tests.
    with open(training_disp_path, "wb") as fp:
        pickle.dump(common_dispersion, fp, -1)

    with open(training_flux_path, "wb") as fp:
        pickle.dump(training_flux, fp, -1)

    with open(training_ivar_path, "wb") as fp:
        pickle.dump(training_ivar, fp, -1)

else:
    with open(training_disp_path, "rb") as fp:
        common_dispersion = pickle.load(fp)

    with open(training_flux_path, "rb") as fp:
        training_flux = pickle.load(fp)

    with open(training_ivar_path, "rb") as fp:
        training_ivar = pickle.load(fp)


# Construct and train a model.
model = tc.L1RegularizedCannonModel(training_set, training_flux, training_ivar,
    dispersion=common_dispersion, threads=-1)

model.s2 = 0
model.regularization = 0
model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
    training_set, tc.vectorizer.polynomial.terminator(label_names, 2))

model.train()
model._set_s2_by_hogg_heuristic()

# Test the model.
test_files = glob("{}/star*_SNR*.txt".format(test_set_dirname))
N = len(test_files)
results = []
for i, filename in enumerate(test_files):
    print("Testing {}/{}: {}".format(i, N - 1, filename))

    dispersion, normalized_flux, normalized_flux_err = np.loadtxt(filename).T
    normalized_ivar = normalized_flux_err**(-2)

    # Ignore bad pixels.
    bad = (normalized_flux_err < 0) + (~np.isfinite(normalized_ivar * normalized_flux))
    normalized_ivar[bad] = 0
    normalized_flux[bad] = np.nan

    labels, cov, meta = model.fit(normalized_flux, normalized_ivar, 
        full_output=True)

    # Identify which star it is, etc.
    basename = os.path.basename(filename)
    star = basename.split("_")[0].lstrip("star")
    snr = int(basename.split("_")[1].split(".")[0].lstrip("SNR"))

    err_labels = np.sqrt(np.diag(cov[0]))

    result = dict(zip(label_names, labels[0]))
    result.update(dict(zip(
        ["E_{}".format(label_name) for label_name in label_names], err_labels)))
    result.update({"star": star, "snr": snr})
    print(result)
    results.append(result)

# Collate and save results.
results = Table(rows=results)
results.write(output_path)

