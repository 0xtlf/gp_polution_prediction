import os
import time
import typing
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, WhiteKernel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import KMeans
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

# Local-GP approximation. k-means partitions the map; one GP fitted per cluster,
# on its own members plus a halo reaching CLUSTER_OVERLAP * cluster_radius past the edge.
# MAX_POINTS_PER_CLUSTER caps the cluster size
N_CLUSTERS = 85
CLUSTER_OVERLAP = 0.35
MAX_POINTS_PER_CLUSTER = 1500

# Matern(nu=5/2) kernel + Optimiser configuration
LENGTH_SCALE = 0.033
LENGTH_SCALE_BOUNDS = (1e-2, 1e0)
NOISE_LEVEL = 0.01
NOISE_LEVEL_BOUNDS = (1e-3, 1e3)
N_RESTARTS_OPTIMIZER = 5

# Fraction of the fitted noise variance that counts towards the safety margin.
# gp_stddev = uncertainty_f^2 + sigma_n^2
# during residential area prediction on unnoised data, we keep only a NOISE_IN_MARGIN fraction of sigma_n^2
# i.e we upscale our predictions by const * sqrt(max(uncertainty_f^2 - (1 - NOISE_IN_MARGIN) * sigma_n^2, 0))
NOISE_IN_MARGIN = 0.45

# Per-cluster progress logging. Run with `python -u` to stream it.
VERBOSE = True


def _log(message: str):
    if VERBOSE:
        print(f"[gp] {message}", flush=True)


def _build_kernel() -> Kernel:
    """
    Kernel structure: constant * spatial_signal + measurement_noise
    spatial_signal: Matern over RBF, since RBF smooths too much
    measurement_noise: training data is noisy, test data is not.
                    Used to discount noise in safety margin
    """
    return ConstantKernel(
        constant_value=1.0, constant_value_bounds=(1e-3, 1e3)
    ) * Matern(
        length_scale=LENGTH_SCALE, length_scale_bounds=LENGTH_SCALE_BOUNDS, nu=2.5
    ) + WhiteKernel(noise_level=NOISE_LEVEL, noise_level_bounds=NOISE_LEVEL_BOUNDS)


def _optimal_std_offset(
    underpredict_weight: float = COST_W_UNDERPREDICT,
    normal_weight: float = COST_W_NORMAL,
) -> float:
    """
    Standard deviations to shift a prediction upwards to minimize the asymmetric cost.

    For f(x) | D following N(mu, sigma^2), a prediction f_pred = mu + z * sigma, the loss is
        - (f - f_pred)^2 if f_pred > f
        - w(f - f_pred)^2 if f_pred <= f
    Let t = (f - mu)/sigma. Then , the following
        - f_pred > f = t < z
        - f_pred f <= t >= z
    Therefore
        F(z) = E[loss]/sigma^2 = w E[(t-z)^2 1(t≥z)] + E[(t-z)^2 1(t<z)] = wA(z) + B(z)
    One can then show that
        F'(z) = 2z[w(1-Phi(z)) + Phi(z)] - 2(w-1)phi(z)
    where Phi is the standard normal cdf and phi the pdf.
    We find the local minimum of F which is global and unique by convexity
    """

    def stationarity(z: float) -> float:
        return underpredict_weight * (z * norm.sf(z) - norm.pdf(z)) + normal_weight * (
            z * norm.cdf(z) + norm.pdf(z)
        )

    if underpredict_weight <= normal_weight:
        return 0.0
    return brentq(stationarity, 0.0, 20.0)


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict_pollution methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        self.kmeans: typing.Optional[KMeans] = None
        self.cluster_gps: typing.List[typing.Optional[GaussianProcessRegressor]] = []
        # Absolute noise variance per cluster; the fitted value is in normalize_y units.
        self.cluster_noise_var: typing.List[float] = []
        self.fallback_mean = 0.0
        self.fallback_std = 0.0
        self.residential_std_offset = _optimal_std_offset()

    def _training_indices(
        self, cluster_index: int, coordinates: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Points used to fit one cluster's GP: its members plus the surrounding halo."""
        members = np.flatnonzero(labels == cluster_index)
        if members.size == 0:
            return members

        distances = np.linalg.norm(
            coordinates - self.kmeans.cluster_centers_[cluster_index], axis=1
        )
        selected = np.flatnonzero(
            distances <= (1.0 + CLUSTER_OVERLAP) * distances[members].max()
        )

        if selected.size > MAX_POINTS_PER_CLUSTER:
            is_member = np.isin(selected, members)
            own, halo = selected[is_member], selected[~is_member]
            budget = max(MAX_POINTS_PER_CLUSTER - own.size, 0)
            own = (
                own
                if own.size <= MAX_POINTS_PER_CLUSTER
                else self.rng.choice(own, size=MAX_POINTS_PER_CLUSTER, replace=False)
            )
            selected = np.concatenate(
                [own, self.rng.choice(halo, size=budget, replace=False)]
            )

        return np.sort(selected)

    # Don't change the name or the signature of this function
    def fit_model(
        self,
        train_coordinates: np.ndarray,
        train_pollution_targets: np.ndarray,
        train_residential_flags: np.ndarray,
    ):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_pollution_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_residential_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """
        coordinates = np.asarray(train_coordinates, dtype=float)
        targets = np.asarray(train_pollution_targets, dtype=float)

        # Used for any cluster that ends up without enough data to fit a GP.
        self.fallback_mean = float(targets.mean())
        self.fallback_std = float(targets.std())

        # The residential flags do not enter the GP
        started = time.time()
        n_clusters = min(N_CLUSTERS, coordinates.shape[0])
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = self.kmeans.fit_predict(coordinates)

        _log(
            f"fitting {n_clusters} local GPs on {coordinates.shape[0]} points "
            f"(margin {self.residential_std_offset:.4f} stddevs)"
        )

        self.cluster_gps = []  # One GP per cluster
        self.cluster_noise_var = []  # Fitted noise variance, from WhiteKernel noise level
        for cluster_index in range(n_clusters):
            indices = self._training_indices(cluster_index, coordinates, labels)
            if indices.size < 2:
                # Too little data to fit a GP
                self.cluster_gps.append(None)
                self.cluster_noise_var.append(0.0)
                continue

            cluster_started = time.time()
            gp = GaussianProcessRegressor(
                kernel=_build_kernel(),
                normalize_y=True,
                n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
                random_state=0,
            ).fit(coordinates[indices], targets[indices])
            self.cluster_gps.append(gp)

            params = gp.kernel_.get_params()
            noise = float(params["k2__noise_level"])
            self.cluster_noise_var.append(float(noise * targets[indices].std() ** 2))
            _log(
                f"{cluster_index + 1:>4}/{n_clusters:<4}  pts={indices.size:>5}  "
                f"ell={params['k1__k2__length_scale']:.4f}  "
                f"ampl={np.sqrt(params['k1__k1__constant_value']):.3f}  "
                f"noise={noise:>7.4f}  "
                f"logML={gp.log_marginal_likelihood_value_:>8.1f}  "
                f"{time.time() - cluster_started:>5.1f}s"
            )

        _log(
            f"fitted {sum(gp is not None for gp in self.cluster_gps)}/{n_clusters} "
            f"local GPs in {time.time() - started:.1f}s"
        )

    # Don't change the name or the signature of this function
    def predict_pollution(
        self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        if self.kmeans is None:
            raise RuntimeError("predict_pollution called before fit_model")

        coordinates = np.asarray(test_coordinates, dtype=float)
        residential = np.asarray(test_residential_flags).astype(bool)

        gp_mean = np.full(coordinates.shape[0], self.fallback_mean, dtype=float)
        gp_std = np.full(coordinates.shape[0], self.fallback_std, dtype=float)
        noise_var = np.zeros(coordinates.shape[0], dtype=float)

        # Each test point is predicted by the GP of its nearest cluster centroid.
        labels = self.kmeans.predict(coordinates)
        for cluster_index, gp in enumerate(self.cluster_gps):
            batch = labels == cluster_index
            if gp is None or not batch.any():
                continue
            gp_mean[batch], gp_std[batch] = gp.predict(
                coordinates[batch], return_std=True
            )
            noise_var[batch] = self.cluster_noise_var[cluster_index]

        # Outside residential areas the cost is a squared error, so the posterior
        # mean is optimal. Inside residential areas underprediction costs 50x, so shift upwards by
        # the cost-minimizing number of stddevs.
        margin_std = np.sqrt(
            np.maximum(gp_std**2 - (1.0 - NOISE_IN_MARGIN) * noise_var, 0.0)
        )
        predictions = gp_mean.copy()
        # Add the safety margin to residential areas
        predictions[residential] += (
            self.residential_std_offset * margin_std[residential]
        )

        shift = self.residential_std_offset * margin_std[residential]
        _log(
            f"predicted {coordinates.shape[0]} points "
            f"({int(residential.sum())} residential)"
        )
        _log(
            f"mean posterior stddev {gp_std.mean():.3f}  ->  "
            f"mean margin stddev {margin_std.mean():.3f}  "
            f"(ratio {margin_std.mean() / max(gp_std.mean(), 1e-12):.3f})"
        )
        _log(f"mean residential shift {shift.mean() if shift.size else 0.0:.3f}")

        return predictions, gp_mean, gp_std


# You don't have to change this function
def calculate_cost(
    ground_truth: np.ndarray, predictions: np.ndarray, residential_flags: np.ndarray
) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert (
        ground_truth.ndim == 1
        and predictions.ndim == 1
        and ground_truth.shape == predictions.shape
    )

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [
        bool(residential_flag) for residential_flag in residential_flags
    ]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_inside_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0]) ** 2 + (
        coordinate[1] - circle_parameters[1]
    ) ** 2 < circle_parameters[2] ** 2


# You don't have to change this function
def determine_residential_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array(
        [
            [0.5488135, 0.71518937, 0.17167342],
            [0.79915856, 0.46147936, 0.1567626],
            [0.26455561, 0.77423369, 0.10298338],
            [0.6976312, 0.06022547, 0.04015634],
            [0.31542835, 0.36371077, 0.17985623],
            [0.15896958, 0.11037514, 0.07244247],
            [0.82099323, 0.09710128, 0.08136552],
            [0.41426299, 0.0641475, 0.04442035],
            [0.09394051, 0.5759465, 0.08729856],
            [0.84640867, 0.69947928, 0.04568374],
            [0.23789282, 0.934214, 0.04039037],
            [0.82076712, 0.90884372, 0.07434012],
            [0.09961493, 0.94530153, 0.04755969],
            [0.88172021, 0.2724369, 0.04483477],
            [0.9425836, 0.6339977, 0.04979664],
        ]
    )

    residential_flags = np.zeros((grid_coordinates.shape[0],))

    for i, coordinate in enumerate(grid_coordinates):
        residential_flags[i] = any(
            [is_inside_circle(coordinate, circ) for circ in circles]
        )

    return residential_flags


# Don't change the name or the signature of this function
def perform_extended_model_evaluation(model: Model, output_dir: str = "/results"):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print("Performing extended evaluation")

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS)
        / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS)
        / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_residential_flags = determine_residential_flags(visualization_grid)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict_pollution(
        visualization_grid, grid_residential_flags
    )
    predictions = np.reshape(
        predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    )
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title("Extended visualization of task 1")
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, "extended_evaluation.pdf")
    fig.savefig(figure_path)
    print(f"Saved extended evaluation to {figure_path}")

    plt.show()


# You do not need to change the name or the signature of this function; convenience helper used by main()
def get_city_area_data(
    train_x: np.ndarray, test_x: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    # Columns are (lon, lat, residential flag). The CSV header only names two of the
    # three, so the files must be read with skiprows=1 rather than by header name.
    train_coordinates = train_x[:, :2].astype(float)
    train_residential_flags = train_x[:, 2].astype(bool)
    test_coordinates = test_x[:, :2].astype(float)
    test_residential_flags = test_x[:, 2].astype(bool)

    assert (
        train_coordinates.shape[0] == train_residential_flags.shape[0]
        and test_coordinates.shape[0] == test_residential_flags.shape[0]
    )
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_residential_flags.ndim == 1 and test_residential_flags.ndim == 1

    return (
        train_coordinates,
        train_residential_flags,
        test_coordinates,
        test_residential_flags,
    )


# you don't have to change this function
def main():
    # Load the training dataset and test features
    train_x = np.loadtxt("train_x.csv", delimiter=",", skiprows=1)
    train_y = np.loadtxt("train_y.csv", delimiter=",", skiprows=1)
    test_x = np.loadtxt("test_x.csv", delimiter=",", skiprows=1)

    # Extract the city_area information
    (
        train_coordinates,
        train_residential_flags,
        test_coordinates,
        test_residential_flags,
    ) = get_city_area_data(train_x, test_x)

    # Fit the model
    print("Training model")
    model = Model()
    model.fit_model(train_coordinates, train_y, train_residential_flags)

    # Predict on the test features
    print("Predicting on test features")
    predictions = model.predict_pollution(test_coordinates, test_residential_flags)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir=".")


if __name__ == "__main__":
    main()
