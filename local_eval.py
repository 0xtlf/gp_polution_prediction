"""
Local train/evaluate harness -- NOT part of the submission and never run by the
checker, which only calls solution.Model.

Holds out a slice of train_x/train_y the same size as the real test set and reports
this task's weighted cost on it.

Usage (use -u so the per-cluster log streams live):
    python -u local_eval.py                # uses N_CLUSTERS from solution.py
    python -u local_eval.py 20 40 60 80    # sweeps those values of N_CLUSTERS

Caveat: train_y carries measurement noise that the real test_y does not, so the cost
here is pessimistic -- roughly 8.5 locally corresponds to 4.9 on the checker. Use it
to rank configurations, not to predict your score.
"""

import sys
import time

import numpy as np
from scipy.stats import norm

import solution
from solution import Model, calculate_cost, get_city_area_data

PUBLIC_BASELINE_COST = 21.487
HELD_OUT_SIZE = 3311  # same size as test_x


def evaluate(n_clusters, coordinates, targets, flags, fit_on, held_out):
    """Fit and score one configuration, returning a row for the summary table."""
    solution.N_CLUSTERS = n_clusters

    started = time.time()
    model = Model()
    model.fit_model(coordinates[fit_on], targets[fit_on], flags[fit_on])
    predictions, gp_mean, gp_std = model.predict_pollution(
        coordinates[held_out], flags[held_out]
    )
    elapsed = time.time() - started

    truth, residential = targets[held_out], flags[held_out]
    cost = calculate_cost(truth, predictions, residential)
    cost_of_mean = calculate_cost(truth, gp_mean, residential)
    rmse = float(np.sqrt(np.mean((truth - gp_mean) ** 2)))

    # Calibration check: with an honest posterior, exactly P(t > z*) of the
    # residential points should be underpredictions. A higher rate means the
    # posterior stddevs are too tight and the safety margin is landing short.
    underpredicted = float(np.mean(predictions[residential] < truth[residential]))
    expected = float(norm.sf(model.residential_std_offset))

    print(
        f"\ncost                      {cost:9.3f}   (public baseline {PUBLIC_BASELINE_COST})"
    )
    print(f"cost of posterior mean    {cost_of_mean:9.3f}   (no safety margin)")
    print(f"posterior-mean RMSE       {rmse:9.3f}")
    print(
        f"residential underpredict  {underpredicted:9.1%}   "
        f"(calibrated would be {expected:.1%})"
    )
    print(f"total time                {elapsed:9.1f}s\n")

    return n_clusters, cost, cost_of_mean, rmse, underpredicted, elapsed


def main():
    cluster_counts = [int(a) for a in sys.argv[1:]] or [solution.N_CLUSTERS]

    train_x = np.loadtxt("train_x.csv", delimiter=",", skiprows=1)
    train_y = np.loadtxt("train_y.csv", delimiter=",", skiprows=1)
    coordinates, flags, _, _ = get_city_area_data(train_x, train_x)

    # Fixed split, so repeated runs stay comparable.
    permutation = np.random.default_rng(1).permutation(coordinates.shape[0])
    held_out, fit_on = permutation[:HELD_OUT_SIZE], permutation[HELD_OUT_SIZE:]
    print(f"fitting on {fit_on.size} points, evaluating on {held_out.size} held out\n")

    results = []
    for n_clusters in cluster_counts:
        print(f"===== N_CLUSTERS = {n_clusters} " + "=" * 40)
        results.append(
            evaluate(n_clusters, coordinates, train_y, flags, fit_on, held_out)
        )

    if len(results) > 1:
        print("===== summary " + "=" * 50)
        print(
            f"{'clusters':>9}  {'cost':>9}  {'mean-only':>9}  {'rmse':>7}  "
            f"{'under':>6}  {'secs':>7}"
        )
        for n_clusters, cost, cost_of_mean, rmse, under, elapsed in results:
            print(
                f"{n_clusters:>9}  {cost:>9.3f}  {cost_of_mean:>9.3f}  {rmse:>7.3f}  "
                f"{under:>5.1%}  {elapsed:>7.1f}"
            )
        best = min(results, key=lambda row: row[1])
        print(f"\nbest: N_CLUSTERS={best[0]} at cost {best[1]:.3f}")


if __name__ == "__main__":
    main()
