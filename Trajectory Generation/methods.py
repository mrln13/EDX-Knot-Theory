"""
Core library for billiard-knot trajectory generation and tile placement.

Coordinate convention
---------------------
The visualisation canvas uses a standard image layout (origin top-left,
X right, Y down).  To match the SEM stage convention (origin top-right,
X leftward), all rendering functions flip the X axis at the **tile-index**
level via ``_flip_x_index``.  Both ``place_tiles`` and ``draw_full_path``
share this flip so that painted tiles and the overlaid path are aligned.

Note: ``PosCalc.py`` does NOT flip grid_x because the microscope control
software (Bruker ESPRIT) already uses a top-right origin natively.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import math
import tifffile
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
from scipy.stats import entropy
from collections import Counter

def _flip_x_index(ix: int, nx: int) -> int:
    """Flip a tile index ix to use a top-right origin over nx tiles."""
    return nx - 1 - ix

def _flip_x_pixels(xpx: int, width_px: int) -> int:
    """Flip a pixel x-position to use a top-right origin over width_px pixels."""
    return width_px - 1 - xpx


def adjust_canvas_size(canvas, num_tiles_x, num_tiles_y):
    """Crop *canvas* so its dimensions are exactly divisible by the tile grid.

    Returns (cropped_canvas, tile_width, tile_height).
    """
    tile_width = canvas.shape[1] // num_tiles_x
    tile_height = canvas.shape[0] // num_tiles_y
    adjusted_width = tile_width * num_tiles_x
    adjusted_height = tile_height * num_tiles_y
    adjusted_canvas = canvas[0:adjusted_height, 0:adjusted_width, :]
    return adjusted_canvas, tile_width, tile_height


def find_valid_tile_size(canvas_shape, min_tile_size, max_tile_size, initial_p, initial_q, maximize_reflections=True, search_window=40):
    """Search for a (p, q) pair and tile size that give full coverage.

    Iterates over (p, q) near (initial_p, initial_q) and checks whether the
    resulting grid fits within [min_tile_size, max_tile_size].  When
    *maximize_reflections* is True, higher (p, q) are tried first, yielding
    smaller tiles and a denser billiard path.

    The search covers *search_window* integer values on either side of the
    starting pair: for maximise_reflections=True the range is
    (initial_p + search_window, initial_p]; otherwise
    [initial_p, initial_p + search_window).

    Returns (tile_width, tile_height, num_tiles_x, num_tiles_y, p, q).

    Raises ValueError if no valid configuration is found.
    """
    height, width = canvas_shape

    # Maximizing reflections (high p and q) will tend to smaller tile sizes and less spread of the billiard curves
    if maximize_reflections:
        p_range = range(initial_p + search_window, initial_p, -1)
        q_range = range(initial_q + search_window, initial_q, -1)
    # Minimizing reflections (low p and q)
    else:
        p_range = range(initial_p, initial_p + search_window, 1)
        q_range = range(initial_q, initial_q + search_window, 1)

    print('Finding p and q values!')
    for p in p_range:  # Test nearby p values
        for q in q_range:  # Test nearby q values
            gcd = math.gcd(p, q)
            num_tiles_x = q // gcd
            num_tiles_y = p // gcd
            if not check_full_coverage(p, q, num_tiles_x, num_tiles_y):
                continue
            tile_width = width // num_tiles_x
            tile_height = height // num_tiles_y
            if min_tile_size <= tile_width <= max_tile_size and min_tile_size <= tile_height <= max_tile_size:
                print(f'p: {p}, q: {q}, with tile size (WH) {tile_width} x {tile_height}')
                return tile_width, tile_height, num_tiles_x, num_tiles_y, p, q
    raise ValueError("No valid tile size found within the bounds.")


def billiard_knot(t, p, q):
    """Evaluate the billiard-knot trajectory at normalised time *t*.

    Returns (x, y) in [0, 1] x [0, 1].  The path is parameterised on [0, 2)
    with reflections at the unit-square boundaries; *p* and *q* control the
    number of reflections along the x and y axes respectively.
    """
    x = (p * t) % 2
    y = (q * t) % 2
    # Reflect coordinates
    x = 2 - x if x > 1 else x
    y = 2 - y if y > 1 else y
    return x, y


def lcm(a, b):
    """Compute least common multiple of two integers."""
    return abs(a * b) // math.gcd(a, b)


def check_full_coverage(p, q, num_tiles_x, num_tiles_y):
    """Return True if the (p, q) billiard visits every tile centre exactly once.

    Full coverage requires that both num_tiles_x and num_tiles_y are divisible
    by gcd(p, q).
    """
    gcd_pq = math.gcd(p, q)
    if num_tiles_x % gcd_pq == 0 and num_tiles_y % gcd_pq == 0:
        return True
    else:
        return False


def compute_time_steps(num_tiles_x, num_tiles_y, p, q):
    """Compute normalised times at which the billiard path crosses tile centres.

    Returns a sorted array of unique t values in [0, 1).
    """
    total_steps = lcm(2 * num_tiles_x, 2 * num_tiles_y)
    centers_x = (np.arange(num_tiles_x) + 0.5) / num_tiles_x
    centers_y = (np.arange(num_tiles_y) + 0.5) / num_tiles_y
    time_steps = []
    print('Computing time steps!')
    for t in range(total_steps):
        t_norm = t / total_steps
        x, y = billiard_knot(t_norm, p, q)
        if any(abs(x - centers_x) < 1e-6) and any(abs(y - centers_y) < 1e-6):
            time_steps.append(t_norm)
    return np.unique(np.sort(time_steps))


def compute_entropy(vector):
    """Compute Shannon entropy of a normalized vector."""
    return entropy(vector, base=2)


def update_global_metrics(key_vect, global_metrics):
    """
    Update global variance and entropy metrics.

    Args:
        key_vect: The current normalized composition vector.
        global_metrics: Dictionary tracking global variance and entropy over time.

    Returns:
        Updated global metrics dictionary.
    """
    variance = np.var(key_vect)
    ent = compute_entropy(key_vect)
    global_metrics['variance'].append(variance)
    global_metrics['entropy'].append(ent)
    return global_metrics


def plot_convergence(global_metrics, min_window_size=5, max_window_size=200, roc_threshold=0.001):
    """Plot demo convergence metrics (variance and entropy rate-of-change).

    This is a diagnostic visualisation for the demo convergence helpers and
    is NOT the EWMA-CLR metric used in the manuscript (see Calc_Stats.R).
    """

    steps = len(global_metrics['variance'])
    if steps < 2:
        print("Not enough data points to plot convergence.")
        return
    x = np.arange(steps)

    # Compute rates of change for variance and entropy
    variance_roc = compute_rate_of_change(global_metrics['variance'])
    entropy_roc = compute_rate_of_change(global_metrics['entropy'])

    # Compute window size
    window_sizes = []
    current_window_size = min_window_size
    for i in range(len(variance_roc)):
        window_size = adjust_window_size(
            global_metrics['variance'][max(0, i - current_window_size + 1):i + 1],
            min_window_size, max_window_size
        )
        window_sizes.append(window_size)
        current_window_size = window_size

    # Compute mean ROC using dynamically adjusted window sizes
    mean_variance_roc = [
        np.mean(variance_roc[max(0, i - window_sizes[i] + 1):i + 1]) for i in range(len(variance_roc))
    ]
    mean_entropy_roc = [
        np.mean(entropy_roc[max(0, i - window_sizes[i] + 1):i + 1]) for i in range(len(entropy_roc))
    ]

    # Assign 1 vals for initial ROC
    mean_variance_roc[0] = mean_entropy_roc[0] = variance_roc[0] = entropy_roc[0] = 1


    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot variance and entropy on the same graph with dual y-axes
    ax1 = axs[0]
    ax2 = ax1.twinx()

    ax1.plot(x, global_metrics['variance'], label="Variance", color="blue")
    ax2.plot(x, global_metrics['entropy'], label="Entropy", color="green")

    ax1.set_ylabel("Variance", color="blue")
    ax2.set_ylabel("Entropy", color="green")
    ax1.set_title("Variance and Entropy Over Time")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid()

    # Plot combined rate of change
    axs[1].plot(x[1:], variance_roc, label="ROC (Variance)", color="orange", linestyle="--")
    axs[1].plot(x[1:], entropy_roc, label="ROC (Entropy)", color="red", linestyle="--")
    axs[1].plot(x[1:], mean_variance_roc, label="Mean ROC (Variance, Dynamic)", color="purple", linestyle="-")
    axs[1].plot(x[1:], mean_entropy_roc, label="Mean ROC (Entropy, Dynamic)", color="brown", linestyle="-")
    axs[1].axhline(roc_threshold, color="blue", linestyle="--", label="ROC Threshold")
    axs[1].set_xlabel("Tiles")
    axs[1].set_ylabel("Rate of Change")
    axs[1].set_title("Rate of Change and Dynamic Mean ROC for Variance and Entropy")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_yscale('log')

    plt.tight_layout()
    plt.show()


def compute_rate_of_change(data):
    """
    Compute the rate of change between consecutive data points.

    Args:
        data (list or numpy array): List of variance or entropy values.

    Returns:
        numpy array: Rate of change values.
    """
    data = np.array(data)
    if len(data) < 2:
        return np.array([])  # No rate of change for fewer than two points
    return np.abs(np.diff(data))  # Absolute difference between consecutive points


def adjust_window_size(sliding_variance, min_window, max_window, growth_rate=1.1):
    """
    Adjust the window size dynamically based on the variance trend.

    Args:
        sliding_variance (list): List of variance values over time.
        min_window (int): Minimum allowed window size.
        max_window (int): Maximum allowed window size.
        growth_rate (float): Rate at which the window size grows.

    Returns:
        int: Updated window size.
    """
    if len(sliding_variance) < min_window:
        return min_window  # Start with minimum size
    window_size = int(min_window * (growth_rate ** (len(sliding_variance) // min_window)))
    return min(window_size, max_window)


def monitor_convergence(global_metrics, window_size, rate_of_change_threshold=0.001, min_steps=50):
    """
    Monitor convergence using variance and entropy trends.

    Args:
        global_metrics (dict): Dictionary containing 'variance' and 'entropy' lists.
        window_size (int): Current sliding window size.
        rate_of_change_threshold (float): Threshold for rate of change to detect convergence.
        min_steps (int): Minimum number of steps to ensure sufficient sampling.

    Returns:
        bool: True if convergence is detected, False otherwise.
    """

    if len(global_metrics['variance']) < min_steps:
        return False # Do not attempt to detect convergence too early

    # Calculate rate of change for variance and entropy
    variance_roc = compute_rate_of_change(global_metrics['variance'][-window_size:])
    entropy_roc = compute_rate_of_change(global_metrics['entropy'][-window_size:])

    # Check if both variance and entropy rates of change are below the threshold
    return (variance_roc.mean() < rate_of_change_threshold and
            entropy_roc.mean() < rate_of_change_threshold)


def place_tiles(input, tile_width, tile_height, num_tiles_x, num_tiles_y, p, q, t_vals, data=False, draw_path=False,
                mark_center=False, limit=1.0, overlay=True, min_window_size=5, max_window_size=100,
                rate_of_change_threshold=0.001, min_tiles=50, stop_at_convergence=False):
    """
    Place tiles on the canvas following the billiard knot trajectory.

    Tile indices are X-flipped (via _flip_x_index) so the canvas displays
    with the SEM top-right origin convention.  Pixel coordinates for slicing
    are derived from the flipped tile index, consistent with draw_full_path.

    Parameters
    ----------
    limit : float
        Fraction of the normalised time range [0, 1] to include.
    data : bool
        If True, treat *input* as a segmented phase map and track
        composition convergence (demo variance/entropy, not EWMA-CLR).
    """

    grid_coverage = np.zeros((num_tiles_y, num_tiles_x), dtype=bool)
    norm = Normalize(vmin=0, vmax=limit)
    cmap = plt.cm.viridis
    sm = ScalarMappable(cmap=cmap, norm=norm)
    alpha = 0.6  # transparency for overlay
    canvas = input if data is False else np.zeros_like(input)  # when data=True, keep original in `input` and draw on blank `canvas`
    output = []
    converged = None

    # --- initialize convergence tracking ALWAYS to avoid UnboundLocalError ---
    global_metrics = {'variance': [], 'entropy': []}
    sliding_variance = []
    variance_history = []
    cumulative_vectors = []
    evolution = {}

    if data:
        print('Calculating ground truth composition!')
        input = input.copy()
        ground_truth = {}
        complete = composition(input)
        gt_vector = populate_dict(ground_truth, complete)
        summed = dict.fromkeys(ground_truth, 0)  # running composition
    else:
        gt_vector = None
        summed = None

    count = 0
    print('Processing canvas!')
    for t in t_vals:
        if t > limit:
            continue  # ignore times beyond limit

        x, y = billiard_knot(t, p, q)
        raw_x = int(x * num_tiles_x)
        grid_x = _flip_x_index(raw_x, num_tiles_x)  # top-right origin for tracking
        grid_y = int(y * num_tiles_y)

        if not grid_coverage[grid_y, grid_x]:
            y_min, y_max = grid_y * tile_height, (grid_y + 1) * tile_height
            # Pixel coords: flip at tile-index level, consistent with draw_full_path
            x_min = grid_x * tile_width
            x_max = x_min + tile_width

            if data:
                # tile composition & cumulative vector
                tile_comp = composition(input[y_min:y_max, x_min:x_max])
                tile_dict = dict.fromkeys(ground_truth, 0)
                populate_dict(tile_dict, tile_comp)
                for key, value in tile_dict.items():
                    summed[key] += value
                current_vector = np.asarray(list(summed.values()), dtype=float)
                current_vector = current_vector / current_vector.sum()

                evolution[count] = similarity(gt_vector, current_vector)
                update_global_metrics(current_vector, global_metrics)

                # dynamic window & convergence
                window_size = adjust_window_size(global_metrics['variance'], min_window_size, max_window_size)
                if monitor_convergence(global_metrics, window_size, rate_of_change_threshold, min_steps=min_tiles) and converged is None:
                    converged = count
                    if stop_at_convergence:
                        print(f"Convergence detected at tile {count + 1}. Stopping sampling.")
                        break
                    print(f"Convergence detected at tile {count + 1}.")

                cumulative_vectors.append(current_vector)
                if len(cumulative_vectors) >= window_size:
                    window = np.array(cumulative_vectors[-window_size:])
                    std_dev = np.std(window, axis=0).mean()
                    sliding_variance.append(std_dev)

                variance_history.append(np.var(current_vector))

                if overlay:
                    a = alpha if converged is None else 0.9
                    color = sm.to_rgba(t)[:3]
                    color_array = (np.array(color) * 254).astype(float)
                    tile = input[y_min:y_max, x_min:x_max].astype(float)
                    blended_tile = (1 - a) * tile + a * color_array
                    input[y_min: y_max, x_min: x_max] = blended_tile.astype(np.uint8)

            # paint the visit onto the visualization canvas (always)
            color = sm.to_rgba(t)[:3]
            color = tuple(int(c * 254) for c in color)
            canvas[y_min: y_max, x_min: x_max] = color
            grid_coverage[grid_y, grid_x] = True

        if mark_center:
            center_x = (grid_x + 0.5) * tile_width
            center_y = (grid_y + 0.5) * tile_height
            col = [255, 255, 255] if converged is None else [255, 0, 0]
            size = 20
            canvas[int(center_y) - size: int(center_y) + size, int(center_x) - size: int(center_x) + size] = col

        count += 1

    if draw_path:
        draw_full_path(canvas, p, q, num_tiles_x, num_tiles_y, tile_width, tile_height)

    output.append(canvas)

    # --- only do convergence plots/prints when data=True ---
    if data:
        output.append(input)
        if sliding_variance:
            print("Final sliding window variance: ", sliding_variance[-1])
        if evolution:
            print("Final similarity to ground truth: ", list(evolution.values())[-1])
        if converged is not None:
            print(f"Similarity to ground truth at tile {converged + 1}: ", list(evolution.values())[converged])

        plot_convergence(global_metrics, min_window_size=min_window_size, max_window_size=max_window_size,
                         roc_threshold=rate_of_change_threshold)

    return output, sm


def draw_full_path(canvas, p, q, num_tiles_x, num_tiles_y, tile_width, tile_height, time_steps=100000):
    """Overlay the continuous billiard path onto *canvas*.

    X is flipped at the tile-index level (via _flip_x_index), matching the
    convention used by place_tiles so tiles and path are aligned.
    """
    canvas_w = num_tiles_x * tile_width
    canvas_h = num_tiles_y * tile_height
    for t_i in range(time_steps):
        t_norm = t_i / time_steps
        x, y = billiard_knot(t_norm, p, q)
        # Flip x via tile index, then compute pixel position within that tile
        raw_x = x * num_tiles_x
        tile_ix = _flip_x_index(int(raw_x), num_tiles_x)
        frac_x = raw_x - int(raw_x)  # fractional position within the tile
        px = min(int((tile_ix + frac_x) * tile_width), canvas_w - 1)
        py = min(int(y * canvas_h), canvas_h - 1)
        canvas[max(0, py - 1):py + 1, max(0, px - 1):px + 1] = 255


def visualize(canvas, sm, tile_width, tile_height, p, q):
    """Show the canvas on-screen with a normalised-time colorbar."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(canvas, interpolation='none', origin='upper')
    ax.axis('off')
    ax.set_title(f"Canvas: {canvas.shape[1]}x{canvas.shape[0]}, Tile: {tile_width}x{tile_height}, p={p}, q={q}")
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Normalized Time Step', rotation=270, labelpad=15)
    plt.show()


def composition(eds_map):
    """Extract unique RGB classes and their pixel counts from a segmented phase map.

    Uses uint32 hashing for efficient unique-detection.

    Returns (unique_classes, counts) where unique_classes has shape (N, 3).
    """
    # Promote to uint32 to prevent overflow
    eds_map = eds_map.astype(np.uint32)
    # Encode RGB values into single integers
    hashed_map = eds_map[..., 0] * 256 ** 2 + eds_map[..., 1] * 256 + eds_map[..., 2]

    # Find unique values and counts
    unique_hashes, counts = np.unique(hashed_map, return_counts=True)

    # Decode unique hashes back into RGB triplets
    unique_classes = np.stack([
        (unique_hashes >> 16) & 255,  # Extract red channel
        (unique_hashes >> 8) & 255,  # Extract green channel
        unique_hashes & 255  # Extract blue channel
    ], axis=1)

    return unique_classes, counts


def populate_dict(dict, tile):
    """Update *dict* with (RGB-tuple -> count) entries from *tile* and return
    the normalised composition vector.

    *tile* is the (unique_classes, counts) tuple returned by composition().
    Returns a uniform vector if all counts are zero.
    """
    for i in range(len(tile[0])):
        dict[tuple(tile[0][i])] = tile[1][i]
    vector = []
    for val in dict.values():
        vector.append(val)
    vector = np.asarray(vector, dtype=float)
    total = vector.sum()
    if total == 0:
        return np.ones_like(vector) / len(vector) if len(vector) else vector
    return vector / total


def similarity(v1, v2):
    """Return the cosine similarity between two vectors (0.0 if either is zero)."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return np.dot(v1, v2) / denom
