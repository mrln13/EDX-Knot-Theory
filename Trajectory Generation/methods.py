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
    """
    Adjust canvas size to ensure it is perfectly divisible by the grid dimensions.
    """
    tile_width = canvas.shape[1] // num_tiles_x
    tile_height = canvas.shape[0] // num_tiles_y
    adjusted_width = tile_width * num_tiles_x
    adjusted_height = tile_height * num_tiles_y
    adjusted_canvas = canvas[0:adjusted_height, 0:adjusted_width, :]
    return adjusted_canvas, tile_width, tile_height


def find_valid_tile_size(canvas_shape, min_tile_size, max_tile_size, initial_p, initial_q, maximize_reflections=True):
    """Find a valid tile size and grid configuration."""
    height, width = canvas_shape

    # Maximizing reflections (high p and q) will tend to smaller tile sizes and less spread of the billiard curves
    if maximize_reflections:
        p_range = range(initial_p + 40, initial_p, -1)
        q_range = range(initial_q + 40, initial_q, -1)
    # Minimizing reflections (low p and q)
    else:
        p_range = range(initial_p, initial_p + 40, 1)
        q_range = range(initial_q, initial_q + 40, 1)

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
    """Generate billiard knot coordinates."""
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
    """
    :param p:
    :param q:
    :param num_tiles_x:
    :param num_tiles_y:
    :return:
    Full coverage is possible when gcd(p,q) > 1, but only if num_tiles_x and num_tiles_y are divisible by gcd(p,q).
    So the interaction between the canvas dimensions and p and q determines whether the trajectory can reach all tile centers.
    This function guarantees p, q combinations that do so, given a number of tiles in the x and y direction.
    """
    gcd_pq = math.gcd(p, q)
    if num_tiles_x % gcd_pq == 0 and num_tiles_y % gcd_pq == 0:
        return True
    else:
        return False


def compute_time_steps(num_tiles_x, num_tiles_y, p, q):
    """
    Compute exact time steps to align billiard knot trajectory with every tile center.
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
    """
    Plot global convergence metrics over time.

    Args:
        global_metrics: Dictionary with keys 'variance' and 'entropy'.
    """

    steps = len(global_metrics['variance'])
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
    Place tiles on the canvas based on the billiard knot trajectory. Limit is fraction of canvas to cover.
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
        grid_x = _flip_x_index(raw_x, num_tiles_x)  # top-right origin
        grid_y = int(y * num_tiles_y)  # top is still origin in Y

        if not grid_coverage[grid_y, grid_x]:
            y_min, y_max = grid_y * tile_height, (grid_y + 1) * tile_height
            x_min, x_max = grid_x * tile_width, (grid_x + 1) * tile_width

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
    for t in range(time_steps):
        t = t / time_steps
        x, y = billiard_knot(t, p, q)
        x_px = int(x * num_tiles_x * tile_width)
        y_px = int(y * num_tiles_y * tile_height)
        grid_x = _flip_x_pixels(x_px, num_tiles_x * tile_width)  # top-right origin
        grid_y = y_px
        canvas[grid_y - 1:grid_y + 1, grid_x - 1:grid_x + 1] = 255


def visualize(canvas, sm, tile_width, tile_height, p, q):
    """
    Visualize the canvas with tiles and tile centers.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(canvas, interpolation='none', origin='upper')
    ax.axis('off')
    ax.set_title(f"Canvas: {canvas.shape[1]}x{canvas.shape[0]}, Tile: {tile_width}x{tile_height}, p={p}, q={q}")
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Normalized Time Step', rotation=270, labelpad=15)
    plt.show()


# Deconstruct 2D array into unique RGB combinations and their frequency
def composition(eds_map):
    """
       Efficiently compute unique classes and their counts in an RGB segmented phase map.

       :param eds_map: RGB segmented phase map (3D numpy array of shape (H, W, 3)).
       :return: unique classes (RGB values) and their counts.
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


# Populate dictionary. Key: tuple with unique RGB values, value: count
def populate_dict(dict, tile):
    """
    Keeps track of measured composition by adding tile compositions
    :param dict:
    :param tile:
    :return: normalized composition
    """
    for i in range(len(tile[0])):
        dict[tuple(tile[0][i])] = tile[1][i]
    vector = []
    for val in dict.values():
        vector.append(val)
    vector = np.asarray(vector)
    vector = vector / sum(vector)
    return vector


def similarity(v1, v2):
    """
    Calculate cosine similarity between vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: cosine similarity (float)
    """
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cosine_sim
