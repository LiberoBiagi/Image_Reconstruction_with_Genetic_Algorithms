# --------- HELPER FUNCTIONS FOR HILL CLIMBING OPTIMIZATION ---------

# --- 1. IMPORTS ---

from PIL import Image, ImageDraw
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import os
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from pathlib import Path
import csv


# --- 2. CORE FUNCTIONS ---
# --- 2.1. Initialization Functions ---

def generate_initial_population(
    pop_size,
    n_triangles,
    img_w,
    img_h,
    target_img=None,
    dx_start=0.55,
    dx_end=0.05,
    area_start=0.04,
    area_end=0.001,
    max_tries=100
):
    """
    Creates the initial GA population of triangle-based images.

    Each individual is a list of triangles (points + RGB color).
    If a target image is provided, triangle placement is guided by error feedback;
    otherwise, triangles are placed randomly with a size schedule.

    Parameters
    ----------
    pop_size : int
        Number of individuals in the population.
    n_triangles : int
        Number of triangles per individual.
    img_w, img_h : int
        Image dimensions.
    target_img : PIL.Image or None
        Optional target image for guided triangle placement.
    dx_start, dx_end : float
        Controls triangle size range over time.
    area_start, area_end : float
        Controls minimum triangle area over time.
    max_tries : int
        Attempts to generate a valid triangle.

    Returns
    -------
    list
        Population of individuals (list of triangle dictionaries).
    """

    pop = []

    # Convert target image to RGB numpy array if it is provided (used for error-based sampling).
    if target_img is None:
        target_arr = None
    else:
        target_arr = np.asarray(target_img.convert("RGB"), dtype=np.uint8)

    # Create each individual of the population.
    for _ in range(pop_size):
        individual = []

        # Start from a white canvas.
        # This canvas is updated after every generated triangle, so the next triangle can be sampled based on the remaining error.
        canvas_img = Image.new("RGB", (img_w, img_h), (255, 255, 255))

        # Generate all triangles for one individual.
        for step in range(n_triangles):

            # Sample triangle based on reconstruction error if target image is given.
            if target_arr is not None:
                points = random_triangle(
                    step=step,
                    max_steps=max(1, n_triangles - 1),
                    canvas_img=canvas_img,
                    picture=target_arr,
                    dx_start=dx_start,
                    dx_end=dx_end,
                    area_start=area_start,
                    area_end=area_end,
                    max_tries=max_tries
                )

            else:
                # Random triangle fallback when no target image is given.
                base_x = np.random.randint(0, img_w)
                base_y = np.random.randint(0, img_h)

                # Compute progress from 0 at the beginning to 1 at the end.
                # This allows triangle sizes to become smaller over time.
                progress = step / max(1, n_triangles - 1)

                # Interpolate the maximum offset of the triangle points.
                # At the beginning triangles can be larger, later they become smaller and more local.
                dx_scale = dx_start * (1 - progress) + dx_end * progress
                area_scale = area_start * (1 - progress) + area_end * progress

                # Convert relative scales to pixel values.
                max_dx = max(1, int(img_w * dx_scale))
                max_dy = max(1, int(img_h * dx_scale))

                # Minimum allowed triangle area.
                # This prevents degenerate or almost invisible triangles.
                min_area = img_w * img_h * area_scale

                points = None

                # Try several times to generate a triangle that is large enough.
                for _ in range(max_tries):
                    candidate = [
                        (base_x, base_y),
                        (
                            int(np.clip(base_x + np.random.randint(-max_dx, max_dx + 1), 0, img_w - 1)),
                            int(np.clip(base_y + np.random.randint(-max_dy, max_dy + 1), 0, img_h - 1))
                        ),
                        (
                            int(np.clip(base_x + np.random.randint(-max_dx, max_dx + 1), 0, img_w - 1)),
                            int(np.clip(base_y + np.random.randint(-max_dy, max_dy + 1), 0, img_h - 1))
                        )
                    ]

                    points = candidate

                    # Accept the triangle if its area is large enough.
                    if triangle_area(points) >= min_area:
                        break

            # Generate a completely random RGB color.
            color = tuple(np.random.randint(0, 256, size=3))

            # Store the triangle as one gene of the individual.
            triangle = {
                "points": points,
                "color": color
            }

            # Add the triangle immediately to the individual.
            individual.append(triangle)

            # Update canvas for next triangle sampling.
            draw_triangle(canvas_img, points, color)

        # Add the completed individual to the population.
        pop.append(individual)

    return pop



def render_and_fitness(individual, target_arr, img_w, img_h):
    """
    Renders an individual and computes RMSE fitness.

    Parameters
    ----------
    individual : list
        List of triangles (dict with 'points' and 'color').
    target_arr : np.ndarray
        Target image array.
    img_w, img_h : int
        Image dimensions.

    Returns
    -------
    PIL.Image
        Rendered image.
    float
        RMSE fitness (lower is better).
    """
    
    # Create blank image.
    canvas = Image.new('RGB', (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas, 'RGB')
    
    # Draw all triangles of the individual onto the canvas.
    for t in individual:
        draw.polygon(t['points'], fill=t['color'])
    
    # Convert to array for comparison.
    gen_arr = np.array(canvas).astype(float)

    # RMSE loss
    rmse = np.sqrt(np.mean((target_arr - gen_arr) ** 2))

    return canvas, rmse


def triangle_area(triangle):
    """
    Computes area of a triangle using the Shoelace formula.

    Parameters
    ----------
    triangle : list
        Three (x, y) points.

    Returns
    -------
    float
        Triangle area.
    """

    (x1, y1), (x2, y2), (x3, y3) = triangle

    return abs(
        x1 * (y2 - y3) +
        x2 * (y3 - y1) +
        x3 * (y1 - y2)
    ) / 2.0


def draw_triangle(img, triangle, color):
    """
    Draws a filled triangle on an image.

    Parameters
    ----------
    img : PIL.Image
    triangle : list
        List of (x, y) points.
    color : tuple
        RGB color.
    """

    draw = ImageDraw.Draw(img)
    draw.polygon(triangle, fill=color)

def random_triangle(
    step,
    max_steps,
    canvas_img,
    picture,
    dx_start,
    dx_end,
    area_start,
    area_end
):
    """
    Generates a triangle biased toward high-error regions of the image.

    Parameters
    ----------
    step : int
        Current step in initialization.
    max_steps : int
        Total number of initialization steps.
    canvas_img : PIL.Image
        Current reconstruction canvas.
    picture : np.ndarray
        Target image.
    dx_start, dx_end : float
        Controls triangle size range over time.
    area_start, area_end : float
        Controls minimum triangle area over time.

    Returns
    -------
    list
        Triangle defined by 3 (x, y) points.
    """

    # Progress from 0 to 1 during initialization.
    progress = step / max(1, max_steps)

    # Triangle size becomes smaller over time.
    dx_scale = np.interp(progress, [0, 1], [dx_start, dx_end])
    area_scale = np.interp(progress, [0, 1], [area_start, area_end])

    img_h, img_w = picture.shape[:2]

    max_dx = max(1, int(img_w * dx_scale))
    max_dy = max(1, int(img_h * dx_scale))

    min_area = img_w * img_h * area_scale

    # Compute the current reconstruction error.
    canvas_arr = np.asarray(canvas_img, dtype=float)
    error_map = np.mean(np.abs(picture.astype(float) - canvas_arr), axis=2)

    # Normalize error values into probabilities.
    probabilities = error_map.flatten()
    probabilities /= probabilities.sum()

    # Sample a base point from high-error regions.
    index = np.random.choice(img_w * img_h, p=probabilities)

    base_y, base_x = divmod(index, img_w)

    best_triangle = None

    # Try several candidate triangles.
    for _ in range(100):
        triangle = [
            (base_x, base_y),

            (
                int(np.clip(
                    base_x + np.random.randint(-max_dx, max_dx + 1),
                    0,
                    img_w - 1
                )),
                int(np.clip(
                    base_y + np.random.randint(-max_dy, max_dy + 1),
                    0,
                    img_h - 1
                ))
            ),

            (
                int(np.clip(
                    base_x + np.random.randint(-max_dx, max_dx + 1),
                    0,
                    img_w - 1
                )),
                int(np.clip(
                    base_y + np.random.randint(-max_dy, max_dy + 1),
                    0,
                    img_h - 1
                ))
            )
        ]

        best_triangle = triangle

        # Stop early if the triangle is large enough.
        if triangle_area(triangle) >= min_area:
            break

    # Always return a triangle.
    return best_triangle
    

# --- 2.2. Optimization Functions ---

def random_neighbor(individual, img_w, img_h, step_geo=10, step_color=25):
    """
    Creates a slightly mutated version of an individual.

    Parameters
    ----------
    individual : list
        List of triangles.
    img_w, img_h : int
        Image dimensions.
    step_geo : int
        Max pixel shift for geometry mutation.
    step_color : int
        Max RGB change for color mutation.

    Returns
    -------
    list
        Mutated individual.
    """

    neighbor = copy.deepcopy(individual)

    # Pick one random triangle to modify.
    tri = random.choice(neighbor)
    mode = random.choice(["geometry", "color"])
 
    if mode == "geometry":
        # Slightly move triangle vertices.
        new_points = []
        for x, y in tri["points"]:
            nx = x + random.randint(-step_geo, step_geo)
            ny = y + random.randint(-step_geo, step_geo)
            nx = max(0, min(img_w - 1, nx))
            ny = max(0, min(img_h - 1, ny))
            new_points.append((nx, ny))
        tri["points"] = new_points
    else:
        # Slight color mutation.
        r, g, b = tri["color"]
        r = int(np.clip(r + random.randint(-step_color, step_color), 0, 255))
        g = int(np.clip(g + random.randint(-step_color, step_color), 0, 255))
        b = int(np.clip(b + random.randint(-step_color, step_color), 0, 255))
        tri["color"] = (r, g, b)
 
    return neighbor



def random_neighbor_multi_tri(individual, img_w, img_h, step_geo=10, step_color=25, n_tri=2):
    """
    Mutates multiple triangles in an individual.

    Parameters
    ----------
    individual : list
        List of triangles.
    img_w, img_h : int
        Image dimensions.
    step_geo : int
        Max geometry shift.
    step_color : int
        Max color change.
    n_tri : int
        Number of triangles to mutate.

    Returns
    -------
    list
        Mutated individual.
    """

    neighbor = copy.deepcopy(individual)

    # Select multiple triangles to modify.
    tris_to_modify = random.sample(neighbor, min(n_tri, len(neighbor)))
 
    for tri in tris_to_modify:
        mode = random.choice(["geometry", "color"])
        if mode == "geometry":
            new_points = []
            for x, y in tri["points"]:
                nx = max(0, min(img_w - 1, x + random.randint(-step_geo, step_geo)))
                ny = max(0, min(img_h - 1, y + random.randint(-step_geo, step_geo)))
                new_points.append((nx, ny))
            tri["points"] = new_points
        else:
            r, g, b = tri["color"]
            r = int(np.clip(r + random.randint(-step_color, step_color), 0, 255))
            g = int(np.clip(g + random.randint(-step_color, step_color), 0, 255))
            b = int(np.clip(b + random.randint(-step_color, step_color), 0, 255))
            tri["color"] = (r, g, b)
 
    return neighbor


def hill_climb_individual(
    individual,
    target_array,
    img_w,
    img_h,
    max_iterations=1000,
    patience=100,
    n_restarts=3,
    n_candidates=5,
    adaptive_step=True,
    use_multi_tri=True,
):
    """
    Hill climbing with:
    - best-of-k neighbors
    - adaptive step
    - random restarts on global best with perturbation
    - multi-triangle moves
    Returns (best_individual, best_score, fitness_history).
    """

    _, init_score = render_and_fitness(individual, target_array, img_w, img_h)
    global_best = copy.deepcopy(individual)
    global_best_score = init_score

    fitness_history = []

    for restart in range(n_restarts):

        # Start from either original or perturbed best.
        if restart == 0:
            current = copy.deepcopy(individual)
            current_score = init_score
        else:
            current = _perturb(global_best, img_w, img_h, strength=restart)
            _, current_score = render_and_fitness(current, target_array, img_w, img_h)

        step_geo   = 15 if not adaptive_step else 20
        step_color = 30 if not adaptive_step else 40
        no_improve = 0

        for iteration in range(max_iterations):

             # Adaptive step size (shrinks when stuck).
            if adaptive_step:
                decay = max(0.3, 1.0 - no_improve / patience)
                cur_step_geo   = max(2, int(step_geo   * decay))
                cur_step_color = max(5, int(step_color * decay))
            else:
                cur_step_geo, cur_step_color = step_geo, step_color

            best_candidate       = None
            best_candidate_score = current_score

            # Try multiple neighbors.
            for _ in range(n_candidates):
                if use_multi_tri and random.random() < 0.3:
                    candidate = random_neighbor_multi_tri(
                        current, img_w, img_h,
                        step_geo=cur_step_geo, step_color=cur_step_color,
                        n_tri=2,
                    )
                else:
                    candidate = random_neighbor(
                        current, img_w, img_h,
                        step_geo=cur_step_geo, step_color=cur_step_color,
                    )

                _, score = render_and_fitness(candidate, target_array, img_w, img_h)

                # Keep the best candidate among the neighbors.
                if score < best_candidate_score:
                    best_candidate       = candidate
                    best_candidate_score = score

            # Accept the best candidate if it's better than current.
            if best_candidate is not None:
                current       = best_candidate
                current_score = best_candidate_score
                no_improve    = 0
                if current_score < global_best_score:
                    global_best       = copy.deepcopy(current)
                    global_best_score = current_score
            else:
                no_improve += 1

            fitness_history.append(global_best_score)

            if no_improve >= patience:
                break

    return global_best, global_best_score, fitness_history
 
 
def _perturb(individual, img_w, img_h, strength=1):
    """
    Strong mutation used for escaping local minima.

    Parameters
    ----------
    individual : list
        Individual to mutate.
    img_w, img_h : int
        Image dimensions.
    strength : int
        Mutation intensity.

    Returns
    -------
    list
        Perturbed individual.
    """

    perturbed = copy.deepcopy(individual)

    # Number of triangles to disturb.
    n = max(1, int(len(perturbed) * 0.2 * strength))
    tris = random.sample(perturbed, min(n, len(perturbed)))
    for tri in tris:
        # Strong geometry shift.
        tri["points"] = [
            (
                max(0, min(img_w - 1, x + random.randint(-40, 40))),
                max(0, min(img_h - 1, y + random.randint(-40, 40))),
            )
            for x, y in tri["points"]
        ]

        # Strong color shift.
        r, g, b = tri["color"]
        tri["color"] = (
            int(np.clip(r + random.randint(-60, 60), 0, 255)),
            int(np.clip(g + random.randint(-60, 60), 0, 255)),
            int(np.clip(b + random.randint(-60, 60), 0, 255)),
        )
    return perturbed
 

def hill_climb_population(
    population,
    target_array,
    img_w,
    img_h,
    max_iterations=1500,
    patience=150,
    top_k=5,
    n_restarts=3,
    n_candidates=5,
):
    """
    Optimizes an entire population using hill climbing.

    Returns top-k individuals and saves fitness statistics.
    """

    results       = []
    all_histories = []

    for i, ind in enumerate(population):
        print(f"  Optimizing individual {i+1}/{len(population)}...")
        best_ind, best_score, history = hill_climb_individual(
            ind,
            target_array,
            img_w,
            img_h,
            max_iterations=max_iterations,
            patience=patience,
            n_restarts=n_restarts,
            n_candidates=n_candidates,
        )
        results.append((best_score, best_ind))
        all_histories.append(history)  # history = list[float], one value per iteration
        print(f"    → score: {best_score:.4f}")

    # --- align histories to the same length (pad with last value) ---
    max_len = max(len(h) for h in all_histories)
    padded  = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]
    arr     = np.array(padded)          # shape: (n_individuals, max_len)

    fitness_mean = arr.mean(axis=0)     # (max_len,)
    fitness_std  = arr.std(axis=0)      # (max_len,)
    iterations   = np.arange(1, max_len + 1)

    # --- save CSV ---
    csv_path = Path("fitness_history.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "fitness_mean", "fitness_std"])
        for it, mean, std in zip(iterations, fitness_mean, fitness_std):
            writer.writerow([it, mean, std])
    print(f"  Fitness history saved to {csv_path}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, fitness_mean, label="mean fitness", color="steelblue", linewidth=2)
    ax.fill_between(
        iterations,
        fitness_mean - fitness_std,
        fitness_mean + fitness_std,
        alpha=0.25,
        color="steelblue",
        label="±1 std",
    )
    # individual traces (faint)
    for h in padded:
        ax.plot(iterations, h, color="gray", alpha=0.15, linewidth=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness score")
    ax.set_title("Fitness over iterations (all individuals)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_path = Path("fitness_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"  Plot saved to {plot_path}")

    # --- top-k selection ---
    results.sort(key=lambda x: x[0])
    top        = results[:top_k]
    top_inds   = [x[1] for x in top]
    top_scores = [x[0] for x in top]
    return top_inds, top_scores


# --- 3. Analytical Functions ---


def render_top_k(top_inds, top_scores, img_w, img_h):
    """
    Displays the best individuals.

    Parameters
    ----------
    top_inds : list
        Best individuals.
    top_scores : list
        Corresponding scores.
    img_w, img_h : int
        Image dimensions.
    """

    k = len(top_inds)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes = [axes]

    for ax, ind, score in zip(axes, top_inds, top_scores):
        ax.set_xlim(0, img_w)
        ax.set_ylim(0, img_h)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('gray')
        ax.set_title(f"score: {score:.1f}", fontsize=9)

        polys, colors = [], []
        for tri in ind:
            pts = np.array(tri["points"], dtype=float)

            # Flip y-axis for display (matplotlib's origin is bottom-left, while our coordinates are top-left).
            pts[:, 1] = img_h - pts[:, 1]
            polys.append(Polygon(pts, closed=True))
            r, g, b = tri["color"]
            colors.append((r/255, g/255, b/255, 0.6))

        ax.add_collection(PatchCollection(polys, facecolor=colors, edgecolor='none'))

    plt.tight_layout()
    plt.show()
    
    
def save_fitness_csv(all_histories, filename="fitness_log.csv"):
    """
    Saves fitness history of individuals to CSV.

    Parameters
    ----------
    all_histories : list of lists
        Fitness values per individual over time.
    filename : str
        Output file name.

    Returns
    -------
    pandas.DataFrame
        Saved dataframe.
    """

    max_len = max(len(h) for h in all_histories)
    rows = []
    for step in range(max_len):
        row = {"iteration": step}
        for i, h in enumerate(all_histories):
            row[f"ind_{i}"] = h[step] if step < len(h) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(os.getcwd(), filename)
    df.to_csv(path, index=False)
    print(f"  → fitness saved to: {path}  ({len(rows)} rows, {len(all_histories)} individuals)")
    return df
    
