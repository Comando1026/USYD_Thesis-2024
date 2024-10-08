import os
import re
import gc
import cProfile
import pstats
import numpy as np
from skimage import io
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from skimage.morphology import skeletonize, remove_small_objects, area_closing, label
from skimage.morphology import closing, disk
from skimage.filters import gaussian
from skimage import img_as_ubyte
import plotly.graph_objects as go
from collections import deque
from tqdm import tqdm  # Import tqdm for progress bars
import csv
import pandas as pd
from io import StringIO
from numba import njit  # Speed up functions
from joblib import Parallel, delayed  # For parallel processing


# Function for midline creation (skeletonization)
def midline_creation(image):
    bool_image = image.astype(bool)
    skeleton = skeletonize(bool_image)
    midline_image = skeleton.astype(np.uint8) * 255
    return midline_image

# Function for processing images (parallelized)
def processing(image):
    processed_image = area_closing(image, area_threshold=64, connectivity=2)
    processed_image = remove_small_objects(image.astype(bool), min_size=20, connectivity=2)
    return processed_image.astype(np.uint8)

# Function to visualize images
def visualize_image(image, title, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to detect end points in skeleton
def end_points(image):
    endpoints = np.zeros_like(image)
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            if image[x, y] == 1:
                neighbourhood = image[x - 1:x + 2, y - 1:y + 2]
                if np.sum(neighbourhood) - image[x, y] == 1:
                    endpoints[x, y] = 1
    return endpoints

# Function to detect branch points in skeleton
def branch_points(image):
    branch_points = np.zeros_like(image)
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            if image[x, y] == 1:
                neighbourhood = image[x - 1:x + 2, y - 1:y + 2]
                if np.sum(neighbourhood) - image[x, y] > 2:
                    branch_points[x, y] = 1
    return branch_points

# Function to get neighboring pixels
def get_neighbors(pixel, skeleton):
    x, y = pixel
    neighbors = [
        (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
        (x, y - 1),           (x, y + 1),
        (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
    ]
    valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < skeleton.shape[0] and 0 <= ny < skeleton.shape[1] and skeleton[nx, ny] == 1]
    return valid_neighbors

# Function for BFS pathfinding on the skeleton
def bfs_path(skeleton, start, end):
    queue = deque([([start], start)])
    visited = set()
    visited.add(start)

    while queue:
        path, current = queue.popleft()
        if current == end:
            return path

        for neighbor in get_neighbors(current, skeleton):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((path + [neighbor], neighbor))

    return []

# Preprocess the image to clean it before skeletonization
def preprocess_image(image, blur_radius=1.0, morph_size=3):
    blurred_image = gaussian(image, sigma=blur_radius)
    binary_image = blurred_image > 0.5  # Adjust the threshold if needed
    smoothed_image = closing(binary_image, disk(morph_size))
    processed_image = img_as_ubyte(smoothed_image)
    return processed_image

# Function to calculate angle between two vectors
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norms_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms_product == 0:
        return 0.0
    cos_angle = np.clip(dot_product / norms_product, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

# Function to detect curve points with significant angle changes
def detect_curve_points(path, angle_threshold=45):
    curve_points = []
    for i in range(1, len(path) - 1):
        prev_vector = np.array(path[i]) - np.array(path[i - 1])
        next_vector = np.array(path[i + 1]) - np.array(path[i])
        angle = angle_between_vectors(prev_vector, next_vector)
        if angle > angle_threshold:
            curve_points.append(path[i])
    return curve_points

# Function to calculate direction vector
def calculate_direction_vector(endpoint, path, distance_threshold):
    for point in path:
        dist = np.linalg.norm(np.array(endpoint) - np.array(point))
        if dist >= distance_threshold:
            direction_vector = np.array(endpoint) - np.array(point)
            return direction_vector / np.linalg.norm(direction_vector)
    return None

# Convert skeleton to shapely LineStrings
def skeleton_to_lines(skeleton):
    lines = []
    labeled_skeleton, _ = label(skeleton, return_num=True)
    for region in regionprops(labeled_skeleton):
        coords = region.coords
        if len(coords) > 1:
            line = LineString(coords)
            lines.append(line)
    return lines

# Function to explode roads into labeled branches
def explode_roads(image, branch_points):
    exploded_image = image.copy()
    exploded_image[branch_points == 1] = 0
    labeled_branches = label(exploded_image)
    return labeled_branches

# Function to find intersections with the frame boundary
def find_intersection_with_frame(endpoint, direction_vector, frame_width, frame_height, skeleton_lines, other_rays):
    x, y = endpoint
    dx, dy = direction_vector

    if dx != 0:
        t_right = (frame_width - 1 - x) / dx
        t_left = -x / dx
    else:
        t_right = t_left = float('inf')

    if dy != 0:
        t_top = -y / dy
        t_bottom = (frame_height - 1 - y) / dy
    else:
        t_top = t_bottom = float('inf')

    t = min([t for t in [t_right, t_left, t_top, t_bottom] if t > 0])

    frame_intersection = Point(x + t * dx, y + t * dy)
    ray_line = LineString([endpoint, frame_intersection])

    closest_intersection = frame_intersection
    min_distance = ray_line.length

    def is_in_direction(point):
        vector_to_point = np.array([point.x - x, point.y - y])
        return np.dot(vector_to_point, np.array([dx, dy])) > 0

    for skeleton_line in skeleton_lines:
        intersection = ray_line.intersection(skeleton_line)
        if not intersection.is_empty and isinstance(intersection, Point) and is_in_direction(intersection):
            dist = ray_line.project(intersection)
            if dist < min_distance:
                min_distance = dist
                closest_intersection = intersection

    for other_ray in other_rays:
        intersection = ray_line.intersection(other_ray)
        if not intersection.is_empty and isinstance(intersection, Point) and is_in_direction(intersection):
            dist = ray_line.project(intersection)
            if dist < min_distance:
                min_distance = dist
                closest_intersection = intersection

    return closest_intersection.x, closest_intersection.y

# Modified vector_extension function to handle branch points, end points, and detect curve points
def vector_extension(branches, endpoints, branch_points, exclusion_radius=5, angle_threshold=45):
    branch_vectors = []
    branch_points_list = []
    end_vectors = []
    end_points_list = []
    curve_points_list = []

    ends = end_points(branches)
    ends = np.where(ends)
    branch_coords = np.column_stack(np.where(branch_points == 1))

    # Process branch points
    if len(branch_coords) > 0:
        for branch_coord in branch_coords:
            start = (branch_coord[0], branch_coord[1])
            for j in range(len(ends[0])):
                end = (ends[0][j], ends[1][j])
                path = bfs_path(branches, start, end)
                if path:
                    curve_points = detect_curve_points(path, angle_threshold)
                    curve_points_list.extend(curve_points)
                    v = calculate_direction_vector(start, path, 4)
                    if v is not None:
                        branch_points_list.append(start)
                        branch_vectors.append(v)

    # Process end points
    if len(ends[1]) > 1:
        for i in range(len(ends[0])):
            start = (ends[0][i], ends[1][i])
            for j in range(i + 1, len(ends[0])):
                end = (ends[0][j], ends[1][j])
                path = bfs_path(branches, start, end)
                if path:
                    curve_points = detect_curve_points(path, angle_threshold)
                    curve_points_list.extend(curve_points)
                    v = calculate_direction_vector(start, path, 4)
                    if v is not None:
                        end_points_list.append(start)
                        end_vectors.append(v)

    return branch_points_list, branch_vectors, end_points_list, end_vectors, curve_points_list

# Main function that processes the image, extracts points, and saves results
def main(image, filename, output_path, tile_x, tile_y, tile_size=150, angle_threshold=45):
    frame_width = image.shape[1]
    frame_height = image.shape[0]

    smoothed_image = preprocess_image(image)

    midline_image = midline_creation(smoothed_image)
    
    pro_image = processing(midline_image)
    
    endpoints = end_points(pro_image)

    bpoints = branch_points(pro_image)

    branches = explode_roads(pro_image, bpoints)
    skeleton_lines = skeleton_to_lines(pro_image)
    other_rays = []

    branch_vectors = []
    branch_points_list = []
    end_vectors = []
    end_points_list = []
    curve_points_list = []

    for region in regionprops(branches):
        branch = branches == region.label
        result = vector_extension(branch, endpoints, bpoints, angle_threshold=angle_threshold)
        if result is not None:
            branch_points_list += result[0]
            branch_vectors += result[1]
            end_points_list += result[2]
            end_vectors += result[3]
            curve_points_list += result[4]

    global_branch_points = np.zeros((len(branch_points_list), 2))
    global_branch_vectors = np.zeros((len(branch_vectors), 2))
    global_end_points = np.zeros((len(end_points_list), 2))
    global_end_vectors = np.zeros((len(end_vectors), 2))
    global_intersections = np.zeros((len(branch_points_list) + len(end_points_list), 2))
    global_curve_points = np.zeros((len(curve_points_list), 2))

    # Process branch points and their intersections
    for i in range(len(branch_points_list)):
        global_branch_points[i] = (branch_points_list[i][0] + tile_y * tile_size, branch_points_list[i][1] + tile_x * tile_size)
        intersection = find_intersection_with_frame(branch_points_list[i], branch_vectors[i], frame_width, frame_height, skeleton_lines, other_rays)
        global_intersections[i] = (intersection[1] + tile_x * tile_size, intersection[0] + tile_y * tile_size)  # Fixed flipped coordinates
        global_branch_vectors[i] = branch_vectors[i]

    # Process end points and their intersections
    for i in range(len(end_points_list)):
        global_end_points[i] = (end_points_list[i][0] + tile_y * tile_size, end_points_list[i][1] + tile_x * tile_size)
        intersection = find_intersection_with_frame(end_points_list[i], end_vectors[i], frame_width, frame_height, skeleton_lines, other_rays)
        global_intersections[len(branch_points_list) + i] = (intersection[1] + tile_x * tile_size, intersection[0] + tile_y * tile_size)

    # Process curve points
    for i in range(len(curve_points_list)):
        global_curve_points[i] = (curve_points_list[i][0] + tile_y * tile_size, curve_points_list[i][1] + tile_x * tile_size)

    return global_branch_points, global_branch_vectors, global_end_points, global_end_vectors, global_intersections, global_curve_points

# Function to export the results to CSV
def export_to_csv(output_path, branch_points, branch_vectors, end_points, end_vectors, intersections, curve_points, filename):
    csv_file = os.path.join(output_path, f"{filename}_coordinates.csv")
    
    headers = ['Branch_Point_X', 'Branch_Point_Y', 'Branch_Vector_X', 'Branch_Vector_Y',
               'End_Point_X', 'End_Point_Y', 'End_Vector_X', 'End_Vector_Y',
               'Intersection_X', 'Intersection_Y', 'Curve_Point_X', 'Curve_Point_Y']
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        max_len = max(len(branch_points), len(end_points), len(curve_points))
        for i in range(max_len):
            branch_data = [
                branch_points[i][1] if i < len(branch_points) else '',
                -branch_points[i][0] if i < len(branch_points) else '',
                branch_vectors[i][1] if i < len(branch_vectors) else '',
                -branch_vectors[i][0] if i < len(branch_vectors) else '',
            ]
            end_data = [
                end_points[i][1] if i < len(end_points) else '',
                -end_points[i][0] if i < len(end_points) else '',
                end_vectors[i][1] if i < len(end_vectors) else '',
                -end_vectors[i][0] if i < len(end_vectors) else '',
            ]
            intersection_data = [
                intersections[i][0] if i < len(intersections) else '',
                -intersections[i][1] if i < len(intersections) else '',
            ]
            curve_data = [
                curve_points[i][1] if i < len(curve_points) else '',
                -curve_points[i][0] if i < len(curve_points) else '',
            ]
            
            writer.writerow(branch_data + end_data + intersection_data + curve_data)
    
    print(f"Data exported to {csv_file}")

# Save progression images with branch points, end points, and curve points
def save_progression_images(image, smoothed_image, midline_image, pro_image, branch_points_image, end_points_image, 
                            branch_vectors, branch_points_list, curve_points, filename, output_folder):
    fig, axs = plt.subplots(1, 6, figsize=(24, 5))
    
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(smoothed_image, cmap='gray')
    axs[1].set_title('Smoothed Image')
    axs[1].axis('off')

    axs[2].imshow(midline_image, cmap='gray')
    axs[2].set_title('Midline Image')
    axs[2].axis('off')

    axs[3].imshow(pro_image, cmap='gray')
    axs[3].scatter(np.where(branch_points_image)[1], np.where(branch_points_image)[0], color='red', s=10, label="Branch Points")
    axs[3].set_title('Branch Points')
    axs[3].axis('off')

    axs[4].imshow(pro_image, cmap='gray')
    axs[4].scatter(np.where(end_points_image)[1], np.where(end_points_image)[0], color='blue', s=10, label="End Points")
    axs[4].set_title('End Points')
    axs[4].axis('off')

    axs[5].imshow(pro_image, cmap='gray')
    curve_points_array = np.array(curve_points)
    axs[5].scatter(curve_points_array[:,1], curve_points_array[:,0], color='yellow', s=15, label="Curve Points")
    axs[5].set_title('Curve Points')
    axs[5].axis('off')

    output_file = os.path.join(output_folder, f'{filename}_progression.png')
    plt.savefig(output_file)
    plt.close()

    print(f"Progression image saved to {output_file}")

# Create interactive map to visualize branch points, vectors, intersections, and curve points
def create_interactive_map(image, points, vectors, intersections, curve_points, output_path, filename):
    fig = go.Figure()
    fig.add_trace(go.Image(z=image))

    points_x = points[:, 1]
    points_y = points[:, 0]
    fig.add_trace(go.Scatter(x=points_x, y=points_y, mode='markers', marker=dict(color='purple', size=8), name="Branch Points"))

    for i in range(len(points)):
        fig.add_trace(go.Scatter(x=[points[i][1], points[i][1] + vectors[i][1] * 20],
                                 y=[points[i][0], points[i][0] + vectors[i][0] * 20],
                                 mode='lines+markers', marker=dict(color='red', size=5),
                                 line=dict(color='green'), name="Direction Vectors"))

    inter_x = intersections[:, 0]
    inter_y = intersections[:, 1]
    fig.add_trace(go.Scatter(x=inter_x, y=inter_y, mode='markers', marker=dict(color='blue', size=6), name="Intersections"))

    if len(curve_points) > 0:
        curve_x = curve_points[:, 1]
        curve_y = curve_points[:, 0]
        fig.add_trace(go.Scatter(x=curve_x, y=curve_y, mode='markers', marker=dict(color='yellow', size=8), name="Curve Points"))

    fig.update_layout(title="Interactive Map with Curve Points", autosize=True, height=image.shape[0], width=image.shape[1], 
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), xaxis_scaleanchor="y", yaxis_scaleanchor="x")

    output_file = os.path.join(output_path, f"{filename}.html")
    fig.write_html(output_file)
    print(f"Interactive map saved to {output_file}")

# Parallelized function to process each tile
def process_tile(filename, input_folder, tile_size, output_folder, tile_x, tile_y, output_folder_progression):
    img_path = os.path.join(input_folder, filename)
    tile_image = io.imread(img_path)
    
    bpoints, branch_vectors, epoints, end_vectors, intersections, curve_points = main(tile_image, filename, output_folder, tile_x, tile_y, tile_size)
    
     # Preprocess the image before skeletonization for visualization
    blur_radius = 1.5  # Adjust the blur radius for desired smoothness
    morph_size = 5     # Size of the disk for morphological closing
    
    smoothed_image = preprocess_image(tile_image, blur_radius=blur_radius, morph_size=morph_size)
    midline_image = midline_creation(smoothed_image)
    pro_image = processing(midline_image)
    bpoints_image = branch_points(pro_image)
    epoints_image = end_points(pro_image)
    
    # Call the save_progression_images function to save progression images
    save_progression_images(
        tile_image, smoothed_image, midline_image, pro_image, bpoints_image, epoints_image,
        branch_vectors, bpoints, curve_points, filename, output_folder_progression
    )
        
        
    return bpoints, branch_vectors, epoints, end_vectors, intersections, curve_points

# Main function to process image tiles (parallelized with tqdm for loading bars)
def stitch_images_and_create_map(input_folder, output_folder, interactive_map_output, output_folder2, tile_size=150):
    image_filenames = [f for f in os.listdir(input_folder) if f.startswith('predicted_tile') and f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Extract tile positions
    tile_positions = [(int(match.group(1)), int(match.group(2))) for filename in image_filenames if (match := re.search(r'predicted_tile_(\d+)_(\d+).png', filename))]

    # Process all tiles in parallel with tqdm to show progress
    results = Parallel(n_jobs=-1)(delayed(process_tile)(filename, input_folder, tile_size, output_folder, x, y,output_folder2)
                                  for filename, (x, y) in tqdm(zip(image_filenames, tile_positions), total=len(image_filenames), desc="Processing tiles"))
    
    # Initialize arrays for global points and vectors
    global_branch_points, global_branch_vectors, global_end_points, global_end_vectors, global_intersections, global_curve_points = zip(*results)
    
    # Concatenate results from all tiles
    global_branch_points = np.concatenate(global_branch_points)
    global_branch_vectors = np.concatenate(global_branch_vectors)
    global_end_points = np.concatenate(global_end_points)
    global_end_vectors = np.concatenate(global_end_vectors)
    global_intersections = np.concatenate(global_intersections)
    global_curve_points = np.concatenate(global_curve_points)

    # Stitch the final image
    max_x = max([x for x, y in tile_positions])
    max_y = max([y for x, y in tile_positions])
    total_width = (max_x + 1) * tile_size
    total_height = (max_y + 1) * tile_size
    stitched_image = np.zeros((total_height, total_width), dtype=np.uint8)
    
    # Use efficient numpy operations for stitching
    for (filename, (x, y)) in zip(image_filenames, tile_positions):
        img_path = os.path.join(input_folder, filename)
        tile_image = io.imread(img_path)
        
        if len(tile_image.shape) == 3 and tile_image.shape[2] == 3:
            tile_image = np.mean(tile_image, axis=2)  # Convert to grayscale
        
        x_start = x * tile_size
        y_start = y * tile_size
        stitched_image[y_start:y_start + tile_size, x_start:x_start + tile_size] = tile_image

    output_file = os.path.join(output_folder, 'stitched_image.png')
    io.imsave(output_file, stitched_image)

    # Export the results to a CSV file with progress bar
    with tqdm(total=100, desc="Exporting results to CSV") as pbar:
        export_to_csv(output_folder, global_branch_points, global_branch_vectors, global_end_points, global_end_vectors, global_intersections, global_curve_points, 'stitched_data')
        pbar.update(100)

# Function to profile the code
def profile_code():
    """Profile the main code with cProfile to find bottlenecks."""
    pr = cProfile.Profile()
    #pr.enable()  # Start profiling
    
    input_folder = './UNET V6.0/road-extraction-main/Output'
    output_folder = './UNET V6.0/road-extraction-main/Output2'
    output_folder2 = './UNET V6.0/road-extraction-main/Progression_Output'
    interactive_map_output = './UNET V6.0/road-extraction-main/Output_map'

    # Display a progress bar for the entire process
    with tqdm(total=100, desc="Overall Process") as overall_pbar:
        stitch_images_and_create_map(input_folder, output_folder, interactive_map_output, output_folder2, tile_size=256)
        overall_pbar.update(100)
        
    pr.disable()  # Stop profiling

    # Output the profiling results
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # print(s.getvalue())
    
# Function to load data from CSV
def load_data_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    points = data[['Point_Y', 'Point_X']].values  # Inverting for plotting (Y, X)
    vectors = data[['Vector_Y', 'Vector_X']].values  # Vectors (Y, X)
    intersections = data[['Intersection_X', 'Intersection_Y']].values  # Intersections (X, Y)
    return points, vectors, intersections


    
profile_code()