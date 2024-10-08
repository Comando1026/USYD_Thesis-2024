# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:00:01 2024

@author: ethan
"""
# In[]
import pandas as pd
import plotly.graph_objects as go
import os
from skimage import io
from PIL import Image
import io as python_io
import base64
import numpy as np
from shapely.geometry import LineString, Point
from rtree import index
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for loading bars
import time
from rtree import index
from scipy.spatial import KDTree
# In[]
def downsample_image(image, scale_percentage=1):
    """
    Downsamples the image based on a percentage of its original size.

    :param image: NumPy array of the original image.
    :param scale_percentage: The percentage of the original size to scale down the image to.
    :return: Downsampled NumPy array of the image.
    """
    # Calculate new size based on percentage
    new_size = (int(image.shape[1] * scale_percentage), int(image.shape[0] * scale_percentage))
    
    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(image)
    
    # Resize the image based on the new size
    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    
    # Convert the PIL image back to a NumPy array
    return pil_image,new_size
# In[]
# Function to convert downsampled image to base64 string for Plotly
def image_to_base64(pil_img):
    buffer = python_io.BytesIO()
    pil_img.save(buffer, format="PNG")
    print(pil_img.size)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# In[]
# Optimize and pre-filter branch points based on distance and relevance
# Add checks for (0,0) vectors and invalid points
def consolidate_branch_points(branch_points, vectors, threshold=5):
    keep_mask = np.ones(len(branch_points), dtype=bool)
    
    for i in range(len(branch_points)):
        if not keep_mask[i]:
            continue
        distances = np.linalg.norm(branch_points[i] - branch_points, axis=1)
        close_indices = np.where((distances < threshold) & (distances > 0))[0]
        
        for idx in close_indices:
            # Check the original vectors before merging
            if np.all(vectors[idx] == 0):
                print(f"Warning: Zero vector found at index {idx} before consolidation")
            keep_mask[idx] = False
    
    consolidated_points = branch_points[keep_mask]
    consolidated_vectors = vectors[keep_mask]

    # Additional check after consolidation
    zero_vector_indices = np.where(np.all(consolidated_vectors == 0, axis=1))[0]
    if len(zero_vector_indices) > 0:
        print(f"Warning: Zero vectors found at indices {zero_vector_indices} after consolidation")
    
    return consolidated_points, consolidated_vectors

# In[]

def check_ray_intersects(ray_start, ray_end, points, threshold=10, max_distance=100):
    """
    Efficiently check if the ray intersects with any branch points within the given threshold,
    filtering out points that are farther than max_distance or are more than 180 degrees off the ray direction.
    
    :param ray_start: Starting point of the ray (numpy array or tuple)
    :param ray_end: Ending point of the ray (numpy array or tuple)
    :param points: Array of branch points (numpy array)
    :param threshold: Distance threshold for intersection
    :param max_distance: Maximum distance from the ray_start to consider points
    :return: Branch point if the ray intersects within threshold, otherwise None
    """
    
    # Ensure points are valid (no NaN or infinite values)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        raise ValueError("Invalid points array: contains NaN or infinite values.")

    # Check if ray_start and ray_end are the same
    if np.allclose(ray_start, ray_end):
        # Skip this ray if the start and end are the same
        return None

    # Compute Euclidean distance from ray_start to all points
    distances_to_start = (np.linalg.norm(points - ray_start, axis=1))
    
    # Filter out points that are farther than the max_distance
    points_filtered = points[distances_to_start <= max_distance]

    if points_filtered.size == 0:
        return None

    # Pre-filter out points that match the ray_start
    points_filtered = points_filtered[~np.all(points_filtered == ray_start, axis=1)]

    if points_filtered.size == 0:
        return None

    # Calculate the ray direction vector
    ray_direction = np.array(ray_end) - np.array(ray_start)
    ray_magnitude = np.linalg.norm(ray_direction)

    # Ensure ray magnitude is not zero (to avoid division by zero)
    if ray_magnitude == 0:
        raise ValueError("Invalid ray: ray_start and ray_end are the same point.")  # This check is now redundant but kept for completeness

    # Normalize the ray direction
    ray_direction = ray_direction / ray_magnitude

    # Calculate vectors from ray_start to each point
    vectors_to_points = points_filtered - ray_start

    # Avoid division by zero in normalization (filter out zero-length vectors)
    point_magnitudes = np.linalg.norm(vectors_to_points, axis=1)
    non_zero_magnitudes = point_magnitudes > 0
    vectors_to_points_normalized = vectors_to_points[non_zero_magnitudes] / point_magnitudes[non_zero_magnitudes][:, None]

    # Compute dot product between the ray direction and vectors to points
    dot_products = np.dot(vectors_to_points_normalized, ray_direction)

    # Create the ray line geometry
    ray_line = LineString([ray_start, ray_end])

    # Convert remaining points to shapely geometries in bulk
    point_geoms = [Point(p) for p in points_filtered]

    # Vectorize the distance calculation by creating an array of distances
    distances = np.array([ray_line.distance(pt) for pt in point_geoms])

    # Find the first point within the threshold, if any
    within_threshold = np.where(distances < threshold)[0]

    if within_threshold.size > 0:
        return points_filtered[within_threshold[0]]  # Return the first point within the threshold

    return None


# In[]
def ray_intersects(p1, v1, p2, v2):
    """
    Check if two rays intersect.
    
    p1, p2: Starting points of the rays (x, y).
    v1, v2: Direction vectors of the rays (dx, dy).
    
    Returns the intersection point (x, y) if they intersect, otherwise returns None.
    """
    # Unpack points and vectors
    x1, y1 = p1
    dx1, dy1 = v1
    x2, y2 = p2
    dx2, dy2 = v2
    
    # Solve for t1 and t2 in the parametric equations:
    # Ray 1: (x1, y1) + t1 * (dx1, dy1)
    # Ray 2: (x2, y2) + t2 * (dx2, dy2)
    
    # Parallel rays do not intersect
    denominator = dx1 * dy2 - dy1 * dx2
    if abs(denominator) < 1e-10:  # If denominator is close to zero, rays are parallel
        return None

    # Calculate the parameters t1 and t2
    t1 = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denominator
    t2 = ((x2 - x1) * dy1 - (y2 - y1) * dx1) / denominator
    
    # For a ray, t1 and t2 must both be >= 0 to intersect in the positive direction
    if t1 >= 0 and t2 >= 0:
        # Compute the intersection point
        intersect_x = x1 + t1 * dx1
        intersect_y = y1 + t1 * dy1
        return (intersect_x, intersect_y)
    
    return None  # No intersection if t1 or t2 is negative (meaning they intersect in the past)

# In[]

# def vector_calcs(points, vectors, intersect_points, inter_vectors,mask1=0,mask2=0 ,flag=False, arrow_scale=400):
#     vectors_x = []
#     vectors_y = []
    
#     # Track which primary rays have already intersected (True means the ray is still valid for checks)
#     if flag == True:
#         ray_active_mask = np.ones(len(points), dtype=bool)
#         ray_active_mask2 = np.ones(len(intersect_points), dtype=bool)
#     else:
#         ray_active_mask = mask2
#         ray_active_mask2 = mask1
    
#     kd_tree = KDTree(points)
    
#     for i, (point, vector) in tqdm(enumerate(zip(points, vectors)), total=len(points), desc="Extending Rays"):  
#         if not ray_active_mask[i]:  # If the ray is already intersected, skip it for primary rays
#             continue
        
#         point_x, point_y = point[1], point[0]  # Unpack and apply offset once
#         intersects = []
        
#         # Extend the vector to create an arrow
#         vector_point_x = point_x + vector[0] * arrow_scale
#         vector_point_y = point_y + vector[1] * arrow_scale
    
#         # Check intersections efficiently with other primary rays
#         point_checked = check_ray_intersects((point_x, point_y), (vector_point_x, vector_point_y), points)
        
#         # Use KDTree to find nearest neighbors within a specific radius
#         neighbors = kd_tree.query_ball_point([point_x, point_y], r=200)
        
#         # Efficient intersection checking using KDTree results (check only nearby points)
#         for j in neighbors:
#             if i != j and ray_active_mask[j]:  # Don't check the ray against itself or inactive rays
#                 other_point = points[j]
#                 other_vector = vectors[j]
#                 intersect = ray_intersects((point_x, point_y), vector, (other_point[0], other_point[1]), other_vector)
#                 if intersect is not None:
#                     intersects.extend([(intersect[0], intersect[1])])
        
#         # Check intersections with the secondary rays (always active)
#         point_checked_secondary = check_ray_intersects((point_x, point_y), (vector_point_x, vector_point_y), intersect_points)
        
#         # Check intersections with secondary rays' vectors (these are always checked)
#         for j, (other_point, other_vector) in enumerate(zip(intersect_points, inter_vectors)):
#             intersect = ray_intersects((point_x, point_y), vector, (other_point[0], other_point[1]), other_vector)
#             if intersect is not None:
#                 intersects.extend([(intersect[0], intersect[1])])

#         # Collect the potential intersection points
#         potential_points = [point_checked, point_checked_secondary] + intersects
        
#         # Filter out None values (no intersection found)
#         valid_points = [p for p in potential_points if p is not None]
        
#         if len(valid_points) == 0:
#             # No intersections found
#             vectors_x.extend([None, None, None])
#             vectors_y.extend([None, None, None])
#         else:
#             # Compute distances to find the closest intersection point
#             distances = np.hypot(np.array([p[0] for p in valid_points]) - point_x,
#                                  np.array([p[1] for p in valid_points]) - point_y)
#             closest_point = valid_points[np.argmin(distances)]

#             if np.hypot(closest_point[0] - point_x, closest_point[1] - point_y) < 100:
#                 vectors_x.extend([point_x, closest_point[0], None])
#                 vectors_y.extend([point_y, closest_point[1], None])
                
#                 # Mark this primary ray as intersected and no longer active
#                 ray_active_mask[i] = False
                
#                 # Optionally, deactivate the intersecting primary ray if needed (find which ray it is)
#                 for j in neighbors:
#                     if np.allclose(closest_point, points[j], atol=1e-6):
#                         ray_active_mask[j] = False  # Deactivate the intersecting ray
#                 closest_point = np.asarray(closest_point)
                
#                 # Compute the element-wise comparison in one go for all points in intersect_points
#                 ray_active_mask2[np.allclose(closest_point, intersect_points, atol=1e-6)] = False

#                 # Handle the 'flag' condition only when necessary
#                 if flag and point_checked_secondary is not None:
#                     vectors_x.extend([point_x, point_checked_secondary[0], None])
#                     vectors_y.extend([point_y, point_checked_secondary[1], None])
                    
#     if flag == True:

#         return vectors_x, vectors_y,ray_active_mask, ray_active_mask2

#     else: 
#         return vectors_x, vectors_y


def vector_calcs(points, vectors, intersect_points,mask =0, mask2 = 1, flag=False, arrow_scale=400):
    vectors_x = []
    vectors_y = []

    for i, (point, vector) in tqdm(enumerate(zip(points, vectors)), total=len(points), desc="Extending Rays"):  
        point_x, point_y = point[1], point[0]  # Unpack and apply offset once
        
        # Extend the vector to create an arrow
        vector_point_x = point_x + vector[0] * arrow_scale
        vector_point_y = point_y + vector[1] * arrow_scale
    
        # Check intersections efficiently
        point_checked = check_ray_intersects((point_x, point_y), (vector_point_x, vector_point_y), intersect_points)
        point_checked2 = check_ray_intersects((point_x, point_y), (vector_point_x, vector_point_y), points)

        # Collect the potential intersection points
        potential_points = [point_checked, point_checked2]
        
        # Filter out None values (no intersection found)
        valid_points = [p for p in potential_points if p is not None]
        
 
        
        if len(valid_points) == 0:
            # No intersections found
            vectors_x.extend([None, None, None])
            vectors_y.extend([None, None, None])
        else:
            # Compute distances to find the closest intersection point
            distances = [np.hypot(p[0] - point_x, p[1] - point_y) for p in valid_points]
            closest_point = valid_points[np.argmin(distances)]

            if distances[0]>100:
                print(distances)
                print(f"closest point {closest_point} of {valid_points} to {point}")

            vectors_x.extend([point_x, closest_point[0], None])
            vectors_y.extend([point_y, closest_point[1], None])

            # Handle the 'flag' condition only when necessary
            if flag and point_checked2 is not None:
                vectors_x.extend([point_x, point_checked2[0], None])
                vectors_y.extend([point_y, point_checked2[1], None])

    return vectors_x, vectors_y
 


# In[]
# Simplified Plotting Function
def export_to_csv(branch_points, branch_vectors, end_points, end_vectors, vx,vy,Evx,Evy, output_path, filename):
    """
    Export branch points, vectors, and end points with vectors to a CSV file.
    
    :param branch_points: Array of branch points.
    :param branch_vectors: Array of branch vectors.
    :param end_points: Array of end points.
    :param end_vectors: Array of end vectors.
    :param output_path: Path to save the CSV file.
    :param filename: Name of the CSV file to be saved.
    """
    # Find the minimum length among the arrays
    min_length = min(len(branch_points), len(branch_vectors), len(end_points), len(end_vectors))

    # Truncate all arrays to the minimum length
    branch_points = branch_points[:min_length]
    branch_vectors = branch_vectors[:min_length]
    end_points = end_points[:min_length]
    end_vectors = end_vectors[:min_length]
    vx =  vx[:min_length]
    vy =  vy[:min_length]
    Evx =  Evx[:min_length]
    Evy =  Evy[:min_length]
    print(vx)

    # Prepare data for export
    data_dict = {
        'Branch_Point_X': branch_points[:, 0],
        'Branch_Point_Y': branch_points[:, 1],
        'Branch_Vector_X': branch_vectors[:, 0],
        'Branch_Vector_Y': branch_vectors[:, 1],
        'End_Point_X': end_points[:, 1],
        'End_Point_Y': end_points[:, 0],  # Swap to match XY format
        'End_Vector_X': end_vectors[:, 0],
        'End_Vector_Y': end_vectors[:, 1],
        'Extended_Vector_Branch_X' : vx,
        'Extended_Vector_Branch_Y' : vy,
        'Extended_Vector_End_X' : Evx,
        'Extended_Vector_End_Y' : Evy,
        
    }

    # Convert to DataFrame
    export_df = pd.DataFrame(data_dict)

    # Create the full path to save the CSV file
    output_file = os.path.join(output_path, f"{filename}.csv")

    # Export to CSV
    export_df.to_csv(output_file, index=False)
    print(f"Data exported to {output_file}")
# In[] (Update the interactive map function to include CSV export)
def create_interactive_map_from_csv(image, csv_file, output_path, filename, scale_percentage):
    # Load data and filter down
    data = pd.read_csv(csv_file)
    offset_scale = 0.97 / 2

    data_clean = data.dropna()

    points = data_clean[['Branch_Point_X', 'Branch_Point_Y']].values * scale_percentage * offset_scale 
    vectors = data_clean[['Branch_Vector_X', 'Branch_Vector_Y']].values * scale_percentage * offset_scale 
    end_points = data_clean[['End_Point_Y', 'End_Point_X']].values * scale_percentage * offset_scale
    end_vectors = data_clean[['End_Vector_X', 'End_Vector_Y']].values * scale_percentage * offset_scale 

    # Assuming the "curve" points are stored in columns 'Curve_Point_X' and 'Curve_Point_Y'
    curve_points = data_clean[['Curve_Point_Y', 'Curve_Point_X']].values * scale_percentage * offset_scale 

    # Pre-filter points for performance
    points, vectors = consolidate_branch_points(points, vectors, threshold=5)

    # Downsample the image for efficient HTML display
    downsampled_image, size = downsample_image(image, scale_percentage)
    image_base64 = image_to_base64(downsampled_image)

    # Plotly figure creation
    fig = go.Figure()
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')

    # Add the downsampled image
    fig.add_layout_image(
        dict(
            source=image_base64,
            xref="x",
            yref="y",
            x=0,
            y=0 + size[1],
            sizex=size[0],
            sizey=size[1],
            sizing="contain",
            layer="below",
            opacity=1
        )
    )
    
    # Adjust points for offset
    Max_y = max(points[:, 1])
    Min_y = min(points[:, 1])
    Min_x = min(points[:, 0])
    Max_x = max(points[:, 0])
    offset = -(Min_y + Max_y)
    
    points[:, 1] = points[:, 1] + offset

    # Plot branch points
    points_trace = go.Scattergl(
        x=(points[:, 0]), 
        y=(points[:, 1]), 
        mode='markers', 
        marker=dict(color='purple', size=10), 
        name="Branch Points"
    )

    # Adjust end points for offset
    EMax_y = max(end_points[:, 0])
    EMin_y = min(end_points[:, 0])
    EMin_x = min(end_points[:, 1])
    EMax_x = max(end_points[:, 1])
    Eoffset = -(EMin_y + EMax_y)
    end_points[:, 0] = end_points[:, 0] + Eoffset

    end_points_trace = go.Scattergl(
        x=(end_points[:, 1]),
        y=(end_points[:, 0]),
        mode='markers', 
        marker=dict(color='blue', size=10), 
        name="End Points"
    )
    fig.add_trace(end_points_trace)
    fig.add_trace(points_trace)

    # Plot the curve points
    curve_trace = go.Scattergl(
        x=(curve_points[:, 0]), 
        y=(curve_points[:, 1]), 
        mode='markers', 
        marker=dict(color='green', size=10), 
        name="Curve Points"
    )
    fig.add_trace(curve_trace)
    
    # Perform vector calculations (if necessary)
    vy, vx = vector_calcs(points, vectors, end_points, end_vectors, flag=True)
    Evx, Evy = vector_calcs(end_points, end_vectors, points, vectors)

    vector_trace = go.Scatter(
        x=vx,
        y=vy,
        mode='lines',
        line=dict(color='red', width=2),
        name="Branch Point Vectors"
    )
    fig.add_trace(vector_trace)

    end_vector_trace = go.Scatter(
        x=Evx,
        y=Evy,
        mode='lines',
        line=dict(color='orange', width=2),
        name="End Point Vectors"
    )
    fig.add_trace(end_vector_trace)

    # Set axis scaling to ensure image and points use the same axis
    fig.update_xaxes(
        scaleanchor="y",  
        scaleratio=1,     
        constrain="domain",  
        range=[Min_x, Max_x]  
    )

    fig.update_yaxes(
        scaleanchor="x",  
        scaleratio=1,     
        constrain="domain",  
        range=[Min_y, Max_y]  
    )

    # Save the interactive map as an HTML file
    output_file = os.path.join(output_path, f"{filename}.html")
    fig.write_html(output_file)
    print(f"Interactive map saved to {output_file}")

    # Optionally, export the data to a CSV
    export_to_csv(points, vectors, end_points,end_vectors, vx,vy,Evx,Evy, output_path, f"{filename}_points_vectors")


# In[] (Usage example with the new export functionality)
csv_file = r"C:\Users\ethan\OneDrive\Desktop\Thesis 2024\Code\Python_UNET_Code\UNET V6.0\road-extraction-main\Output2\stitched_data_coordinates.csv"
stitched_image_path = './Predictions/Test_output/downsampled_image.png'
interactive_map_output = './UNET V6.0'
stitched_image = io.imread(stitched_image_path)
scale = 1

create_interactive_map_from_csv(stitched_image, csv_file, interactive_map_output, 'optimized_stitched_map', scale)