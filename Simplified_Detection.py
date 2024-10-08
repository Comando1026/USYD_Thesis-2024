# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:30:28 2024

@author: ethan
"""

import os
import cv2
import numpy as np
import csv
from skimage.morphology import skeletonize
from skimage import img_as_ubyte

def extract_images(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            img = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

def skeletonize_image(image):
    skeleton = skeletonize(image // 255)
    return img_as_ubyte(skeleton)

def remove_small_islands(image, min_size=100):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components -= 1

    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

def find_endpoints(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    filtered = cv2.filter2D(skeleton, -1, kernel)
    endpoints = np.argwhere(filtered == 11)
    return endpoints

def find_branch_points(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    filtered = cv2.filter2D(skeleton, -1, kernel)
    branch_points = np.argwhere(filtered >= 13)
    return branch_points

def find_curve_midpoints(skeleton):
    # Placeholder for curve detection and midpoint placement logic
    midpoints = []
    # Implementation required
    return midpoints

def connect_points(points):
    # Placeholder for connecting points with lines logic
    lines = []
    # Implementation required
    return lines

def export_to_csv(endpoints, branch_points, midpoints, lines, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "X", "Y"])
        for point in endpoints:
            writer.writerow(["End", point[1], point[0]])
        for point in branch_points:
            writer.writerow(["Branch", point[1], point[0]])
        for point in midpoints:
            writer.writerow(["Curve", point[1], point[0]])
        writer.writerow(["Lines"])
        writer.writerow(["Start_X", "Start_Y", "End_X", "End_Y"])
        for line in lines:
            writer.writerow(line)

def process_folder(folder_path, output_file):
    images = extract_images(folder_path)
    all_endpoints = []
    all_branch_points = []
    all_midpoints = []
    all_lines = []
    
    for image in images:
        skeleton = skeletonize_image(image)
        skeleton = remove_small_islands(skeleton)
        endpoints = find_endpoints(skeleton)
        branch_points = find_branch_points(skeleton)
        midpoints = find_curve_midpoints(skeleton)
        lines = connect_points(endpoints + branch_points + midpoints)
        
        all_endpoints.extend(endpoints)
        all_branch_points.extend(branch_points)
        all_midpoints.extend(midpoints)
        all_lines.extend(lines)
    
    export_to_csv(all_endpoints, all_branch_points, all_midpoints, all_lines, output_file)

# Example usage
input_folder = './UNET V6.0/road-extraction-main/Output'
CSV = './UNET V6.0/road-extraction-main/Output/Data.csv'
process_folder(input_folder, CSV)