"""
Alignment Measurements Script

This script calculates biomechanical alignment parameters from keypoint annotations.

Input Format:
    - LabelMe JSON format annotation files
    - Each JSON file should contain 36 keypoints in the standard order
    - Keypoints can be exported from LLRFormer using tools/export_predictions.py

Usage:
    # Single file mode:
    python alignment_measurements.py
    
    # Batch processing mode:
    python alignment_measurements.py <folder_path> [output_csv] [pixel_spacing]
    
    # Example:
    python alignment_measurements.py output/predictions biomechanics_results.csv 0.1237

Note:
    - To export predictions from LLRFormer model, use:
      python tools/export_predictions.py --cfg configs/llrformer.yaml --model-file <model_path> --output-dir output/predictions
"""

import json
import math
import numpy as np
from PIL import Image

def load_points_from_json(json_path):
    """Load keypoint coordinates from LabelMe format JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    points = {}
    for shape in data['shapes']:
        name = shape['label']
        if shape['shape_type'] == 'circle':
            p1, p2 = shape['points']
            if name in ['R_FC', 'L_FC']:
                points[name] = p1
            else:
                center = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
                points[name] = center
        else:
            points[name] = shape['points'][0]
    return points, data

def midpoint(p1, p2):
    """Calculate midpoint of two points."""
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]

def line_length(p1, p2):
    """Calculate distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def intersection(p1, p2, p3, p4):
    """Calculate intersection of two line segments."""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return [px, py]

def point_line_distance(point, line_p1, line_p2):
    """Calculate distance from point to line."""
    return abs((line_p2[0]-line_p1[0])*(line_p1[1]-point[1]) -
               (line_p1[0]-point[0])*(line_p2[1]-line_p1[1])) / line_length(line_p1, line_p2)

def angle_between_lines(p1, p2, p3, p4):
    """Calculate angle between two lines in degrees."""
    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
    v2 = np.array([p4[0]-p3[0], p4[1]-p3[1]])
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return math.degrees(angle)

def calculate_anatomical_medial_angle(line1_p1, line1_p2, line2_p1, line2_p2, side='R'):
    """Calculate medial angle."""
    v1 = np.array([line1_p2[0] - line1_p1[0], line1_p2[1] - line1_p1[1]])
    v2 = np.array([line2_p2[0] - line2_p1[0], line2_p2[1] - line2_p1[1]])
    
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle = np.arccos(np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0))
    
    cross_product = v1[0]*v2[1] - v1[1]*v2[0]
    
    if side == 'R':
        if cross_product < 0:
            return math.degrees(angle)
        else:
            return math.degrees(angle)
    else:
        if cross_product > 0:
            return math.degrees(angle)
        else:
            return math.degrees(angle)

def calculate_anatomical_lateral_angle(line1_p1, line1_p2, line2_p1, line2_p2, side='R'):
    """Calculate lateral angle."""
    v1 = np.array([line1_p2[0] - line1_p1[0], line1_p2[1] - line1_p1[1]])
    v2 = np.array([line2_p2[0] - line2_p1[0], line2_p2[1] - line2_p1[1]])
    
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle = np.arccos(np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0))
    
    cross_product = v1[0]*v2[1] - v1[1]*v2[0]
    
    if side == 'R':
        if cross_product > 0:
            return math.degrees(angle)
        else:
            return math.degrees(angle)
    else:
        if cross_product < 0:
            return math.degrees(angle)
        else:
            return math.degrees(angle)

def compute_biomechanics(json_path, image_path=None, pixel_spacing=0.1237):
    points, data = load_points_from_json(json_path)
    results = {}
    
    img_height = data.get('imageHeight', 7621)
    img_width = data.get('imageWidth', 2808)
    
    print(f"Image size: {img_width} x {img_height}")
    print(f"Pixel spacing: {pixel_spacing} mm/pixel")

    for side in ['R', 'L']:
        print(f"\n=== {side} side calculation ===")
        
        FC = points[f'{side}_FC']
        GT = points[f'{side}_GT']
        FNeck_Cut_Up = points[f'{side}_FNeck_Cut_Up']
        FNeck_Cut_Down = points[f'{side}_FNeck_Cut_Down']
        Cdy_Up = points[f'{side}_Cdy_Up']
        Cdy_Down = points[f'{side}_Cdy_Down']
        IF = points[f'{side}_IF']
        IR = points[f'{side}_IR']
        LLP = points[f'{side}_LLP']
        MLP = points[f'{side}_MLP']
        LPC = points[f'{side}_LPC']
        MPC = points[f'{side}_MPC']
        LE = points[f'{side}_LE']
        ME = points[f'{side}_ME']
        Cyd_Up = points[f'{side}_Cyd_Up']
        Cyd_Down = points[f'{side}_Cyd_Down']
        DLP = points[f'{side}_DLP']
        DMP = points[f'{side}_DMP']

        femoral_mech_axis = (FC, IF)
        tibial_mech_axis = (IR, midpoint(DLP, DMP))
        anatomical_axis_femur = (Cdy_Up, Cdy_Down)
        lower_limb_mech_axis = (FC, midpoint(DLP, DMP))
        tibial_anatomical_axis = (Cyd_Up, Cyd_Down)
        hip_joint_orientation_line = (FC, GT)
        knee_joint_line_distal_femur = (LLP, MLP)
        knee_joint_line_proximal_tibia = (LPC, MPC)
        ankle_joint_line = (DLP, DMP)
        femoral_neck_axis = (FC, midpoint(FNeck_Cut_Up, FNeck_Cut_Down))
        IF_IR_mid = midpoint(IF, IR)
        LE_ME_line = (LE, ME)
        vertical_axis = ([FC[0], 0], [FC[0], img_height])

        try:
            results[f'{side}_aMPFA'] = round(180 - calculate_anatomical_medial_angle(*anatomical_axis_femur, *hip_joint_orientation_line, side), 2)
            results[f'{side}_mLPFA'] = round(calculate_anatomical_lateral_angle(*femoral_mech_axis, *hip_joint_orientation_line, side), 2)
            results[f'{side}_mLDFA'] = round(calculate_anatomical_lateral_angle(*femoral_mech_axis, *knee_joint_line_distal_femur, side), 2)
            results[f'{side}_aLDFA'] = round(calculate_anatomical_lateral_angle(*anatomical_axis_femur, *knee_joint_line_distal_femur, side), 2)
            results[f'{side}_MPTA'] = round(calculate_anatomical_medial_angle(*knee_joint_line_proximal_tibia, *tibial_mech_axis, side), 2)
            results[f'{side}_LDTA'] = round(calculate_anatomical_lateral_angle(*ankle_joint_line, *tibial_mech_axis, side), 2)
            results[f'{side}_HKA'] = round(180 - calculate_anatomical_lateral_angle(*tibial_mech_axis, *femoral_mech_axis, side), 2)
            results[f'{side}_NSA'] = round(180 - calculate_anatomical_medial_angle(*anatomical_axis_femur, *femoral_neck_axis, side), 2)
            results[f'{side}_MA'] = round(angle_between_lines(*femoral_mech_axis, *vertical_axis), 2)
            results[f'{side}_aFTA'] = round(180 - calculate_anatomical_lateral_angle(*tibial_anatomical_axis, *anatomical_axis_femur, side), 2)
            results[f'{side}_JLCA'] = round(angle_between_lines(*knee_joint_line_distal_femur, *knee_joint_line_proximal_tibia), 2)
            results[f'{side}_aMFA'] = round(angle_between_lines(*femoral_mech_axis, *anatomical_axis_femur), 2)
            
            knee_center = intersection(*lower_limb_mech_axis, *LE_ME_line)
            if knee_center:
                tibial_plateau_width = line_length(LE, ME)
                if side == 'R':
                    medial_point = ME
                else:
                    medial_point = LE
                
                mad_px = point_line_distance(IF_IR_mid, *lower_limb_mech_axis)
                results[f'{side}_MAD_mm'] = round(mad_px * pixel_spacing, 2)
                
                if side == 'R':
                    position = "lateral" if IF_IR_mid[0] > knee_center[0] else "medial"
                else:
                    position = "lateral" if IF_IR_mid[0] < knee_center[0] else "medial"
                results[f'{side}_MAD_position'] = position
                
                if knee_center and tibial_plateau_width > 0:
                    inter_ME_dist = line_length(knee_center, medial_point)
                    results[f'{side}_dis%'] = round(inter_ME_dist / tibial_plateau_width, 3)
            
        except Exception as e:
            print(f"Error calculating {side} side: {e}")
            continue

    return results

import os
import csv
from pathlib import Path

def batch_process_folder(folder_path, output_csv="biomechanics_results.csv", pixel_spacing=0.1237):
    """Batch process all images in a folder."""
    folder_path = Path(folder_path)
    
    json_files = list(folder_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    csv_headers = [
        'Filename', 'R_aMPFA', 'R_mLPFA', 'R_mLDFA', 'R_aLDFA', 'R_MPTA', 'R_LDTA', 
        'R_HKA', 'R_NSA', 'R_MA', 'R_aFTA', 'R_JLCA', 'R_aMFA', 'R_MAD_mm', 
        'R_MAD_position', 'R_dis%',
        'L_aMPFA', 'L_mLPFA', 'L_mLDFA', 'L_aLDFA', 'L_MPTA', 'L_LDTA', 
        'L_HKA', 'L_NSA', 'L_MA', 'L_aFTA', 'L_JLCA', 'L_aMFA', 'L_MAD_mm', 
        'L_MAD_position', 'L_dis%'
    ]
    
    results_list = []
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\nProcessing file {i}/{len(json_files)}: {json_file.name}")
        
        try:
            image_file = None
            base_name = json_file.stem
            for ext in ['.jpeg', '.jpg', '.png', '.bmp']:
                potential_image = json_file.parent / f"{base_name}{ext}"
                if potential_image.exists():
                    image_file = str(potential_image)
                    break
            
            if not image_file:
                print(f"  Info: Corresponding image file not found, will use image size info from JSON")
            
            results = compute_biomechanics(str(json_file), image_file, pixel_spacing)
            
            row_data = [json_file.name]
            
            for param in ['aMPFA', 'mLPFA', 'mLDFA', 'aLDFA', 'MPTA', 'LDTA', 'HKA', 'NSA', 'MA', 'aFTA', 'JLCA', 'aMFA']:
                key = f'R_{param}'
                row_data.append(results.get(key, ''))
            
            row_data.append(results.get('R_MAD_mm', ''))
            row_data.append(results.get('R_MAD_position', ''))
            row_data.append(results.get('R_dis%', ''))
            
            for param in ['aMPFA', 'mLPFA', 'mLDFA', 'aLDFA', 'MPTA', 'LDTA', 'HKA', 'NSA', 'MA', 'aFTA', 'JLCA', 'aMFA']:
                key = f'L_{param}'
                row_data.append(results.get(key, ''))
            
            row_data.append(results.get('L_MAD_mm', ''))
            row_data.append(results.get('L_MAD_position', ''))
            row_data.append(results.get('L_dis%', ''))
            
            results_list.append(row_data)
            print(f"  ✓ Calculation completed")
            
        except Exception as e:
            print(f"  ✗ Processing failed: {e}")
            empty_row = [json_file.name] + [''] * (len(csv_headers) - 1)
            results_list.append(empty_row)
            continue
    
    output_path = folder_path / output_csv
    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            writer.writerows(results_list)
        
        print(f"\n✓ Batch processing completed!")
        print(f"✓ Results saved to: {output_path}")
        print(f"✓ Successfully processed: {len([r for r in results_list if any(r[1:])])} files")
        print(f"✓ Failed: {len([r for r in results_list if not any(r[1:])])} files")
        
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def batch_process_with_progress(folder_path, output_csv="biomechanics_results.csv", pixel_spacing=0.1237):
    """Batch processing with progress display."""
    folder_path = Path(folder_path)
    json_files = list(folder_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Starting batch processing of {len(json_files)} files...")
    print("=" * 60)
    
    batch_process_folder(folder_path, output_csv, pixel_spacing)
    
    print("\n" + "=" * 60)
    print("Batch Processing Summary:")
    print(f"Input folder: {folder_path}")
    print(f"Output file: {folder_path / output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "biomechanics_results.csv"
        pixel_spacing = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1237
        
        print("=" * 60)
        print("Lower Limb Full-Length X-ray Biomechanics Measurement Analysis")
        print("=" * 60)
        print(f"Input folder: {folder_path}")
        print(f"Output file: {output_csv}")
        print(f"Pixel spacing: {pixel_spacing} mm/pixel")
        print("=" * 60)
        
        batch_process_with_progress(folder_path, output_csv, pixel_spacing)
        
    else:
        json_file = "1_2_410_200049_2_47176438882121_3_1_20180409083633651_83807.json"
        image_file = "1_2_410_200049_2_47176438882121_3_1_20180409083633651_83807.jpeg"
        
        print("=" * 60)
        print("Lower Limb Full-Length X-ray Biomechanics Measurement Analysis")
        print("=" * 60)
        print(f"JSON file: {json_file}")
        print(f"Image file: {image_file}")
        print("=" * 60)
        
        try:
            res = compute_biomechanics(json_file, image_file)
            
            print("\n" + "=" * 60)
            print("Calculation Results:")
            print("=" * 60)
            
            angle_params = ['aMPFA', 'mLPFA', 'mLDFA', 'aLDFA', 'MPTA', 'LDTA', 'HKA', 'NSA', 'MA', 'aFTA', 'JLCA', 'aMFA']
            distance_params = ['MAD_mm', 'dis%']
            
            for side in ['R', 'L']:
                print(f"\n{side} side results:")
                print("-" * 30)
                
                for param in angle_params:
                    key = f'{side}_{param}'
                    if key in res:
                        print(f"{param}: {res[key]}°")
                
                for param in distance_params:
                    key = f'{side}_{param}'
                    if key in res:
                        if 'MAD' in param:
                            position = res.get(f'{side}_MAD_position', '')
                            print(f"{param}: {res[key]} mm ({position})")
                        else:
                            print(f"{param}: {res[key]}")
            
            print("\n" + "=" * 60)
            print("Calculation completed!")
            print("=" * 60)
            
        except Exception as e:
            print(f"Runtime error: {e}")
            import traceback
            traceback.print_exc()
