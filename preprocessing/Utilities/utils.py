import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import math
from scipy.spatial import KDTree


def closest_points(target, points, threshold):
    import math
    closest = []
    indicies = []
    for index,point in enumerate(points):
        distance = math.dist(target,point)
        if distance <= threshold:
            closest.append(point)
            indicies.append(index)
    return closest,indicies

def find_next_val(index,threshold_breaks,frame_filter,port_type):
    p2_indicies = np.where(threshold_breaks == port_type)[0]
    try:
        p2_min_val = min(i for i in p2_indicies if i > index)
        distance = p2_min_val - index
    except:
        distance = 9999999
    if distance<frame_filter:
        return p2_min_val
    else:
        return -1

def find_p5_segments(threshold_breaks, frame_filter, find_next_val):
    start_ind = []
    end_ind = []
    P5_in_ind = []  # kept for parity, though unused in original code

    index = 0
    n = len(threshold_breaks)

    while index < n - 1:
        break_ = threshold_breaks[index]

        if break_ == 5 and threshold_breaks[index + 1] != 5:
            p2_ind = find_next_val(index, threshold_breaks, frame_filter, 2)
            if p2_ind == -1:
                index += 1
                continue

            p3_ind = find_next_val(p2_ind, threshold_breaks, frame_filter, 3)
            if p3_ind == -1:
                index += 1
                continue

            p4_ind = find_next_val(p3_ind, threshold_breaks, frame_filter, 4)
            if p4_ind == -1:
                index += 1
                continue

            p5_ind = find_next_val(p4_ind, threshold_breaks, frame_filter, 5)
            if p5_ind == -1:
                index += 1
                continue

            p5_out_ind = find_next_val(p5_ind, threshold_breaks, frame_filter, 0)
            if p5_out_ind == -1:
                index += 1
                continue

            start_ind.append(index)
            end_ind.append(p5_out_ind)

            # jump index forward
            index = p5_out_ind - 1
        else:
            index += 1

    return start_ind, end_ind

def find_closest_point(target, points):
    import math
    min_distance = float('inf')
    closest_point = None
    closest_index = None
    for index, point in enumerate(points):
        distance = math.dist(target,point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
            closest_index = index
    return closest_point,closest_index

def find_closest_points(traject_coords,port_centroids):
    
    split_trajects_port_points = []
    split_trajects_port_indicies = []

    closest_points = []
    closest_inds = []
    for ind_,centroids in enumerate(port_centroids[1::]):
        # skip 0 ^see above and skip port 3: 
        if not centroids == port_centroids[2]:
            if not centroids == port_centroids[-1]:
                if ind_ ==  0:
                    # find closest to port 2
                    closest_point, closest_index = find_closest_point(centroids,traject_coords)
                    closest_points += [closest_point]
                    closest_inds += [closest_index]
                else:
                    #find closest to port 4 (from after port 2 onwards)
                    closest_point, closest_index = find_closest_point(centroids,traject_coords[closest_inds[-1]::])
                    closest_points += [closest_point]
                    closest_inds += [closest_index + closest_inds[-1]]
            else:
                # if port 5 search from port 4 onwards
                closest_point, closest_index = find_closest_point(centroids,traject_coords[closest_inds[-1]::])
                closest_points += [closest_point]
                closest_inds += [closest_index + closest_inds[-1]]

    ###make it so that point closest to port 1 can only be between port 5 and 2! 
    closest_point_1, closest_index_1 = find_closest_point(port_centroids[0],traject_coords[0:closest_inds[0]])

    ###make it so that point closest to port 3 can only be between port 2 and 4! 
    closest_point_3, closest_index_3 = find_closest_point(port_centroids[2],traject_coords[closest_inds[0]:closest_inds[1]])
    # this finds relative ind so I need to add on the offst from the start of the full traject
    closest_index_3 = closest_index_3 + closest_inds[0]

    split_trajects_port_points += [[closest_point_1] + [closest_points[0]] + [closest_point_3] + closest_points[1::]]
    split_trajects_port_indicies += [[closest_index_1] + [closest_inds[0]] + [closest_index_3] + closest_inds[1::]]
    
    return split_trajects_port_points, split_trajects_port_indicies


def interpolate_to_longest_and_find_average_curve(curves):
    
    

    # Find the length of the longest curve
    max_length = max([len(curve) for curve in curves])

    # Interpolate each curve to the length of the longest curve
    interpolated_curves = []
    for curve in curves:
        if len(curve) > 0:
            x = [point[0] for point in curve]
            y = [point[1] for point in curve]

            # find lots of points on the piecewise linear curve defined by x and y
            M = max_length
            t = np.linspace(0, len(x), M)
            x_interp = np.interp(t, np.arange(len(x)), x)
            y_interp = np.interp(t, np.arange(len(y)), y)

            interpolated_curves.append([[x, y] for x, y in zip(x_interp, y_interp)])

    # # Average the x and y coordinates of all the interpolated curves
    average_curve = []
    for i in range(max_length):
        x_sum = 0
        y_sum = 0
        for curve in interpolated_curves:
            x_sum += curve[i][0]
            y_sum += curve[i][1]
        average_curve.append([x_sum / len(interpolated_curves), y_sum / len(interpolated_curves)])

    return average_curve


def total_length_of_curve(curve):
    x = [point[0] for point in curve]
    y = [point[1] for point in curve]
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.sum(dists)


def closest_point(line1, line2):
    tree = KDTree(line2)
    dist, index = tree.query(line1)
    return index, dist


def sortperm_neurons(bkgd_log_proportions_array,config,neuron_response_df, sequence_ordering=None, th=0.2):
    ## this is number of neurons in total
    N_neurons= bkgd_log_proportions_array.shape[1]
    ## number of sequences from json file 
    n_sequences = config["num_sequence_types"]
    # the 18 neuron params for each neuron from the last iteration
    all_final_globals = neuron_response_df.iloc[-N_neurons:]
    # this cuts it down to just the first 6 params - i think this correspond sto the first param for each seq type? response probABILITY - ie the chance that a neuron spikes in a given latent seq 
    resp_prop = np.exp(all_final_globals.values[:, :n_sequences])#
    # this takes the next 6 params - which i think are the offset values
    offset = all_final_globals.values[-N_neurons:, n_sequences:2*n_sequences]
    ## finds the max response value - ie. which seq it fits to? 
    peak_response = np.amax(resp_prop, axis=1)
    # then threshold the reponse
    has_response = peak_response > np.quantile(peak_response, th)
    # I thin this is the sequence that the neuron has the max response for: ie. we are ordering them by max response 
    preferred_type = np.argmax(resp_prop, axis=1)
    if sequence_ordering is None:
        # order them by max reponse 
        ordered_preferred_type = preferred_type
    else:
        #order them differnetly 
        ordered_preferred_type = np.zeros(N_neurons)#
        # loop through each sequence
        for seq in range(n_sequences):
            # where does  max repsone = user defined seque
            seq_indices = np.where(preferred_type == sequence_ordering[seq])
            # change order to different seq
            ordered_preferred_type[seq_indices] = seq

    # reorder the offset params according to max respsone
    preferred_delay = offset[np.arange(N_neurons), preferred_type]
    Z = np.stack([has_response, ordered_preferred_type+1, preferred_delay], axis=1)
    indexes = np.lexsort((Z[:, 2], Z[:, 1], Z[:, 0]))
    return indexes,ordered_preferred_type

# Function to find corresponding number in another column
def find_corresponding(nums):
    return [df_dict[num] for num in nums]

def split_list(nums):
    sublists = []
    current_sublist = [nums[0]]
    current_element = nums[0]
    for i in range(1,len(nums)):
        if nums[i] == current_element:
            current_sublist.append(nums[i])
        else:
            sublists.append(current_sublist)
            current_sublist = [nums[i]]
            current_element = nums[i]
    sublists.append(current_sublist)
    return sublists

def conactinate_nth_items(startlist):
    concatinated_column_vectors = []
    for c in range(len(max(startlist, key=len))):
        column = []
        for t in range(len(startlist)):
            if c <= len(startlist[t])-1:
                column = column + [startlist[t][c]]
        concatinated_column_vectors.append(column)
    return concatinated_column_vectors

def load_H5_bodypart(tracking_path,video_type, tracking_point):

    # Load in all '.h5' files for a given folder:
    TFiles_unsort = list_files(tracking_path, 'h5')

    for file in TFiles_unsort:
        print(file)
        if video_type in file or video_type.upper() in file:
            if 'BACK' in file or 'task' in file:
                if not 'PORT' in file:
                    back_file = pd.read_hdf(tracking_path + file)     
                
    # drag data out of the df
    scorer = back_file.columns.tolist()[0][0]
    body_part = back_file[scorer][tracking_point]
    
    parts=[]
    for item in list(back_file[scorer]):
        parts+=[item[0]]
    print(np.unique(parts))
    
    # clean and interpolate frames with less than 98% confidence
    clean_and_interpolate(body_part,0.98)
    
    return(body_part)
  
def load_H5_ports(tracking_path,video_type):

    # Load in all '.h5' files for a given folder:
    TFiles_unsort = list_files(tracking_path, 'h5')

    for file in TFiles_unsort:
        print(file)
        if video_type in file or video_type.upper() in file:
            if 'PORT' in file or 'port' in file:
                back_ports_file = pd.read_hdf(tracking_path + file)

    ## same for the ports:
    scorer = back_ports_file.columns.tolist()[0][0]
        
    try:
        port1 =back_ports_file[scorer]['port2']
        port2 =back_ports_file[scorer]['port1']
        port3 =back_ports_file[scorer]['port6']
        port4 =back_ports_file[scorer]['port3']
        port5 =back_ports_file[scorer]['port7']
    except:
        port1 =back_ports_file[scorer]['Port2']
        port2 =back_ports_file[scorer]['Port1']
        port3 =back_ports_file[scorer]['Port6']
        port4 =back_ports_file[scorer]['Port3']
        port5 =back_ports_file[scorer]['Port7']

    clean_and_interpolate(port1,0.98)
    clean_and_interpolate(port2,0.98)
    clean_and_interpolate(port3,0.98)
    clean_and_interpolate(port4,0.98)
    clean_and_interpolate(port5,0.98)
    
    return(port1,port2,port3,port4,port5)

def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def clean_and_interpolate(head_centre,threshold):
    bad_confidence_inds = np.where(head_centre.likelihood.values<threshold)[0]
    newx = head_centre.x.values
    newx[bad_confidence_inds] = 0
    newy = head_centre.y.values
    newy[bad_confidence_inds] = 0

    start_value_cleanup(newx)
    interped_x = interp_0_coords(newx)

    start_value_cleanup(newy)
    interped_y = interp_0_coords(newy)
    
    head_centre['interped_x'] = interped_x
    head_centre['interped_y'] = interped_y
    
def start_value_cleanup(coords):
    # This is for when the starting value of the coords == 0; interpolation will not work on these coords until the first 0 
    #is changed. The 0 value is changed to the first non-zero value in the coords lists
    for index, value in enumerate(coords):
        working = 0
        if value > 0:
            start_value = value
            start_index = index
            working = 1
            break
    if working == 1:
        for x in range(start_index):
            coords[x] = start_value
            
def interp_0_coords(coords_list):
    #coords_list is one if the outputs of the get_x_y_data = a list of co-ordinate points
    for index, value in enumerate(coords_list):
        if value == 0:
            if coords_list[index-1] > 0:
                value_before = coords_list[index-1]
                interp_start_index = index-1
                #print('interp_start_index: ', interp_start_index)
                #print('interp_start_value: ', value_before)
                #print('')

        if index < len(coords_list)-1:
            if value ==0:
                if coords_list[index+1] > 0:
                    interp_end_index = index+1
                    value_after = coords_list[index+1]
                    #print('interp_end_index: ', interp_end_index)
                    #print('interp_end_value: ', value_after)
                    #print('')

                    #now code to interpolate over the values
                    try:
                        interp_diff_index = interp_end_index - interp_start_index
                    except UnboundLocalError:
#                         print('the first value in list is 0, use the function start_value_cleanup to fix')
                        break
                    #print('interp_diff_index is:', interp_diff_index)

                    new_values = np.linspace(value_before, value_after, interp_diff_index)
                    #print(new_values)

                    interp_index = interp_start_index+1
                    for x in range(interp_diff_index):
                        #print('interp_index is:', interp_index)
                        #print('new_value should be:', new_values[x])
                        coords_list[interp_index] = new_values[x]
                        interp_index +=1
        if index == len(coords_list)-1:
            if value ==0:
                for x in range(30):
                    coords_list[index-x] = coords_list[index-30]
                    #print('')
#     print('function exiting')
    return(coords_list)

def SaveFig(file_name,figure_dir):
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(figure_dir + file_name, bbox_inches='tight')
    plt.close()

def SaveFig_noclose(file_name,figure_dir):
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(figure_dir + file_name, bbox_inches='tight')

def convolve_movmean(y,N):
    y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
    return y_smooth

# Function to find the closest index using np.searchsorted
def find_closest_indices(timestamps, target_times):
    indices = np.searchsorted(timestamps, target_times)
    indices = np.clip(indices, 1, len(timestamps) - 1)  # Ensure indices are within bounds
    
    # Compare target_time with its neighbors to find the closest
    left_indices = indices - 1
    right_indices = indices
    left_diffs = np.abs(timestamps[left_indices] - target_times)
    right_diffs = np.abs(timestamps[right_indices] - target_times)
    
    return np.where(left_diffs <= right_diffs, left_indices, right_indices)