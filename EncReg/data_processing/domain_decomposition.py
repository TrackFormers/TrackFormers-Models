import numpy as np 
import pandas as pd

def make_bins(data, parameter, bin_edges, overlap):
    '''
    Function that creates bins in parameter, using specified bin_edges and overlap.
    It creates additional binary columns in the data dataframe, denoting whether a 
    hit is in a certain bin. The rotational invariance problem of phi has been taken
    into account in the bin creation.
    '''
    num_bins = len(bin_edges) - 1
    bin_names = [f'{parameter}_bin{i}' for i in range(1, num_bins + 1)]

    for i in range(num_bins):
        lower_threshold = bin_edges[i] - overlap
        upper_threshold = bin_edges[i + 1] + overlap
        if i == 0 and parameter == 'phi':
            data[bin_names[i]] = np.logical_or(data[parameter] < upper_threshold, data[parameter] > np.pi - overlap)
        elif i == num_bins-1 and parameter == 'phi':
            data[bin_names[i]] = np.logical_or(data[parameter] >= lower_threshold, data[parameter] < -np.pi + overlap)
        else:
            data[bin_names[i]] = np.logical_and(data[parameter] >= lower_threshold, data[parameter] < upper_threshold)


def split_event(data, event_id):
    '''
    Function for accomplishing domain decomposition of a single event.
    '''
    overlap_theta = overlap_phi = 0.

    # Calculate theta and phi of each hit
    p = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['theta'] = np.arccos(data['z']/p)
    data['phi'] = np.arctan2(data['y'], data['x'])

    # Create the bins of theta (values currently chosen based on distribution of theta)
    # theta_bin_edges = np.array([0, 0.5, 2.5, np.pi]) # 3 bins
    theta_bin_edges = np.array([0, 0.3, 1.45, 2.8, np.pi]) # 4 bins
    make_bins(data, 'theta', theta_bin_edges, overlap_theta)

    # Create the bins of phi (values currently chosen based on distribution of phi)
    # phi_bin_edges = np.array([-np.pi, -1, 1, np.pi]) # 3 bins
    phi_bin_edges = np.array([-np.pi, -1.57, 0, 1.57, np.pi]) # 4 bins
    make_bins(data, 'phi', phi_bin_edges, overlap_phi)

    # plt.hist(data['phi'].values, bins=[-np.pi, -1.57, 0, 1.57, np.pi])
    # plt.show()
    
    # Create the bins of theta-phi combinations
    classes = []
    for phi_bin in range(1, len(phi_bin_edges)):
        for theta_bin in range(1, len(theta_bin_edges)):
            class_name = f'class_{phi_bin}_{theta_bin}'
            phi_condition = data[f'phi_bin{phi_bin}']
            theta_condition = data[f'theta_bin{theta_bin}']
            data[class_name] = np.logical_and(phi_condition, theta_condition)
            classes.append(class_name)

    # For every row in the dataset, check which "classes" it got assigned to
    # and add all of them to a new list of rows. Necessary step since the overlap
    # might lead to the same hit belonging in e.g. 3 different "classes" and in 
    # that case we want that row to be duplicated with a different event ID 3 times
    event_class = []
    new_rows = []
    for i, class_name in enumerate(classes):
        for _, row in data[data[class_name]].iterrows():
            event_class.append(f"{event_id}_{i}")
            new_rows.append(row)

    # Create the new dataframe with the newly composed lists
    new_data = pd.DataFrame(new_rows, columns=data.columns)
    new_data['event_class'] = event_class
    print(new_data['event_class'].value_counts())

    # Evaluate the split by calculating its "efficiency" score
    evaluate_split_event(data, new_data)

    # Important: event_class is now the new event_id to follow for separation 
    # between events in other scripts!
    return new_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x", "event_id", "event_class"]] 


def evaluate_split_event(old_data, data):
    '''
    Function meant to evaluate the efficiency of the domain decomposition.
    '''
    # Make a dictionary with all particle_ids and the number of hits they have
    all_particle_ids = old_data['particle_id'].unique().tolist()
    particle_dict = {}
    for part_id in all_particle_ids:
        indices = old_data['particle_id'] == part_id
        particle_dict[part_id] = len(old_data[indices])

    # For every track/particle, calculate the ratio that shows how well represented 
    # it is by the class it's in: (hits of that particle in this class)/(all hits for that particle).
    # Where we go over each class and each particle, so that if a particle is in multiple
    # classes, we take the one it's best represented in
    class_ids = data['event_class'].unique().tolist()
    portions = {}
    for cl_id in class_ids:
        indices = data['event_class'] == cl_id
        hits_data = data[indices]
        particle_ids = hits_data['particle_id'].unique().tolist()
        for part_id in particle_ids:
            indices = hits_data['particle_id'] == part_id
            nr_hits_in_class = len(hits_data[indices])
            if part_id not in portions.keys():
                portions[part_id] = nr_hits_in_class/particle_dict[part_id]
            else:
                if nr_hits_in_class/particle_dict[part_id] > portions[part_id]:
                    portions[part_id] = nr_hits_in_class/particle_dict[part_id]

    portions_arr = np.array(list(portions.values()))
    print(portions_arr)
    print("Average efficiency:", portions_arr.mean())
    print("Efficiency standard dev:", portions_arr.std())