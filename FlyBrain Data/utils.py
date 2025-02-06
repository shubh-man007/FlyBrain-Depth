import os
import pandas as pd
from neuprint import Client, fetch_neurons, fetch_adjacencies, fetch_roi_hierarchy, NeuronCriteria as NC
from dotenv import load_dotenv

def load_client():
    """
    Load the environment variables and create a neuprint Client.
    """
    load_dotenv()
    token = os.getenv('token')
    if not token:
        raise ValueError("No token found in environment variables. Please set the token in your .env file.")
    client = Client('https://neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=token)
    return client

def get_valid_rois(client, mark_primary=True, format='dict'):
    """
    Retrieve the ROI hierarchy from neuprint and return a list of valid ROI names.
    """
    roi_hierarchy = fetch_roi_hierarchy(client, mark_primary=mark_primary, format=format)
    if format == 'dict':
        valid_rois = list(roi_hierarchy.keys())
    else:
        valid_rois = []  
    return valid_rois

def get_visual_rois(valid_rois, desired_rois = ["LO(R)", "LO(L)", "ME(R)", "ME(L)", "LOP(R)", "LOP(L)"]):
    """
    From a list of valid ROIs, select those corresponding to the visual system.
    """
    visual_rois = [roi for roi in desired_rois if roi in valid_rois]
    return visual_rois

def get_neuron_data(client, visual_rois, status='Traced', max_neurons=5000):
    """
    Fetch neurons matching the criteria (visual ROIs and status) from neuprint.
    If there are more than max_neurons, randomly sample.
    Returns:
        neuron_df: DataFrame with neuron properties.
        roi_counts_df: DataFrame with per-ROI synapse counts.
    """
    neuron_criteria = NC(rois=visual_rois, status=status)
    neuron_df, roi_counts_df = fetch_neurons(neuron_criteria)
    if len(neuron_df) > max_neurons:
        neuron_df = neuron_df.sample(n=max_neurons, random_state=42)
    return neuron_df, roi_counts_df

def get_neuron_ids(neuron_df):
    """
    Return a list of neuron bodyIds from the neuron DataFrame.
    """
    return neuron_df['bodyId'].tolist()

def fetch_all_connections(neuron_ids, batch_size=100):
    """
    Fetch downstream synaptic connections for the given neuron_ids in batches.
    Returns a DataFrame of all connections.
    """
    all_connections = []
    for i in range(0, len(neuron_ids), batch_size):
        batch = neuron_ids[i:i + batch_size]
        print(f"Fetching batch {i // batch_size + 1}/{len(neuron_ids) // batch_size + 1}...")
        try:
            _, conn_df = fetch_adjacencies(batch, None)
            all_connections.append(conn_df)
        except Exception as e:
            print(f"Error fetching batch {i // batch_size + 1}: {e}")
    if all_connections:
        conn_df = pd.concat(all_connections, ignore_index=True)
    else:
        conn_df = pd.DataFrame()  
    return conn_df

def save_data(neuron_df, conn_df, 
              neuron_features_filename="filtered_neurons.csv", 
              neuron_all_features_filename="filtered_neurons_all_features.csv",
              connections_filename="filtered_cells.csv", weight_threshold=10):
    """
    Save the retrieved data to CSV files. For connection data, only save rows with weight > weight_threshold.
    """
    if not conn_df.empty:
        conn_df = conn_df[conn_df['weight'] > weight_threshold]
        conn_df.to_csv(connections_filename, index=False)
        print(f"Saved connections to {connections_filename}")
    else:
        print("No connection data to save.")
        
    neuron_features = neuron_df[['bodyId', 'type', 'pre', 'post', 'size']]
    neuron_features.to_csv(neuron_features_filename, index=False)
    neuron_df.to_csv(neuron_all_features_filename, index=False)
    print(f"Saved neuron features to {neuron_features_filename} and {neuron_all_features_filename}")
