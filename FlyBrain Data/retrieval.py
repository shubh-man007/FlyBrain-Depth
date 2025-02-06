from utils import load_client, get_valid_rois, get_visual_rois, get_neuron_data, get_neuron_ids, fetch_all_connections, save_data

def main():
    client = load_client()
  
    valid_rois = get_valid_rois(client, mark_primary=True, format='dict')
    print("Valid ROIs:", valid_rois)

    visual_rois = get_visual_rois(valid_rois)
    print("Visual ROIs:", visual_rois)
    
    neuron_df, roi_counts_df = get_neuron_data(client, visual_rois, status='Traced', max_neurons=5000)
    print(f"Retrieved {len(neuron_df)} neurons.")
    
    neuron_ids = get_neuron_ids(neuron_df)
    print(f"Number of neuron IDs: {len(neuron_ids)}") 
    conn_df = fetch_all_connections(neuron_ids, batch_size=100)
    save_data(neuron_df, conn_df,
              neuron_features_filename="filtered_neurons.csv",
              neuron_all_features_filename="filtered_neurons_all_features.csv",
              connections_filename="filtered_cells.csv",
              weight_threshold=10)

if __name__ == '__main__':
    main()
