import pandas as pd
import numpy as np
from datetime import time, datetime, timedelta

filepath = "../datasets/dataset-3.csv"


def calculate_distance_matrix(filepath)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    df = pd.read_csv(filepath)

    

    # Create a set of unique IDs
    unique_ids = sorted(set(df['id_start'].unique()).union(df['id_end'].unique()))

    # Initialize an empty DataFrame for the distance matrix
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)

    # Set diagonal values to 0
    np.fill_diagonal(distance_matrix.values, 0)

    # Populate the matrix with known distances
    df_pivot = df.pivot(index='id_start', columns='id_end', values='distance')
    distance_matrix.update(df_pivot)
    distance_matrix.update(df_pivot.T)

    # Fill in missing values by calculating cumulative distances
    while pd.isna(distance_matrix.values).any():
        missing_indices = np.argwhere(pd.isna(distance_matrix.values))
        for idx in missing_indices:
            i, j = idx
            k_values = distance_matrix.iloc[i, :] + distance_matrix.iloc[:, j]
            valid_k_values = k_values.dropna()
            if not valid_k_values.empty:
                distance_matrix.iloc[i, j] = distance_matrix.iloc[j, i] = valid_k_values.min()

    return distance_matrix

calculate_distance_matrix(filepath)

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    # Extract unique IDs from the index of the distance matrix
    unique_ids = distance_matrix.index

    # Initialize lists to store unrolled data
    id_start_list, id_end_list, distance_list = [], [], []

    # Iterate through unique ID pairs and populate lists
    for id_start in unique_ids:
        for id_end in unique_ids:
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                id_start_list.append(id_start)
                id_end_list.append(id_end)
                distance_list.append(distance)

    # Create a DataFrame from the lists
    unrolled_df = pd.DataFrame({
        'id_start': id_start_list,
        'id_end': id_end_list,
        'distance': distance_list
    })

    return unrolled_df    
    


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    # Filter DataFrame for the specified reference value
    reference_df = unrolled_dataframe[unrolled_dataframe['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold_lower = average_distance - 0.1 * average_distance
    threshold_upper = average_distance + 0.1 * average_distance

    # Filter DataFrame for values within the threshold range
    within_threshold_df = unrolled_dataframe[
        (unrolled_dataframe['distance'] >= threshold_lower) &
        (unrolled_dataframe['distance'] <= threshold_upper)
    ]

    # Extract and sort unique 'id_start' values within the threshold range
    result_ids = sorted(within_threshold_df['id_start'].unique())
    return result_ids
    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    # Create new columns for toll rates based on vehicle types
    unrolled_dataframe['moto'] = 0.8 * unrolled_dataframe['distance']
    unrolled_dataframe['car'] = 1.2 * unrolled_dataframe['distance']
    unrolled_dataframe['rv'] = 1.5 * unrolled_dataframe['distance']
    unrolled_dataframe['bus'] = 2.2 * unrolled_dataframe['distance']
    unrolled_dataframe['truck'] = 3.6 * unrolled_dataframe['distance']

    # Remove the 'distance' column
    unrolled_dataframe = unrolled_dataframe.drop(columns=['distance'])

    return unrolled_dataframe
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define time ranges
    weekday_time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]

    weekend_time_range = (time(0, 0, 0), time(23, 59, 59), 0.7)

    # Create lists of unique values
    unique_ids_start = result_dataframe.index.unique()
    unique_ids_end = result_dataframe.index.unique()

    # Initialize an empty DataFrame to store the results
    result_with_time_based_rates = pd.DataFrame(index=result_dataframe.index)

    # Iterate over unique (id_start, id_end) pairs
    for id_start in unique_ids_start:
        for id_end in unique_ids_end:
            # Select rows corresponding to the current (id_start, id_end) pair
            subset = result_dataframe.loc[(id_start, id_end), :]

            # Initialize an empty DataFrame to store the time-based rates
            time_based_rates = pd.DataFrame(index=subset.index)

            # Check if it's a weekday or weekend
            is_weekday = datetime.now().weekday() < 5

            # Iterate over time ranges
            for start_time, end_time, discount_factor in (weekday_time_ranges if is_weekday else [weekend_time_range]):
                # Select rows within the current time range
                time_range_subset = subset.between_time(start_time, end_time)

                # Apply the discount factor to vehicle columns
                time_range_subset.iloc[:, 3:] *= discount_factor

                # Concatenate with the time_based_rates DataFrame
                time_based_rates = pd.concat([time_based_rates, time_range_subset])

            # Concatenate the time_based_rates DataFrame with the result_with_time_based_rates DataFrame
            result_with_time_based_rates = pd.concat([result_with_time_based_rates, time_based_rates])

    return result_with_time_based_rates

calculate_time_based_toll_rates(unrolled_dataframe)
return df
