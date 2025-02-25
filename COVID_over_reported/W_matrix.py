import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform

# Mandatory Vaccination Laws (for public entity mandates) Across 50 States
# 1 if required; 0.5 if limits mandates; 0 if no mandates
# The data is according to: 
# https://nashp.org/state-tracker/state-efforts-to-ban-or-enforce-covid-19-vaccine-mandates-and-passports/


state_mandates = {
    'AK': 0.5, 'AL': 0.5, 'AR': 0.5, 'AZ': 0.5, 'CA': 0, 'CO': 0, 'CT': 0,
    'DE': 0, 'FL': 0.5, 'GA': 0.5, 'HI': 0, 'IA': 0.5, 'ID': 0.5, 'IL': 0,
    'IN': 0.5, 'KS': 0.5, 'KY': 0, 'LA': 0.5, 'MA': 0, 'MD': 0, 'ME': 1,
    'MI': 0.5, 'MN': 0, 'MO': 0.5, 'MS': 0.5, 'MT': 0.5, 'NC': 0.5, 'ND': 0.5,
    'NE': 0, 'NH': 0.5, 'NJ': 0, 'NM': 0, 'NV': 0, 'NY': 0, 'OH': 0, 'OK': 0.5,
    'OR': 0, 'PA': 1, 'RI': 1, 'SC': 0.5, 'SD': 0.5, 'TN': 0.5, 'TX': 0.5,
    'UT': 0.5, 'VA': 0.5, 'VT': 0, 'WA': 0, 'WI': 0, 'WV': 0.5, 'WY': 0.5
}

# COVID Vaccination Rates Across 50 States
# The data is according to the CDC website: 
# https://www.cdc.gov/covidvaxview/interactive/adults.html
# https://usafacts.org/visualizations/covid-vaccine-tracker-states/
 
state_vaccination_rates = {
    'AK': 0.72, 'AL': 0.64, 'AR': 0.68, 'AZ': 0.76, 'CA': 0.85, 'CO': 0.82, 'CT': 0.95,
    'DE': 0.86, 'FL': 0.81, 'GA': 0.67, 'HI': 0.90, 'IA': 0.69, 'ID': 0.63, 'IL': 0.78,
    'IN': 0.63, 'KS': 0.74, 'KY': 0.67, 'LA': 0.62, 'MA': 0.95, 'MD': 0.90, 'ME': 0.94,
    'MI': 0.68, 'MN': 0.77, 'MO': 0.68, 'MS': 0.61, 'MT': 0.67, 'NC': 0.89, 'ND': 0.68,
    'NE': 0.72, 'NH': 0.84, 'NJ': 0.93, 'NM': 0.91, 'NV': 0.77, 'NY': 0.92, 'OH': 0.64,
    'OK': 0.73, 'OR': 0.80, 'PA': 0.88, 'RI': 0.95, 'SC': 0.69, 'SD': 0.81, 'TN': 0.63,
    'TX': 0.75, 'UT': 0.74, 'VA': 0.88, 'VT': 0.95, 'WA': 0.83, 'WI': 0.73, 'WV': 0.66,
    'WY': 0.59
}


# Educational Level Across 50 States
# Calculates as the percentage of the population with a bachelor's degree or higher
# The data is according to the Wikipedia page:
# https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_educational_attainment?utm_source=chatgpt.com

state_educational_level = {
    'AK': 0.3279, 'AL': 0.2743, 'AR': 0.2527, 'AZ': 0.3243, 'CA': 0.3619, 'CO': 0.4442, 'CT': 0.4213,
    'DE': 0.3562, 'FL': 0.3316, 'GA': 0.3463, 'HI': 0.3530, 'IA': 0.3054, 'ID': 0.3072, 'IL': 0.3714,
    'IN': 0.2888, 'KS': 0.3540, 'KY': 0.2698, 'LA': 0.2645, 'MA': 0.5062, 'MD': 0.4855, 'ME': 0.3599,
    'MI': 0.3167, 'MN': 0.3890, 'MO': 0.3172, 'MS': 0.2478, 'MT': 0.3480, 'NC': 0.3491, 'ND': 0.3174,
    'NE': 0.3445, 'NH': 0.4098, 'NJ': 0.4310, 'NM': 0.3005, 'NV': 0.2757, 'NY': 0.4090, 'OH': 0.3072,
    'OK': 0.2792, 'OR': 0.3633, 'PA': 0.3654, 'RI': 0.3650, 'SC': 0.3153, 'SD': 0.3167, 'TN': 0.3048,
    'TX': 0.3312, 'UT': 0.3681, 'VA': 0.4181, 'VT': 0.4444, 'WA': 0.4097, 'WI': 0.3254, 'WV': 0.2412,
    'WY': 0.2924
}


# Political Party Affiliation Across 50 States
# 1 if Republican; 0.5 if Swing; 0 if Democrat

state_political_party_affiliation = {
    'AK': 1, 'AL': 1, 'AR': 1, 'AZ': 0.5, 'CA': 0, 'CO': 0, 'CT': 0, 'DE': 0, 'FL': 1, 
    'GA': 0.5, 'HI': 0, 'IA': 1, 'ID': 1, 'IL': 0, 'IN': 1, 'KS': 1, 'KY': 1, 'LA': 1, 
    'MA': 0, 'MD': 0, 'ME': 0.5, 'MI': 0.5, 'MN': 0, 'MO': 1, 'MS': 1, 'MT': 1, 'NC': 0.5, 
    'ND': 1, 'NE': 1, 'NH': 0.5, 'NJ': 0, 'NM': 0, 'NV': 0.5, 'NY': 0, 'OH': 1, 'OK': 1, 
    'OR': 0, 'PA': 0.5, 'RI': 0, 'SC': 1, 'SD': 1, 'TN': 1, 'TX': 1, 'UT': 1, 'VA': 0, 
    'VT': 0, 'WA': 0, 'WI': 0.5, 'WV': 1, 'WY': 1
}



# Reported Number of Cases Across 50 States
# The data is according to the CDC website:
# https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/cases-in-us.html
state_reported_number_of_cases = {
    'AK': 532, 'AL': 1474, 'AR': 1966, 'AZ': 5081, 'CA': 27236, 'CO': 11019, 'CT': 3038, 'DE': 602, 'FL': 9743, 
    'GA': 4021, 'HI': 1045, 'IA': 1837, 'ID': 1523, 'IL': 6514, 'IN': 3581, 'KS': 1494, 'KY': 4526, 'LA': 1523, 
    'MA': 6651, 'MD': 4353, 'ME': 1259, 'MI': 15025, 'MN': 7012, 'MO': 3204, 'MS': 1059, 'MT': 948, 'NC': 5135, 
    'ND': 476, 'NE': 1395, 'NH': 1401, 'NJ': 7362, 'NM': 1597, 'NV': 1813, 'NY': 12031, 'OH': 7250, 'OK': 2259, 
    'OR': 3486, 'PA': 7005, 'RI': 908, 'SC': 2037, 'SD': 645, 'TN': 3935, 'TX': 12255, 'UT': 1351, 'VA': 5360, 
    'VT': 755, 'WA': 7256, 'WI': 4339, 'WV': 846, 'WY': 326
}

state_reported_proportion = {
    'AK': 0.0007254, 'AL': 0.00029338, 'AR': 0.00065283, 'AZ': 0.00071048, 'CA': 0.00068885, 'CO': 0.00190848,
    'CT': 0.0008425, 'DE': 0.00060811, 'FL': 0.00045236, 'GA': 0.00037538, 'HI': 0.00071808, 'IA': 0.0005758,
    'ID': 0.00082812, 'IL': 0.00050841, 'IN': 0.00052774, 'KS': 0.00050853, 'KY': 0.00100448, 'LA': 0.00032698,
    'MA': 0.0009461, 'MD': 0.00070724, 'ME': 0.00092413, 'MI': 0.00149097, 'MN': 0.00122878, 'MO': 0.00051868,
    'MS': 0.00035762, 'MT': 0.00087436, 'NC': 0.00049189, 'ND': 0.00061097, 'NE': 0.00071119, 'NH': 0.00101704,
    'NJ': 0.00079255, 'NM': 0.00075418, 'NV': 0.00058397, 'NY': 0.00059556, 'OH': 0.00061444, 'OK': 0.00057055,
    'OR': 0.0008227, 'PA': 0.00053873, 'RI': 0.00082743, 'SC': 0.00039797, 'SD': 0.00072744, 'TN': 0.0005694,
    'TX': 0.00042048, 'UT': 0.00041295, 'VA': 0.00062099, 'VT': 0.00117404, 'WA': 0.00094169, 'WI': 0.00073621,
    'WV': 0.00047165, 'WY': 0.00056514
}


data_dict = {
    'State': state_mandates.keys(),
    'Mandatory_Vaccination_Laws': state_mandates.values(),
    'Vaccination_Rates': state_vaccination_rates.values(),
    'Educational_Level': state_educational_level.values(),
    'Political_Party_Affiliation': state_political_party_affiliation.values(),
    'Reported_Number_of_Cases': state_reported_number_of_cases.values(),
    'Reported_Proportion': state_reported_proportion.values()
}

state_data_df = pd.DataFrame(data_dict)
output_path = '/Users/a59611/code/Graph_Prediction/python_version/data/state_graph_data_revised.csv'
state_data_df.to_csv(output_path, index=False)




# The graph data has now been saved
# Now we need to create the weight matrix

def categorical_similarity(categorical_data):
    column_1 = categorical_data[:, 0]
    column_2 = categorical_data[:, 1]

    similarity_matrix_1 = 1 - np.abs(column_1[:, np.newaxis] - column_1[np.newaxis, :])
    similarity_matrix_2 = 1 - np.abs(column_2[:, np.newaxis] - column_2[np.newaxis, :])
    return similarity_matrix_1, similarity_matrix_2

def numerical_similarity(numerical_data):
    X_min = np.min(numerical_data)
    X_max = np.max(numerical_data)
    X_normalized = (numerical_data - X_min) / (X_max - X_min)
    X_normalized = np.array(X_normalized)
    similarity_matrix = 1 - np.abs(X_normalized[:, np.newaxis] - X_normalized[np.newaxis, :])
    return similarity_matrix

categorical_cols = ['Mandatory_Vaccination_Laws', 'Political_Party_Affiliation']
categorical_data = state_data_df[categorical_cols].values
categorical_sim_1, categorical_sim_2 = categorical_similarity(categorical_data)


num_sim_1 = numerical_similarity(state_data_df['Reported_Proportion'])
num_sim_2 = numerical_similarity(state_data_df['Vaccination_Rates'])
num_sim_3 = numerical_similarity(state_data_df['Educational_Level'])

final_similarity_matrix = (
    0.6 * num_sim_1 + 
    0.1 * num_sim_2 + 
    0.1 * num_sim_3 + 
    0.1 * categorical_sim_1 + 
    0.1 * categorical_sim_2
)

for i, state in enumerate(state_data_df['State']):
    # Get all similarities except self-similarity (diagonal)
    similarities = np.concatenate([final_similarity_matrix[i,:i], final_similarity_matrix[i,i+1:]])
    min_similarity = np.min(similarities)
    print(f"State {state}: Minimum similarity = {min_similarity:.4f}")


np.fill_diagonal(final_similarity_matrix, 0)


similarity_df = pd.DataFrame(final_similarity_matrix, 
                           index=state_data_df['State'], 
                           columns=state_data_df['State'])
output_path = '/Users/a59611/code/Graph_Prediction/python_version/data/W_matrix.csv'
similarity_df.to_csv(output_path)
print(f"Weight matrix saved to {output_path}")