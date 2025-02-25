import pandas as pd

# # Reloading the newly uploaded 2023 VAERS files
# vaers_data_2023_path = '/Users/a59611/code/Graph_Prediction/python_version/data/2022VAERSDATA.csv'
# vaers_vax_2023_path = '/Users/a59611/code/Graph_Prediction/python_version/data/2022VAERSVAX.csv'

# # Read the files with potential encoding issues handled
# vaers_data_2023 = pd.read_csv(vaers_data_2023_path, encoding='ISO-8859-1')
# vaers_vax_2023 = pd.read_csv(vaers_vax_2023_path, encoding='ISO-8859-1')

# # Filter VAERSVAX for rows where VAX_TYPE contains "COVID" or starts with "COVID"
# covid_vax_ids_2023 = vaers_vax_2023[vaers_vax_2023['VAX_TYPE'].str.contains('^COVID', case=False, na=False)]['VAERS_ID'].unique()

# # Filter VAERSDATA for rows with matching VAERS_IDs
# filtered_vaers_data_2023 = vaers_data_2023[vaers_data_2023['VAERS_ID'].isin(covid_vax_ids_2023)]

# # Save the filtered data to a new CSV file
# output_path_2023 = '/Users/a59611/code/Graph_Prediction/python_version/data/filtered_vaers_data_2022_final.csv'
# filtered_vaers_data_2023.to_csv(output_path_2023, index=False, encoding='utf-8')


# filtered_vaers_data_2022 = pd.read_csv('/Users/a59611/code/Graph_Prediction/python_version/data/filtered_vaers_data_2022_cleaned.csv')
# filtered_vaers_data_2022_cleaned = filtered_vaers_data_2022.dropna(subset=['STATE', 'AGE_YRS'])

# valid_states = [
#     'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL',
#     'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
#     'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
#     'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
# ]

# # Filter the data to keep only rows with valid states
# filtered_valid_states = filtered_vaers_data_2022_cleaned[filtered_vaers_data_2022_cleaned['STATE'].isin(valid_states)]

# # Save the cleaned dataset
# output_cleaned_path = '/Users/a59611/code/Graph_Prediction/python_version/data/cleaned_filtered_vaers_data_2022.csv'
# filtered_valid_states.to_csv(output_cleaned_path, index=False)


# file1 = pd.read_csv('/Users/a59611/code/Graph_Prediction/python_version/data/cleaned_filtered_vaers_data_2022.csv')
# file2 = pd.read_csv('/Users/a59611/code/Graph_Prediction/python_version/data/cleaned_filtered_vaers_data.csv')
# combined_data = pd.concat([file1.iloc[0:], file2.iloc[1:]], ignore_index=True)

# # Save the combined dataset
# output_combined_path = '/Users/a59611/code/Graph_Prediction/python_version/data/combined_vaers_data_new.csv'
# combined_data.to_csv(output_combined_path, index=False)


# # File path
# file_filtered = '/Users/a59611/code/Graph_Prediction/python_version/data/final_data.csv'
# # Read the file
# filtered_data = pd.read_csv(file_filtered)

# print(len(filtered_data['STATE'].unique()))
# # Count the number of records for each state
# state_counts = filtered_data['STATE'].value_counts()
# print("\nNumber of records per state:")
# print(state_counts)


# zero_values = [
#     "0", "-", "—", "-NA-", "-NIL-", "-none-", "?", "(none)", "#NAME?", "unknown", "No", "none", "NO", "NONE", "None."
# ]
# # Add blank (empty string) and strings starting with certain patterns
# starts_with_zero = ["None", "none", "zero", "Zero!", "No", "no"]

# # Assign new values to the 'CUR_ILL' column
# data['CUR_ILL'] = data['CUR_ILL'].apply(
#     lambda x: 0 if (
#         pd.isna(x) or  # Handle NaN values
#         str(x).strip() in zero_values or  # Exact match in zero_values
#         any(str(x).strip().startswith(prefix) for prefix in starts_with_zero)  # Matches prefixes
#     ) else 1
# )

# data.loc[data['NUMDAYS'] >= 300, 'NUMDAYS'] = 300
# data.to_csv('/Users/a59611/code/Graph_Prediction/python_version/data/final_data_cleaned.csv', index=False)

# data.to_csv('/Users/a59611/code/Graph_Prediction/python_version/data/final_data_cleaned.csv', index=False)


# Now we create the X data, for predicting log n.
data = pd.read_csv('/Users/a59611/code/Graph_Prediction/python_version/data/VAERS_data.csv')

valid_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL',
    'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
    'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
    'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]
states = sorted(valid_states)  # Sort the list of 50 states alphabetically
state_to_index = {state: index for index, state in enumerate(states)}

X_matrix = []

for index, state in enumerate(states):
    state_data = data[data['STATE'] == state]
    age_0_24 = len(state_data[state_data['AGE_YRS'] < 25])/len(state_data)
    age_25_44 = len(state_data[(state_data['AGE_YRS'] >= 25) & (state_data['AGE_YRS'] < 50)]) / len(state_data)
    age_45_64 = len(state_data[(state_data['AGE_YRS'] >= 50) & (state_data['AGE_YRS'] < 75)]) / len(state_data)
    L_threaten = len(state_data[state_data['L_THREAT'] == 'Y'])/len(state_data)
    Hospitalized = len(state_data[state_data['HOSPITAL'] == 'Y'])/len(state_data)
    X_stay = len(state_data[state_data['X_STAY'] == 'Y'])/len(state_data)
    Recovered = len(state_data[state_data['RECOVD'] == 'Y'])/len(state_data)
    Num_days = state_data['NUMDAYS'].mean()
    CUR_ILL = state_data['CUR_ILL'].mean()
    X_matrix.append([1, age_0_24, age_25_44, age_45_64, L_threaten, Hospitalized, X_stay, Recovered, Num_days, CUR_ILL])

columns = [
    'Const','Age_0_24', 'Age_25_44', 'Age_45_64', 'L_Threaten', 'Hospitalized',
    'X_Stay', 'Recovered', 'Num_Days', 'CUR_ILL'
]

X_df = pd.DataFrame(X_matrix, index=states, columns=columns)


# One more column is added for X_df: population in the state
state_populations = {
    "AK": 733391, "AL": 5024279, "AR": 3011524, "AZ": 7151502, "CA": 39538223, "CO": 5773714,
    "CT": 3605944, "DE": 989948, "FL": 21538187, "GA": 10711908, "HI": 1455271, "IA": 3190369,
    "ID": 1839106, "IL": 12812508, "IN": 6785528, "KS": 2937880, "KY": 4505836, "LA": 4657757,
    "MA": 7029917, "MD": 6154913, "ME": 1362359, "MI": 10077331, "MN": 5706494, "MO": 6177224,
    "MS": 2961279, "MT": 1084225, "NC": 10439388, "ND": 779094, "NE": 1961504, "NH": 1377529,
    "NJ": 9288994, "NM": 2117522, "NV": 3104614, "NY": 20201249, "OH": 11799448, "OK": 3959353,
    "OR": 4237256, "PA": 13002700, "RI": 1097379, "SC": 5118425, "SD": 886667, "TN": 6910840,
    "TX": 29145505, "UT": 3271616, "VA": 8631393, "VT": 643077, "WA": 7705281, "WI": 5893718,
    "WV": 1793716, "WY": 576851
}
X_df['Population'] = [state_populations[state] for state in states]

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

X_df['Vaccination_Rate'] = [state_vaccination_rates[state] for state in states]


y_counts = data['STATE'].value_counts()
y_df = pd.DataFrame(index=states)
y_df['Count'] = [y_counts.get(state, 0) for state in states]


X_df.to_csv('/Users/a59611/code/Graph_Prediction/python_version/data/X_matrix_2.csv')
y_df.to_csv('/Users/a59611/code/Graph_Prediction/python_version/data/y_matrix.csv')