import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import geopandas as gpd
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.affinity import scale, translate

m = 50
W = pd.read_csv('COVID_over_reported/W_matrix.csv')
X = pd.read_csv('COVID_over_reported/X_matrix.csv')
y = pd.read_csv('COVID_over_reported/y_matrix.csv')

W_values = W.iloc[0:, 1:].astype(float)
X_values = X.iloc[0:, 1:].astype(float)
y_values = y.iloc[0:, 1].astype(float)

idx_codes = X.iloc[0:, 0].astype(str)
results_dir = 'COVID_over_reported/results_plots'

plt.rcParams.update({
    "font.size": 12,           # Larger font size
    "figure.figsize": (6, 4),  # Consistent figure size
    "axes.grid": True,         # Add grid to plots
    "grid.alpha": 0.5,         # Transparency for grid
    "savefig.dpi": 300,        # High-resolution output
    "lines.linewidth": 2,      # Thicker lines
})

D = pd.DataFrame(0.0, index=W_values.index, columns=W_values.columns)
np.fill_diagonal(D.values, W_values.sum(axis=1).values)


L = D - W_values

epsilon = 1e-6
L = L + epsilon * np.eye(L.shape[0])


L = L.values
D = D.values
W_values = W_values.values
y_values = y_values.values
X_values = X_values.values


X_values[:,-1] = np.log(X_values[:,-1])

reported_proportion = y_values/np.exp(X_values[:,-1])

selected_idx = np.where(((reported_proportion >= 0.0004) & (reported_proportion <= 0.0008)))[0]
selected_codes = idx_codes[selected_idx]


m = 50 # number of states
lambda_1 = 0.005
lambda_2 = 0.9
outer_iter = 5000


H = np.eye(m) - X_values @ np.linalg.pinv(X_values.T @ X_values) @ X_values.T
H = H + epsilon * np.eye(H.shape[0])

# Initializations
n_est = y_values
p_est_1 = np.zeros(m)+1.0

step = 0.4
inner_iter = 1
loss = np.zeros(outer_iter)

v = cp.Variable(m)
u = cp.Variable(m)


log_y = np.log(y_values)
objective = cp.norm(log_y - v - u, 2)**2 + lambda_1 * cp.quad_form(v, L) + lambda_2 * cp.quad_form(u, H)

constraints = [
    log_y >= u,  # np.log(y_values) >= u
    v >= 0,      # 0 <= v
    v <= np.log(2)  # v <= log 2
]
problem = cp.Problem(cp.Minimize(objective), constraints)
problem.solve()

p_est = np.exp(v.value)-1
n_est = np.exp(u.value)


selected_p_est = p_est[selected_idx]
selected_n_est = n_est[selected_idx]
selected_state_values = dict(zip(selected_codes, selected_p_est))
corrected_proportion = n_est/np.exp(X_values[:,-1])


plt.figure()
x1 = np.arange(1, m + 1)
x2 = np.arange(1, m + 1) - 0.25  # Shift left for observed y
plt.vlines(x1, 0, reported_proportion, colors='#F08080', label='Reported Proportion',linewidth=0.8)
plt.vlines(x2, 0, corrected_proportion, colors='#5A9BD3', label='Corrected Proportion',linewidth=0.8)
plt.scatter(x1, reported_proportion, color='#F08080', zorder=3, s=20)
plt.scatter(x2, corrected_proportion, color='#5A9BD3', zorder=3, s=20)
plt.xlabel("Node index")
plt.title(r"Reported vs. Corrected Proportions")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '{:.2f}'.format(x*1e3)))
plt.ylabel('Value ($\\times 10^{-3}$)')
plt.legend(loc="upper right", prop={'size': 8})
plt.savefig(results_dir+f"/reported_vs_corrected_proportion.png")

plt.figure()
x1 = np.arange(1, m + 1)
x3 = np.arange(1, m + 1) - 0.25  # Shift left for observed y
plt.vlines(x1, 0, n_est, colors='r', label='Estimated n',linewidth=0.8)
plt.vlines(x3, 0, y_values, colors='b', label='Observed y',linewidth=0.8)
plt.scatter(x1, n_est, color='r', zorder=3, s=20)
plt.scatter(x3, y_values, color='b', zorder=3, s=20)
plt.xlabel("Node index")
plt.ylabel("Value")
plt.title(r"Comparison of $\mathrm{n}$ and $\mathrm{n}_{\mathrm{est}}$")
plt.legend(loc="upper right", prop={'size': 8})
plt.savefig(results_dir+f"/n_comparison.png")

plt.figure()
x1 = np.arange(1, m+1)
plt.vlines(x1, 0, p_est, colors='#5A9BD3', label='Estimated p',linewidth=0.8)
plt.scatter(x1, p_est, color='#5A9BD3', zorder=3, s=20)
plt.xlabel("Node index")
plt.ylabel("Value")
plt.title(r"Reporting Bias Probability Estimation $\mathrm{p}_{\mathrm{est}}$")
plt.ylim(0, 1.0)  # Set y-axis limits from 0 to 0.8
plt.legend(loc="upper right", prop={'size': 8})
plt.savefig(results_dir+f"/p_comparison.png")
#plt.savefig(results_dir+f"/p_comparison_m={m}_lambda_={lambda_1},{lambda_2}_iter={outer_iter}_inner={inner_iter}.pdf")


us_states = gpd.read_file('COVID_over_reported/US_plot_data/ne_10m_admin_1_states_provinces.shp', engine='pyogrio')
us_states = us_states[us_states['iso_a2'] == 'US']

us_states = gpd.read_file('COVID_over_reported/US_plot_data/ne_10m_admin_1_states_provinces.shp', engine='pyogrio')
state_graph_data = pd.read_csv('COVID_over_reported/state_graph_data_revised.csv')
us_states = us_states[us_states['iso_a2'] == 'US']

state_abbreviations = [
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID',
    'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT',
    'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
    'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
]

state_name_to_abbr = {
    'Alaska': 'AK', 'Alabama': 'AL', 'Arkansas': 'AR', 'Arizona': 'AZ', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Iowa': 'IA', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Massachusetts': 'MA', 'Maryland': 'MD',
    'Maine': 'ME', 'Michigan': 'MI', 'Minnesota': 'MN', 'Missouri': 'MO', 'Mississippi': 'MS',
    'Montana': 'MT', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Nebraska': 'NE',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'Nevada': 'NV',
    'New York': 'NY', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Virginia': 'VA', 'Vermont': 'VT', 'Washington': 'WA',
    'Wisconsin': 'WI', 'West Virginia': 'WV', 'Wyoming': 'WY'
}

us_states = us_states[us_states['name'].isin(state_name_to_abbr.keys())]
us_states['state_abbr'] = us_states['name'].map(state_name_to_abbr)

# These nodes are ignored because their originally reported proportions are abnormally higher than those of other states.
ignored_nodes = [0, 5, 10, 21, 22]  # Indices for 'AK' 'CO', 'HI' 'MI', 'MN'

edu_level = state_graph_data['Educational_Level']
vaccine_level = state_graph_data['Vaccination_Rates']
us_states['p_est'] = np.nan
for idx, abbr in enumerate(state_abbreviations):
    if idx not in ignored_nodes:
        print((idx,abbr,p_est[idx]))
        us_states.loc[us_states['state_abbr'] == abbr, 'p_est'] = p_est[idx]
        us_states.loc[us_states['state_abbr'] == abbr, 'edu_level'] = edu_level[idx]
        us_states.loc[us_states['state_abbr'] == abbr, 'vaccine_level'] = vaccine_level[idx]

alaska = us_states[us_states['name'] == 'Alaska']
hawaii = us_states[us_states['name'] == 'Hawaii']
continental_us = us_states[~us_states['name'].isin(['Alaska', 'Hawaii'])]
us_adjusted = pd.concat([continental_us, alaska, hawaii])

cmap = mcolors.LinearSegmentedColormap.from_list(
    "CustomGradient", ["#FFFFB3", "#FFC857", "#42A5F5", "#00509E"]
)

cmap2 = mcolors.LinearSegmentedColormap.from_list(
    "CustomGradient", 
    [
        (0.00, "#FFFFB3"),  # Light Yellow
        (0.20, "#FFC857"),  # Peach
        (0.50, "#42A5F5"),  # Teal Blue
        (1.00, "#00509E")   # Deep Blue
    ]
)

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
alaska_geom = us_adjusted.loc[us_adjusted['name'] == 'Alaska', 'geometry'].copy()
hawaii_geom = us_adjusted.loc[us_adjusted['name'] == 'Hawaii', 'geometry'].copy()
alaska_geom = alaska_geom.apply(
    lambda g: scale(g, xfact=0.70, yfact=0.70)
)

alaska_geom = alaska_geom.apply(
    lambda g: translate(g, xoff=-35.0, yoff=-7.0)
)

hawaii_geom = hawaii_geom.apply(
    lambda g: scale(g, xfact=1.0, yfact=1.0, origin='center')
)

hawaii_geom = hawaii_geom.apply(
    lambda g: translate(g, xoff=30.0, yoff=10.0)
)

us_adjusted.loc[us_adjusted['name'] == 'Alaska', 'geometry'] = alaska_geom
us_adjusted.loc[us_adjusted['name'] == 'Hawaii', 'geometry'] = hawaii_geom

us_adjusted.plot(
    column='p_est',
    cmap=cmap,
    legend=True,
    legend_kwds={'shrink': 0.5},
    missing_kwds={"color": "grey", "label": "Ignored States"},
    ax=ax
)
ax.set_xlim([-160, -70])

ax.set_aspect('equal', adjustable='datalim')

plt.title("Estimated Reporting Bias Probability ($p_{est}$) by State", 
        fontsize=25, x=0.56, y=0.85)
plt.axis('off')

fig.savefig(results_dir + "/US_p_est_map_adjusted.png", bbox_inches='tight', dpi=300)
fig.savefig(results_dir + "/US_p_est_map_adjusted.pdf", bbox_inches='tight', dpi=300)



# fig2, ax2 = plt.subplots(1, 1, figsize=(15, 10))
# us_adjusted.plot(
#     column='vaccine_level',
#     cmap=cmap,
#     legend=True,
#     missing_kwds={"color": "grey", "label": "Ignored States"},
#     ax=ax2
# )
# ax2.set_xlim(-170, -70)
# ax2.set_ylim(40, 50) 
# ax2.set_aspect('equal', adjustable='datalim')
# plt.title("Vaccine Level by State", fontsize=25)
# plt.axis('off')
# fig2.savefig(results_dir + "/US_vaccine_level_map.png", bbox_inches='tight', dpi=300)



# fig3, ax3 = plt.subplots(1, 1, figsize=(15, 10))
# us_adjusted.plot(
#     column='edu_level',
#     cmap=cmap2,
#     legend=True,
#     missing_kwds={"color": "grey", "label": "Ignored States"},
#     ax=ax3
# )
# ax3.set_xlim(-170, -70)
# ax3.set_ylim(40, 50) 
# ax3.set_aspect('equal', adjustable='datalim')
# plt.title("Education Level by State", fontsize=25)
# plt.axis('off')
# fig3.savefig(results_dir + "/US_edu_level_map.png", bbox_inches='tight', dpi=300)