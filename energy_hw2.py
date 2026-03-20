import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt

xls = pd.ExcelFile('Prices.xlsx')

elec_df = pd.read_excel(xls, 'PRICE_ELECTRIC')
gas_df  = pd.read_excel(xls, 'PRICE_GAS')
co2_df  = pd.read_excel(xls, 'PRICE_CO2')

elec_df['OPERATING_DATE'] = pd.to_datetime(elec_df['OPERATING_DATE'])
gas_df['OPERATING_DATE'] = pd.to_datetime(gas_df['OPERATING_DATE'])

# Configuration Constants
CO2_PRICE = 28.30  # $/tonne
EF_CO2 = 0.05307   # Assumption: Metric tonnes CO2 per MMBtu of Natural Gas

start_date, end_date = '2022-03-21', '2022-03-27'

df = elec_df[(elec_df['OPERATING_DATE'] >= start_date) &
             (elec_df['OPERATING_DATE'] <= end_date)].merge(gas_df, on='OPERATING_DATE')
df = df.sort_values(['OPERATING_DATE', 'HOUR_ENDING']).reset_index(drop=True)

# Technical Parameters
# Configs: 1:Off, 2:1CT, 3:2CT, 4:1x1, 5:2x1
configs = [1, 2, 3, 4, 5]
hours = range(len(df))

pmin = {1:0, 2:57, 3:114, 4:150, 5:312}
pmax = {1:0, 2:190, 3:380, 4:340, 5:610}
vom =  {1:0, 2:5.0, 3:5.0, 4:2.5, 5:2.0}
hr_min = {1:0, 2:12.591, 3:12.591, 4:7.695, 5:7.121}

# Piecewise Incremental Heat Rates (MMBtu/MWh)
# Segments: [Min-60%, 60-80%, 80-100%]
inc_hr = {
    2: [8.692, 9.177, 9.564],
    3: [8.692, 9.177, 9.564],
    4: [6.421, 6.525, 6.640],
    5: [6.292, 6.361, 6.452]
}

# Startup Parameters
su_fuel = {1:0, 2:220, 3:440, 4:850, 5:1070}
su_cost = {1:0, 2:7250, 3:14500, 4:16500, 5:23750}
min_up = {1:0, 2:1, 3:1, 4:2, 5:3}
min_down = {1:0, 2:1, 3:1, 4:2, 5:3}

# MILP
prob = pulp.LpProblem("CCGT_Optimization", pulp.LpMaximize)

# Decision Variables
x = pulp.LpVariable.dicts("status", (configs, hours), cat=pulp.LpBinary)
s = pulp.LpVariable.dicts("start", (configs, hours), cat=pulp.LpBinary)
# Generation variables for the 3 piecewise segments above Min Load
g_seg = pulp.LpVariable.dicts("g_seg", (configs, range(3), hours), lowBound=0)

# Objective Function: Total Revenue - (Fuel + VOM + Startup)
obj_terms = []
for t in hours:
    mfc = df.loc[t, 'PG&E Citygate ($/MMBtu)'] + (EF_CO2 * CO2_PRICE)
    p_elec = df.loc[t, 'NP15 ($/MWh)']

    for c in configs:
        if c == 1: continue
        # Revenue and Variable Cost at Min Load
        obj_terms.append(x[c][t] * pmin[c] * (p_elec - (hr_min[c] * mfc + vom[c])))
        # Revenue and Variable Cost for incremental segments
        for seg in range(3):
            obj_terms.append(g_seg[c][seg][t] * (p_elec - (inc_hr[c][seg] * mfc + vom[c])))
        # Startup Costs (Fuel + Non-fuel)
        obj_terms.append(-s[c][t] * (su_cost[c] + su_fuel[c] * mfc))

prob += pulp.lpSum(obj_terms)

# Constraints
for t in hours:
    # 1. Mutual Exclusivity: Only one configuration active
    prob += pulp.lpSum([x[c][t] for c in configs]) == 1

    # 2. Startup Logic
    for c in configs:
        if t == 0:
            prob += s[c][t] >= x[c][t] # Assume initially off
        else:
            prob += s[c][t] >= x[c][t] - x[c][t-1]

    # 3. Piecewise Segment Bounds
    for c in configs:
        if c == 1: continue
        p_max = pmax[c]
        p_min = pmin[c]
        # Segments: Min-60%, 60-80%, 80-100%
        # Widths: (0.6*Max - Min), (0.2*Max), (0.2*Max)
        seg_widths = [max(0, 0.6*p_max - p_min), 0.2*p_max, 0.2*p_max]
        for seg in range(3):
            prob += g_seg[c][seg][t] <= x[c][t] * seg_widths[seg]

# Solve
prob.solve(pulp.PULP_CBC_CMD(msg=0))

results = []
for t in hours:
    active_c = [c for c in configs if pulp.value(x[c][t]) > 0.5][0]
    gen = pmin[active_c] + sum(pulp.value(g_seg[active_c][seg][t]) for seg in range(3)) if active_c > 1 else 0
    results.append({
        'OPERATING_DATE': df.loc[t, 'OPERATING_DATE'].strftime('%Y-%m-%d'),
        'HOUR_ENDING': df.loc[t, 'HOUR_ENDING'],
        'PRICE_ELECTRIC': df.loc[t, 'NP15 ($/MWh)'],
        'PRICE_GAS': df.loc[t, 'PG&E Citygate ($/MMBtu)'],
        'CONFIGURATION_ACTIVE': active_c,
        'MW_GENERATION': gen
    })

res_df = pd.DataFrame(results)
res_df.to_csv('CCGT_CAISO.csv', index=False)

# Plots and Stats
# Plot MW vs Hour
plt.figure(figsize=(12, 5))
plt.plot(res_df.index, res_df['MW_GENERATION'], color='blue', label='Generation (MW)')
plt.title('2.c.iv: CCGT Dispatch (MW) vs. Hour (March 21-27, 2022)')
plt.xlabel('Hour of Week')
plt.ylabel('MW')
plt.grid(True)
plt.savefig('dispatch_plot.png')

# Plot Config vs Time
plt.figure(figsize=(12, 3))
plt.step(res_df.index, res_df['CONFIGURATION_ACTIVE'], where='post', color='red')
plt.yticks(configs, ['Off', '1 CT', '2 CT', '1x1', '2x1'])
plt.title('2.c.v: Configuration Mode vs. Time')
plt.xlabel('Hour of Week')
plt.grid(True)
plt.savefig('config_plot.png')

# Summary Stats
total_rev = sum(res_df['MW_GENERATION'] * res_df['PRICE_ELECTRIC'])
total_starts = sum(pulp.value(s[c][t]) for c in configs for t in hours if c > 1)
# Calculate Total Costs from Objective (Negative of the cost portion)
total_costs = total_rev - pulp.value(prob.objective)
fuel_costs = sum((pmin[active_c] * hr_min[active_c] + sum(pulp.value(g_seg[active_c][seg][t]) * inc_hr[active_c][seg] for seg in range(3))) * (df.loc[t, 'PG&E Citygate ($/MMBtu)'] + EF_CO2 * CO2_PRICE)
                 for t, active_c in enumerate(res_df['CONFIGURATION_ACTIVE']) if active_c > 1)

print(f"Gross Margin ($/kW): {pulp.value(prob.objective) / (610 * 1000):.4f}")
print(f"Capacity Factor (%): {(res_df['MW_GENERATION'].mean() / 610) * 100:.2f}%")
print(f"Total Revenue ($): ${total_rev:,.2f}")
print(f"Total Costs ($): ${total_costs:,.2f}")
print(f"Fuel Costs ($): ${fuel_costs:,.2f}")
print(f"Number of Starts: {int(total_starts)}")

