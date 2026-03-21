import pandas as pd
import matplotlib.pyplot as plt
from pulp import *

xls = pd.ExcelFile('Prices.xlsx')
elec_df = pd.read_excel(xls, 'PRICE_ELECTRIC')
gas_df  = pd.read_excel(xls, 'PRICE_GAS')
co2_df  = pd.read_excel(xls, 'PRICE_CO2')

co2_price = co2_df.iloc[0, 0] if not co2_df.empty else 28.3
ef_co2 = 0.05307
start_date, end_date = '2022-03-21', '2022-03-27'

elec_df['OPERATING_DATE'] = pd.to_datetime(elec_df['OPERATING_DATE'])
gas_df['OPERATING_DATE']  = pd.to_datetime(gas_df['OPERATING_DATE'])
mask = (elec_df['OPERATING_DATE'] >= start_date) & (elec_df['OPERATING_DATE'] <= end_date)
df = elec_df.loc[mask].sort_values(['OPERATING_DATE', 'HOUR_ENDING']).reset_index(drop=True)

#Parameters (1x1 Pseudo-Unit Specs)
pmin, pmax = 150, 305
hr_min, hr_inc = 7.695, 6.528
su_cost, su_fuel = 16500, 850
vom = 2.50

prob = LpProblem("Pseudo_Unit_Optimization", LpMaximize)
hours = range(len(df))
units = [1, 2]

#Variables
u = LpVariable.dicts("u", (units, hours), cat=LpBinary)
g_inc = LpVariable.dicts("g_inc", (units, hours), lowBound=0)
s = LpVariable.dicts("s", (units, hours), cat=LpBinary)

#Objective Function
obj_list = []
for t in hours:
    curr_date = df.loc[t, 'OPERATING_DATE']
    gas_row = gas_df.loc[gas_df['OPERATING_DATE'] == curr_date]
    gas_p = gas_row.iloc[0, 1] if not gas_row.empty else 7.0

    mfc = gas_p + (ef_co2 * co2_price)
    elec_p = df.loc[t, 'NP15 ($/MWh)']

    for i in units:
        rev = elec_p * (u[i][t] * pmin + g_inc[i][t])
        cost = (s[i][t] * (su_cost + su_fuel * mfc) +
                u[i][t] * (pmin * (hr_min * mfc + vom)) +
                g_inc[i][t] * (hr_inc * mfc + vom))

        tie_breaker = - (i * 0.001) * u[i][t]
        obj_list.append(rev - cost + tie_breaker)

prob += lpSum(obj_list)

#Constraints
for i in units:
    for t in hours:
        prob += g_inc[i][t] <= u[i][t] * (pmax - pmin)
        if t == 0:
            prob += s[i][t] >= u[i][t]
        else:
            prob += s[i][t] >= u[i][t] - u[i][t-1]

#Solve
prob.solve(PULP_CBC_CMD(msg=0))

results = []
for t in hours:
    curr_date = df.loc[t, 'OPERATING_DATE']
    gas_p = gas_df.loc[gas_df['OPERATING_DATE'] == curr_date].iloc[0, 1]
    results.append({
        'OPERATING_DATE': curr_date.strftime('%Y-%m-%d'),
        'HOUR_ENDING': df.loc[t, 'HOUR_ENDING'],
        'PRICE_ELECTRIC': df.loc[t, 'NP15 ($/MWh)'],
        'PRICE_GAS': gas_p,
        'MW_GENERATION_Unit1': value(u[1][t]*pmin + g_inc[1][t]),
        'MW_GENERATION_Unit2': value(u[2][t]*pmin + g_inc[2][t])
    })

csv_df = pd.DataFrame(results)
csv_df.to_csv('CCGT_PSEUDO.csv', index=False)

plt.figure(figsize=(14, 6))
plt.plot(csv_df.index, csv_df['MW_GENERATION_Unit1'], label='Pseudo-Unit 1', color='blue', alpha=0.7)
plt.plot(csv_df.index, csv_df['MW_GENERATION_Unit2'], label='Pseudo-Unit 2', color='orange', linestyle='--', alpha=0.7)
plt.title('Maximized Dispatch Plan: Independent Pseudo-Units (March 21-27, 2022)')
plt.xlabel('Hour of Operating Period')
plt.ylabel('MW Generation')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

for i in units:
    total_mwh = sum(value(u[i][t]*pmin + g_inc[i][t]) for t in hours)
    total_rev = sum(df.loc[t, 'NP15 ($/MWh)'] * value(u[i][t]*pmin + g_inc[i][t]) for t in hours)

    total_fuel_cost = 0
    total_vom_cost = 0
    total_su_cost = 0

    for t in hours:
        curr_date = df.loc[t, 'OPERATING_DATE']
        gas_p = gas_df.loc[gas_df['OPERATING_DATE'] == curr_date].iloc[0, 1]
        mfc = gas_p + (ef_co2 * co2_price)

        total_fuel_cost += value(s[i][t]*su_fuel*mfc + u[i][t]*pmin*hr_min*mfc + g_inc[i][t]*hr_inc*mfc)
        total_vom_cost += value(u[i][t]*pmin*vom + g_inc[i][t]*vom)
        total_su_cost += value(s[i][t]*su_cost)

    total_costs = total_fuel_cost + total_vom_cost + total_su_cost
    gross_margin = (total_rev - total_costs) / (305 * 1000)
    cap_factor = (total_mwh / (305 * 168)) * 100
    num_starts = sum(value(s[i][t]) for t in hours)

    print(f"\nMetrics for Pseudo-Unit {i}")
    print(f"Gross Margin ($/kW): {gross_margin:.4f}")
    print(f"Capacity Factor (%): {cap_factor:.2f}")
    print(f"Total Revenue ($): {total_rev:,.2f}")
    print(f"Total Costs ($): {total_costs:,.2f}")
    print(f"Fuel Costs ($): {total_fuel_cost:,.2f}")
    print(f"Number of Starts: {int(num_starts)}")
