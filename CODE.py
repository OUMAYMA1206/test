import pandas as pd
import numpy as np
from openpyxl import load_workbook
import json
import sys
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# COMPONENT CLASSES
# ============================================================
class PVPlant:
    """Represents a Photovoltaic (PV) plant."""
    def __init__(self, pv_profile: pd.Series):
        self.profile = pv_profile

class Battery:
    """
    Represents a battery energy storage system.
    Handles charging and discharging based on capacity, efficiency, power limits,
    and defined State of Charge (SOC) limits.
    """
    def __init__(self, capacity_kwh, soc_initial, efficiency_ch, efficiency_dis,
                 max_charge_kw, max_discharge_kw, depth_of_discharge,
                 max_charge_soc_limit_percent=0.80, min_discharge_soc_limit_percent=0.20):
        self.capacity = capacity_kwh
        self.eta_ch = efficiency_ch
        self.eta_dis = efficiency_dis
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.min_soc = self.capacity * (1 - depth_of_discharge)
        self.max_soc = self.capacity * max_charge_soc_limit_percent
        self.soc = min(max(soc_initial * capacity_kwh, self.min_soc), self.max_soc)
        self.history = []

    def charge(self, energy_kwh):
        potential_charge_from_source = min(energy_kwh, self.max_charge_kw)
        space_to_max_soc_kwh_supplied = (self.max_soc - self.soc) / self.eta_ch if self.eta_ch > 0 else float('inf')
        actual_charge_kwh_supplied = min(potential_charge_from_source, space_to_max_soc_kwh_supplied)
        self.soc += actual_charge_kwh_supplied * self.eta_ch
        return energy_kwh - actual_charge_kwh_supplied

    def discharge(self, demand_kwh):
        potential_discharge_to_demand = min(demand_kwh, self.max_discharge_kw)
        available_energy_from_soc_delivered = (self.soc - self.min_soc) * self.eta_dis if self.eta_dis > 0 else 0.0
        actual_discharge_kwh_delivered = min(potential_discharge_to_demand, available_energy_from_soc_delivered)
        self.soc -= actual_discharge_kwh_delivered / self.eta_dis if self.eta_dis > 0 else 0.0
        return demand_kwh - actual_discharge_kwh_delivered

    def record(self):
        self.history.append(self.soc)

class Electrolyzer:
    """Represents an electrolyzer for hydrogen production."""
    def __init__(self, specific_consumption_kwh_per_kg, nominal_capacity_kw):
        self.specific_consumption = specific_consumption_kwh_per_kg
        self.nominal_capacity = nominal_capacity_kw

    def produce(self, available_energy_kwh):
        energy_consumed = min(available_energy_kwh, self.nominal_capacity)
        mass_h2 = energy_consumed / self.specific_consumption if self.specific_consumption > 0 else 0.0
        return mass_h2, energy_consumed

class H2Storage:
    """Represents a hydrogen storage tank."""
    def __init__(self, capacity_kg, initial_kg, min_level_kg, max_outflow_rate):
        self.capacity = capacity_kg
        self.min_level = min_level_kg
        self.level = max(initial_kg, min_level_kg)
        self.max_outflow_rate = max_outflow_rate
        self.history = []

    def record(self):
        self.history.append(self.level)

class Grid:
    """Represents the electrical grid, providing power and associated CO2 emissions."""
    def __init__(self, emission_factor_kgCO2_per_kwh):
        self.emission_factor = emission_factor_kgCO2_per_kwh

    def draw(self, demand_kwh):
        return demand_kwh * self.emission_factor

# =======================================
# SIMULATION FUNCTION (LOGIC CORRECTED)
# =======================================
def simulate(pv, battery, electrolyzer, grid, co2_threshold_tCO2_per_tH2,
             storage, comp_rate, truck_mass, compressor_power_kw):
    """Runs the hourly simulation of the hydrogen system with corrected logic."""
    records = []
    grid_block_events = []
    monthly_co2 = 0.0
    monthly_h2 = 0.0
    current_month = None
    co2_threshold_kg = co2_threshold_tCO2_per_tH2 * 1000
    truck_idx = 1
    current_truck_fill = 0.0
    max_h2_per_h = electrolyzer.nominal_capacity / electrolyzer.specific_consumption if electrolyzer.specific_consumption > 0 else 0.0
    base_min_op = 0.25 * electrolyzer.nominal_capacity

    for ts, pv_gen in pv.profile.items():
        if ts.month != current_month:
            current_month = ts.month
            monthly_co2 = 0.0
            monthly_h2 = 0.0

        weekday, hour = ts.weekday(), ts.hour
        in_truck_window = (weekday != 6) and (6 <= hour < 18)
        is_production_hours = (6 <= hour < 18)

        # H2 Production Logic
        target_h2 = min(comp_rate, max_h2_per_h) if is_production_hours else 0.0
        h2_pv = h2_batt = h2_grid = 0.0
        draw_elty_from_grid = 0.0
        co2_elty = 0.0

        if target_h2 > 0:
            required_energy_kwh = target_h2 * electrolyzer.specific_consumption
            
            # 1. Use PV for electrolyzer
            pv_for_el = min(pv_gen, required_energy_kwh)
            produced_h2_pv, used_pv = electrolyzer.produce(pv_for_el)
            h2_pv = produced_h2_pv
            required_energy_kwh -= used_pv
            
            # Excess PV to battery
            excess_pv = pv_gen - used_pv
            if excess_pv > 0:
                _ = battery.charge(excess_pv)

            # 2. Use Battery for electrolyzer
            if required_energy_kwh > 0:
                rem_after_batt = battery.discharge(required_energy_kwh)
                used_batt = required_energy_kwh - rem_after_batt
                h2_batt = used_batt / electrolyzer.specific_consumption if electrolyzer.specific_consumption > 0 else 0.0
                required_energy_kwh = rem_after_batt
            
            # 3. Use Grid for electrolyzer
            if required_energy_kwh > 0:
                projected_h2 = monthly_h2 + (h2_pv + h2_batt)
                future_ratio = (monthly_co2 / 1000) / max((projected_h2 / 1000), 1e-6)
                if future_ratio <= co2_threshold_tCO2_per_tH2:
                    pot_draw = min(required_energy_kwh, electrolyzer.nominal_capacity)
                    if pot_draw >= base_min_op:
                        produced_h2_grid, used_kwh = electrolyzer.produce(pot_draw)
                        h2_grid = produced_h2_grid
                        draw_elty_from_grid = used_kwh
                        co2_elty = grid.draw(draw_elty_from_grid)
        
        if pv_gen > 0 and not 'used_pv' in locals(): # handles case where target_h2 is 0
             _ = battery.charge(pv_gen)

        H = h2_pv + h2_batt + h2_grid

        # ### CORRECTED H2 DISTRIBUTION LOGIC ###
        release_from_storage_for_truck = 0.0
        comp_to_storage = 0.0
        comp_to_truck_from_prod = 0.0
        total_h2_to_truck = 0.0
        truck_inflow_limit_per_h = storage.max_outflow_rate

        if in_truck_window and current_truck_fill < truck_mass:
            truck_space_needed = truck_mass - current_truck_fill
            max_fill_this_hour = min(truck_space_needed, truck_inflow_limit_per_h)
            
            # Step 1: Use current production (H) to fill the truck.
            comp_to_truck_from_prod = min(H, max_fill_this_hour)
            
            # Step 2: If more H2 is needed, take from storage.
            amount_still_needed = max_fill_this_hour - comp_to_truck_from_prod
            if amount_still_needed > 0:
                available_in_storage = max(0, storage.level - storage.min_level)
                release_from_storage_for_truck = min(amount_still_needed, available_in_storage, storage.max_outflow_rate)
            
            total_h2_to_truck = comp_to_truck_from_prod + release_from_storage_for_truck
            current_truck_fill += total_h2_to_truck
            storage.level -= release_from_storage_for_truck

            # Step 3: Put any remaining production into storage.
            prod_remaining = H - comp_to_truck_from_prod
            storage_space = max(0, storage.capacity - storage.level)
            comp_to_storage = min(prod_remaining, storage_space)
            storage.level += comp_to_storage
        else:
            # No truck filling, all production goes to storage.
            storage_space = max(0, storage.capacity - storage.level)
            comp_to_storage = min(H, storage_space)
            storage.level += comp_to_storage
            total_h2_to_truck = 0.0

        logged_truck_idx = truck_idx
        logged_truck_fill = current_truck_fill
        if current_truck_fill >= truck_mass and truck_mass > 0:
            truck_idx += 1
            current_truck_fill = 0.0

        # Compressor energy
        total_compressed_mass = comp_to_truck_from_prod + comp_to_storage
        total_compr_energy_kwh = (total_compressed_mass / comp_rate) * compressor_power_kw if comp_rate > 0 else 0.0
        rem_compr_demand_after_batt = battery.discharge(total_compr_energy_kwh)
        draw_compr_from_grid = rem_compr_demand_after_batt
        co2_compr = grid.draw(draw_compr_from_grid)
        
        # Final Tracking
        total_draw_kwh = draw_elty_from_grid + draw_compr_from_grid
        total_co2_hour = co2_elty + co2_compr
        monthly_h2 += H
        monthly_co2 += total_co2_hour
        
        battery.record()
        storage.record()

        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'pv_gen_kw': pv_gen,
            'batt_soc_kwh': battery.soc,
            'h2_prod_kg': H,
            'h2_from_pv': h2_pv,
            'h2_from_battery': h2_batt,
            'h2_from_grid': h2_grid,
            'grid_draw_kwh': total_draw_kwh,
            'co2_emitted_kg': total_co2_hour,
            'storage_kg': storage.level,
            'release_from_storage_kg': release_from_storage_for_truck,
            'comp_to_truck_kg': total_h2_to_truck,
            'comp_to_storage_kg': comp_to_storage,
            'truck_fill_kg': logged_truck_fill,
            'electrolyzer_kwh': H * electrolyzer.specific_consumption,
            'compressor_kwh': total_compr_energy_kwh,
            'truck_index': logged_truck_idx,
        })
        # Reset used_pv for next iteration
        if 'used_pv' in locals():
            del used_pv

    results_df = pd.DataFrame(records)
    grid_log_df = pd.DataFrame(grid_block_events, columns=['timestamp', 'reason', 'co2_ratio'])
    return results_df, grid_log_df

# =================
# MAIN SCRIPT
# =================
if __name__ == '__main__':
    try:
        # Load and prepare PV data
        raw = pd.read_excel('pvsyst_output.xlsx', header=None).dropna(how='all')
        data = raw.iloc[3:]
        if data.shape[1] == 1:
            split_data = data[0].astype(str).str.split(',', expand=True)
            split_data.columns = ['DateTime', 'E_Avail_kW']
        else:
            split_data = data.iloc[:, :2]
            split_data.columns = ['DateTime', 'E_Avail_kW']
        
        split_data['DateTime'] = pd.to_datetime(split_data['DateTime'], errors='coerce', dayfirst=True)
        split_data['E_Avail_kW'] = pd.to_numeric(split_data['E_Avail_kW'], errors='coerce')
        split_data = split_data.dropna()
        if split_data.empty:
            raise ValueError("No valid data found in pvsyst_output.xlsx.")

        pv_df = split_data.set_index('DateTime')
        all_hours = pd.date_range(start=pv_df.index.min().normalize(), end=pv_df.index.max(), freq='h')
        pv_df = pv_df.reindex(all_hours, fill_value=0)
        pv = PVPlant(pv_df['E_Avail_kW'])

        # Define simulation parameters directly
        p = {
            'battery_capacity_kwh': 7300,
            'battery_soc_initial': 0.5,
            'battery_efficiency_ch': 0.92,
            'battery_efficiency_dis': 0.92,
            'battery_max_charge_kw': 1700,
            'battery_max_discharge_kw': 1700,
            'battery_depth_of_discharge': 0.8,
            'electrolyzer_specific_consumption_kwh_per_kg': 55.5,
            'electrolyzer_nominal_capacity_kw': 1159.4,
            'grid_emission_factor_kgCO2_per_kwh': 0.23,
            'co2_threshold_tCO2_per_tH2': 3,
            'h2_storage_capacity_kg': 400,
            'h2_storage_initial_kg': 400,
            'compressor_rate_kg_per_h': 21,
            'truck_fill_mass_kg': 440,
            'h2_storage_min_level_kg': 152,
            'h2_storage_max_outflow_rate_kg_per_h': 72, # <--- HADA HOWA L-BIDIL
            'compressor_power_kw': 52,
            'battery_max_charge_soc_limit_percent': 0.80,
            'battery_min_discharge_soc_limit_percent': 0.20
        }

        # Initialize components
        battery = Battery(
            capacity_kwh=float(p['battery_capacity_kwh']),
            soc_initial=float(p['battery_soc_initial']),
            efficiency_ch=float(p['battery_efficiency_ch']),
            efficiency_dis=float(p['battery_efficiency_dis']),
            max_charge_kw=float(p['battery_max_charge_kw']),
            max_discharge_kw=float(p['battery_max_discharge_kw']),
            depth_of_discharge=float(p['battery_depth_of_discharge']),
            max_charge_soc_limit_percent=float(p['battery_max_charge_soc_limit_percent']),
            min_discharge_soc_limit_percent=float(p['battery_min_discharge_soc_limit_percent'])
        )
        electrolyzer = Electrolyzer(
            specific_consumption_kwh_per_kg=float(p['electrolyzer_specific_consumption_kwh_per_kg']),
            nominal_capacity_kw=float(p['electrolyzer_nominal_capacity_kw'])
        )
        grid = Grid(float(p['grid_emission_factor_kgCO2_per_kwh']))
        storage = H2Storage(
            capacity_kg=float(p['h2_storage_capacity_kg']),
            initial_kg=float(p['h2_storage_initial_kg']),
            min_level_kg=float(p['h2_storage_min_level_kg']),
            max_outflow_rate=float(p['h2_storage_max_outflow_rate_kg_per_h'])
        )

        # Run simulation
        df_results, grid_log_df = simulate(
            pv, battery, electrolyzer, grid,
            float(p['co2_threshold_tCO2_per_tH2']),
            storage,
            float(p['compressor_rate_kg_per_h']),
            float(p['truck_fill_mass_kg']),
            float(p['compressor_power_kw'])
        )

        # --- Monthly and Yearly Summary ---
        df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
        df_results_indexed = df_results.set_index('timestamp')
        
        monthly = df_results_indexed.resample('ME').sum()
        monthly['ratio_tco2_tH2'] = (monthly['co2_emitted_kg'] / 1000) / (monthly['h2_prod_kg'] / 1000)
        monthly['ratio_tco2_tH2'] = monthly['ratio_tco2_tH2'].replace([np.inf, -np.inf], np.nan).fillna(0)
        print("\n=== Monthly Summary ===")
        print(monthly[['h2_prod_kg', 'grid_draw_kwh', 'co2_emitted_kg', 'ratio_tco2_tH2']])

        yearly = df_results_indexed.resample('YE').sum()
        yearly['ratio_tco2_tH2'] = (yearly['co2_emitted_kg'] / 1000) / (yearly['h2_prod_kg'] / 1000)
        yearly['ratio_tco2_tH2'] = yearly['ratio_tco2_tH2'].replace([np.inf, -np.inf], np.nan).fillna(0)
        print("\n=== Yearly Summary ===")
        print(yearly[['h2_prod_kg', 'grid_draw_kwh', 'co2_emitted_kg', 'ratio_tco2_tH2']])

        # --- Export All Results to a Single Excel File ---
        output_excel_path = 'final_simulation_output.xlsx'
        df_out_params = pd.DataFrame(p.items(), columns=['Parameter', 'Value'])
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            df_out_params.to_excel(writer, sheet_name='Dati input', index=False)
            df_results.to_excel(writer, sheet_name='Hourly_Data', index=False)
            monthly.to_excel(writer, sheet_name='Monthly_Summary')
            yearly.to_excel(writer, sheet_name='Yearly_Summary')
            grid_log_df.to_excel(writer, sheet_name='Grid_Block_Log', index=False)
        print(f"\nAll simulation results have been exported to: {output_excel_path}")

        # Plots
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 8))
        df_results_indexed[['storage_kg', 'truck_fill_kg']].plot(ax=plt.gca(), secondary_y=['truck_fill_kg'])
        plt.title('H2 Storage vs. Truck Fill Level (First 100 Hours)')
        plt.xlim(df_results_indexed.index[0], df_results_indexed.index[100])
        plt.savefig('storage_vs_truck.png')
        plt.show()

    except FileNotFoundError as e:
        print(f"\nX ERROR: File not found. Make sure '{e.filename}' is in the same folder.", file=sys.stderr)
    except Exception as e:
        print(f"\nX ERROR: An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)