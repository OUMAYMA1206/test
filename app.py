from flask import Flask, render_template
import pandas as pd
import json
import traceback
import numpy as np

app = Flask(__name__)

#--- Function to handle data types that JSON doesn't understand
def json_converter(obj):
    """Custom JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

@app.route('/')
def professional_dashboard():
    """Renders the professional dashboard by reading data from the simulation output Excel file."""
    try:
        excel_file_path = 'final_simulation_output.xlsx'
        
        # 1. Read input parameters from the 'Dati input' sheet
        df_params = pd.read_excel(excel_file_path, sheet_name='Dati input', index_col='Parameter')['Value']
        parameters_dict = df_params.to_dict()

        # 2. Read hourly results from the 'Hourly_Data' sheet
        df_hourly = pd.read_excel(excel_file_path, sheet_name='Hourly_Data', engine='openpyxl')
        df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp'])

        # 3. Read annual summary from the 'Yearly_Summary' sheet
        df_summary = pd.read_excel(excel_file_path, sheet_name='Yearly_Summary')
        annual_summary_dict = df_summary.to_dict('records')[0]
        
        # 4. Add total consumption to the annual summary from the hourly data
        annual_summary_dict['total_electrolyzer_kwh'] = df_hourly['electrolyzer_kwh'].sum()
        annual_summary_dict['total_compressor_kwh'] = df_hourly['compressor_kwh'].sum()

        # 5. Prepare data for the dashboard as JSON strings
        data_hourly_json = df_hourly.to_json(orient='records', date_format='iso')
        annual_summary_json = json.dumps(annual_summary_dict, default=json_converter)
        parameters_json = json.dumps(parameters_dict, default=json_converter)

        return render_template('dashboard.html',
                               data_hourly_data=data_hourly_json,
                               annual_summary_data=annual_summary_json,
                               parameters_data=parameters_json)

    except Exception as e:
        error_details = traceback.format_exc()
        return f"""<div style='font-family: monospace; padding: 2em; background-color: #111; color: #ff5555;'>
                     <h1 style='color: #ff8888;'>An Error Occurred in Python Backend</h1>
                     <pre style='background-color: #222; padding: 1em; border-radius: 5px; color: #eee; white-space: pre-wrap;'>{error_details}</pre>
                   </div>"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)