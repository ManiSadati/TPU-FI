import pandas as pd

# Load the CSV file
df = pd.read_csv('fault_injection_results.csv')

# Step 1: Define FIT rates per fault type
fault_types_fit_rates = {
    'single': 13.41935484,
    'small-box': 3.634408602,
    'medium-box': 8.946236559
}

# Step 2: Compute all required columns
df['portion_of_tpu'] = df['num_ops'] * 100 / 1258291200
df['fault_fit_rate'] = df['type'].map(fault_types_fit_rates)
df['layer_vs_fault_fit_rate'] = df['portion_of_tpu'] * df['fault_fit_rate']
df['fit_times_avf'] = df['error'] * df['layer_vs_fault_fit_rate'] / df['total runs']
df['fit_times_avf_critical'] = df['misclassification rate'] * df['layer_vs_fault_fit_rate']

# Step 3: Save the enriched data to a new file
df.to_csv('fault_injection_results_with_fit.csv', sep='\t', index=False)

# Step 4: Compute FIT sums per fault type
fit_per_type = df.groupby('type')[['fit_times_avf', 'fit_times_avf_critical']].sum()
fit_per_type.columns = ['fit_sum', 'critical_fit_sum']

# Step 5: Compute FIT sums per layer type
fit_per_layer = df.groupby('name')[['fit_times_avf', 'fit_times_avf_critical']].sum()
fit_per_layer.columns = ['fit_sum', 'critical_fit_sum']

# Step 6: Compute SDC rate (total SDC / total runs) per fault type
sdc_per_type = df.groupby('type').apply(lambda x: x['missclassification'].sum() / x['total runs'].sum()).reset_index()
sdc_per_type.columns = ['type', 'sdc_rate_total']

# Step 7: Compute SDC rate per layer type
sdc_per_layer = df.groupby('name').apply(lambda x: x['missclassification'].sum() / x['total runs'].sum()).reset_index()
sdc_per_layer.columns = ['layer_type', 'sdc_rate_total']

# Step 8: Print everything
print("\nFIT Rates (used for scaling):")
for k, v in fault_types_fit_rates.items():
    print(f"{k:12} : {v:.6f}")

print("\nFIT Sum and Critical FIT Sum per Fault Type:")
print(fit_per_type)

print("\nFIT Sum and Critical FIT Sum per Layer Type:")
print(fit_per_layer)

print("\nTotal SDC / Total Runs per Fault Type:")
print(sdc_per_type)

print("\nTotal SDC / Total Runs per Layer Type:")
print(sdc_per_layer)
