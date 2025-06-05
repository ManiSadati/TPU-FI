import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = './segmentation_fi_results.csv'
df = pd.read_csv(file_path, sep='\t')

# Compute sum(sdcs)/sum(total runs) for each type
summary = df.groupby('type').apply(lambda x: x['sdc_count'].sum() / x['total runs'].sum()).reset_index()
summary.columns = ['type', 'total_sdc_rate']

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Prepare and save plots for each type
types = df['type'].unique()
for t in types:
    plt.figure()
    subset = df[df['type'] == t]
    sns.barplot(x='layer', y='sdc_rate', data=subset)
    plt.title(f'SDC Rate per Layer for {t}')
    plt.ylabel('SDC Rate')
    plt.xlabel('Layer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/sdc_rate_{t}.png')  # Save figure
    plt.close()  # Close to avoid displaying in non-interactive environments

print(summary)
