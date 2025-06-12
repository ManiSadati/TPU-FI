import os, shutil
import pandas as pd

prefix_path = "./results"
models = ['vit8', 'segmentation128', 'segmentation256']
# ['layer', 'name', 'type', 'total runs', 'errors', 'sdc_count', 'sdc_rate', 'd(out_c)', 'layer area', 'num_ops']
static_columns = ['layer', 'name', 'type', 'd(out_c)', 'layer area', 'num_ops']
dynamic_columns = ['total runs', 'errors', 'sdc_count']

def merge_files():

    for model in models:
        dfs = []
        for i in range(32):
            file_path = os.path.join(prefix_path, f"{model}/file{i}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dfs.append(df)
            else:
                print(f"file {file_path} is missing ...")
                exit()
        print(dfs[0].shape)
        print(dfs[0].columns)
        for i in range(1, len(dfs)):
            if(dfs[i].shape != dfs[0].shape):
                print(f"file {i} has different shape: {dfs[i].shape} != {dfs[0].shape}")
                exit()
            for row in range(dfs[0].shape[0]):
                for col in static_columns:
                    if dfs[0][col][row] != dfs[i][col][row]:
                        print(f"file {i} has different value at row {row}, column {col}: {dfs[0][col][row]} != {dfs[i][col][row]}")
                        exit()
                for col in dynamic_columns:
                    dfs[0].loc[row, col] += dfs[i].loc[row, col]
        
        dfs[0].drop(columns=['sdc_rate'], inplace=True)

        dfs[0].rename(columns={'layer area': 'tmp',}, inplace=True)
        dfs[0].rename(columns={'num_ops': 'layer area',}, inplace=True)
        dfs[0].rename(columns={'d(out_c)': 'num_ops',}, inplace=True)
        dfs[0].rename(columns={'tmp': 'd(out_c)',}, inplace=True)

        new_file_path = os.path.join(prefix_path, f"Merged_{model}.csv")
        dfs[0].to_csv(new_file_path, index=False)
        print(f"merged file saved to {new_file_path}")

def add_fit_columns():
    fault_types_fit_rates = {
        'single': 13.41935484,
        'small-box': 3.634408602,
        'medium-box': 8.946236559
    }
    
    for model in models:
        file_path = os.path.join(prefix_path, f"Merged_{model}.csv")
        df = pd.read_csv(file_path)
        df['sdc_rate'] = df['errors'] / df['total runs']
        df['critical_sdc_rate'] = df['sdc_count'] / df['total runs']
        df['portion_of_tpu'] = df['num_ops'] * 100 / 1258291200
        df['fault_fit_rate'] = df['type'].map(fault_types_fit_rates)
        df['layer_vs_fault_fit_rate'] = df['portion_of_tpu'] * df['fault_fit_rate']
        df['fit_times_avf'] = df['errors'] * df['layer_vs_fault_fit_rate'] / df['total runs']
        df['fit_times_avf_critical'] = df['sdc_count'] * df['layer_vs_fault_fit_rate'] / df['total runs']
        file_path = os.path.join(prefix_path, f"Full_{model}.csv")
        df.to_csv(file_path, index=False)
        print(f"full file saved to {file_path}")
    
    return df

def get_fit_sums():
    for model in models:
        file_path = os.path.join(prefix_path, f"Full_{model}.csv")
        df = pd.read_csv(file_path)

        fit_sum_per_type = df.groupby('type')[['fit_times_avf', 'fit_times_avf_critical']].sum()
        fit_sum_per_type['avg_sdc_rate'] = df.groupby('type')['sdc_rate'].mean()
        fit_sum_per_type['avg_critical_sdc_rate'] = df.groupby('type')['critical_sdc_rate'].mean()
        fit_sum_per_type = fit_sum_per_type.reset_index()
        fit_sum_per_type.to_csv(os.path.join(prefix_path, f"fit_sum_per_type_{model}.csv"), index=False)
        print(fit_sum_per_type)

        fit_sum_per_name = df.groupby('name')[['fit_times_avf', 'fit_times_avf_critical']].sum()
        fit_sum_per_name['avg_sdc_rate'] = df.groupby('name')['sdc_rate'].mean()
        fit_sum_per_name['avg_critical_sdc_rate'] = df.groupby('name')['critical_sdc_rate'].mean()
        fit_sum_per_name = fit_sum_per_name.reset_index()
        fit_sum_per_name.to_csv(os.path.join(prefix_path, f"fit_sum_per_name_{model}.csv"), index=False)
        print(fit_sum_per_name)
        



if __name__ == "__main__":
    # merge_files()
    add_fit_columns()
    get_fit_sums()
    