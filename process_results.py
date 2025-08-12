import os, shutil
import pandas as pd

prefix_path = "./results"
static_columns = ['layer', 'name', 'type', 'd(out_c)', 'layer area', 'num_ops']
dynamic_columns = ['total runs', 'errors', 'sdc_count']

def merge_files():

    models = [
        "aug9-vit8",
        "aug9-vit16",
        "aug9-deeplab1",
        "aug9-deeplab2",
        "aug9-unet1",
        "aug9-unet2"
    ]
    for model in models:
        dfs = []
        for i in range(32):
            file_path = os.path.join(prefix_path, f"{model}/FI-vit-results_{i}.csv")
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


        new_file_path = os.path.join(prefix_path, f"Merged_{model}.csv")
        dfs[0].to_csv(new_file_path, index=False)
        print(f"merged file saved to {new_file_path}")

def add_fit_columns():
    fault_types_fit_rates = {
        'single': 13.41935484,
        'small-box': 3.634408602,
        'medium-box': 8.946236559,
        'cpu': 0.0
    }
    models = [
        "aug9-vit8",
        "aug9-vit16",
        "aug9-deeplab1",
        "aug9-deeplab2",
        "aug9-unet1",
        "aug9-unet2"
    ]

    for model in models:
        file_path = os.path.join(prefix_path, f"Merged_{model}.csv")
        df = pd.read_csv(file_path)
        df['sdc_rate'] = df['errors'] / df['total runs']
        df['num_ops_limited'] = df['num_ops'].clip(upper=256*256)
        df['critical_sdc_rate'] = df['sdc_count'] / df['total runs']
        df['portion_of_tpu'] = df['num_ops_limited'] * 100 / 1258291200
        df['fault_type_fit_rate'] = df['type'].map(fault_types_fit_rates)
        df['layer_vs_fault_fit_rate'] = df['portion_of_tpu'] * df['fault_type_fit_rate']
        df['fit_times_avf'] = df['errors'] * df['layer_vs_fault_fit_rate'] / df['total runs']
        df['fit_times_avf_critical'] = df['sdc_count'] * df['layer_vs_fault_fit_rate'] / df['total runs']
        file_path = os.path.join(prefix_path, f"Full_{model}.csv")
        df.to_csv(file_path, index=False)
        print(f"full file saved to {file_path}")
    
    return df

import os
import pandas as pd

def get_fit_sums():
    weights = {
        'single': 0.48,
        'small-box': 0.15,
        'medium-box': 0.37,
        'cpu': 0.0
    }

    models = [
        "aug9-vit8",
        "aug9-vit16",
        "aug9-deeplab1",
        "aug9-deeplab2",
        "aug9-unet1",
        "aug9-unet2"
    ]

    for model in models:
        print(f"\n\nProcessing model: {model}")
        file_path = os.path.join(prefix_path, f"Full_{model}.csv")
        if not os.path.exists(file_path):
            print(f"  File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # ---- Basic checks ----
        needed = ['layer','name','type','sdc_rate','critical_sdc_rate','fit_times_avf','fit_times_avf_critical']
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"  Missing required columns: {missing}")
            continue

        df['layer'] = pd.to_numeric(df['layer'], errors='coerce')
        num_layers_global = df['layer'].nunique()
        layers_per_type = df.groupby('name')['layer'].nunique()

        # ---- BY LAYER TYPE ----
        df['w'] = df['type'].map(weights).fillna(0.0)
        df['w_sdc'] = df['sdc_rate'] * df['w']
        df['w_sdc_crit'] = df['critical_sdc_rate'] * df['w']

        per_layer_weighted = (
            df.groupby(['name','layer'], as_index=False)
              .agg(w_sdc_sum=('w_sdc','sum'),
                   w_sum=('w','sum'),
                   w_sdc_crit_sum=('w_sdc_crit','sum'))
        )

        per_layer_weighted['weighted_sdc_rate'] = per_layer_weighted.apply(
            lambda r: (r['w_sdc_sum'] / r['w_sum']) if r['w_sum'] > 0 else 0.0, axis=1
        )
        per_layer_weighted['weighted_critical_sdc_rate'] = per_layer_weighted.apply(
            lambda r: (r['w_sdc_crit_sum'] / r['w_sum']) if r['w_sum'] > 0 else 0.0, axis=1
        )

        by_layer_type_sdc = (
            per_layer_weighted
            .groupby('name', as_index=False)
            .agg(weighted_sdc_rate=('weighted_sdc_rate','mean'),
                 weighted_critical_sdc_rate=('weighted_critical_sdc_rate','mean'))
        )

        fit_sums_by_type = (
            df.groupby('name', as_index=False)
              .agg(fit_times_avf_sum=('fit_times_avf','sum'),
                   fit_times_avf_critical_sum=('fit_times_avf_critical','sum'))
        )
        fit_sums_by_type['num_layers_type'] = fit_sums_by_type['name'].map(layers_per_type)
        fit_sums_by_type['fit_times_avf_per_layer'] = (
            fit_sums_by_type['fit_times_avf_sum'] / fit_sums_by_type['num_layers_type'].replace(0, pd.NA)
        )
        fit_sums_by_type['fit_times_avf_critical_per_layer'] = (
            fit_sums_by_type['fit_times_avf_critical_sum'] / fit_sums_by_type['num_layers_type'].replace(0, pd.NA)
        )

        by_layer_type = fit_sums_by_type[['name','num_layers_type','fit_times_avf_per_layer','fit_times_avf_critical_per_layer']].merge(
            by_layer_type_sdc, on='name', how='left'
        ).rename(columns={
            'name':'layer_type',
            'num_layers_type':'num_layers'
        })

        out_by_layer_type = os.path.join(prefix_path, f"{model}_by_layer_type.csv")
        by_layer_type.to_csv(out_by_layer_type, index=False)
        print(f"  Saved by-layer-type to {out_by_layer_type}")

        # ---- BY FAULT TYPE ----
        fit_by_fault = (
            df.groupby('type', as_index=False)
              .agg(fit_times_avf_sum=('fit_times_avf','sum'),
                   fit_times_avf_critical_sum=('fit_times_avf_critical','sum'))
        )
        fit_by_fault['fit_times_avf_per_layer'] = fit_by_fault['fit_times_avf_sum'] / (num_layers_global if num_layers_global else 1)
        fit_by_fault['fit_times_avf_critical_per_layer'] = fit_by_fault['fit_times_avf_critical_sum'] / (num_layers_global if num_layers_global else 1)

        sdc_layer_means = (
            df.groupby(['type','layer'], as_index=False)
              .agg(sdc_rate=('sdc_rate','mean'),
                   critical_sdc_rate=('critical_sdc_rate','mean'))
        )
        sdc_by_fault = (
            sdc_layer_means
            .groupby('type', as_index=False)
            .agg(sdc_rate_mean=('sdc_rate','mean'),
                 critical_sdc_rate_mean=('critical_sdc_rate','mean'))
        )

        by_fault_type = fit_by_fault.merge(sdc_by_fault, on='type', how='left').rename(columns={
            'type':'fault_type'
        })

        out_by_fault_type = os.path.join(prefix_path, f"{model}_by_fault_type.csv")
        by_fault_type.to_csv(out_by_fault_type, index=False)
        print(f"  Saved by-fault-type to {out_by_fault_type}")



if __name__ == "__main__":
    # merge_files()
    add_fit_columns()
    get_fit_sums()