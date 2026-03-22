import os
import pandas as pd


def add_fit_columns():
    fault_types_fit_rates = {
        'single': 13.41935484,
        'small-box': 3.634408602,
        'medium-box': 8.946236559,
        'cpu': 0.0
    }
    results_dir = "./results"

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # Iterate over all CSV files in ./results that have not been processed at all (start with FI)
    for filename in os.listdir(results_dir):
        if not filename.endswith(".csv"):
            continue
        
        if not filename.startswith("FI-"):
            continue

        file_path = os.path.join(results_dir, filename)
        model = os.path.splitext(filename)[0]

        print(f"\n\nProcessing model: {model}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"  Failed to read {file_path}: {e}")
            continue
        
        df['sdc_rate'] = df['errors'] / df['total runs']
        df['num_ops_limited'] = df['num_ops'].clip(upper=256*256)
        df['critical_sdc_rate'] = df['sdc_count'] / df['total runs']
        df['portion_of_tpu'] = df['num_ops_limited'] * 100 / 1258291200
        df['fault_type_fit_rate'] = df['type'].map(fault_types_fit_rates)
        df['layer_vs_fault_fit_rate'] = df['portion_of_tpu'] * df['fault_type_fit_rate']
        df['fit_times_avf'] = df['errors'] * df['layer_vs_fault_fit_rate'] / df['total runs']
        df['fit_times_avf_critical'] = df['sdc_count'] * df['layer_vs_fault_fit_rate'] / df['total runs']
        file_path = os.path.join(results_dir, f"Full_{model}.csv")
        df.to_csv(file_path, index=False)
        print(f"full file saved to {file_path}")
    
    return df

def get_fit_sums():
    weights = {
        'single': 0.48,
        'small-box': 0.15,
        'medium-box': 0.37,
        'cpu': 0.0
    }

    results_dir = "./results"

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # Iterate over all CSV files in ./results that already have been processed in the first stage (start with Full)
    for filename in os.listdir(results_dir):
        if not filename.endswith(".csv"):
            continue

        if not filename.startswith("Full"):
            continue

        file_path = os.path.join(results_dir, filename)
        model = os.path.splitext(filename)[0]

        print(f"\n\nProcessing model: {model}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"  Failed to read {file_path}: {e}")
            continue
        
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

        out_path = os.path.join(results_dir, f"ByLayerType_{model}.csv")
        by_layer_type.to_csv(out_path, index=False)
        print(f"  Saved by-layer-type to {out_path}")

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

        out_path = os.path.join(results_dir, f"ByFaultType_{model}.csv")
        by_fault_type.to_csv(out_path, index=False)
        print(f"  Saved by-fault-type to {out_path}")



if __name__ == "__main__":
    add_fit_columns()
    get_fit_sums()