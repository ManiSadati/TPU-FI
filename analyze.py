import pandas as pd
#layer,name,type,total runs,error,missclassification,sdc rate,layer area,num_ops

# Load CSV
df = pd.read_csv("fault_injection_results.csv")

# Loop through rows and check layer name
cnt = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0
total = 0


sdco = 0
sdcf = 0
sdc_e =0
sdc_e1 = 0
sdc_e2 = 0
sdc_e3 = 0
last_l = -1
mx = 0 
for idx, row in df.iterrows():
    if row['name'].lower() == 'fullyconnected':
        print(f"{cnt} Layer {row['layer']} is FullyConnected with type {row['type']} and SDC rate {row['sdc rate']}")
        if last_l != row['layer']:
            cnt += 1
            last_l = row['layer']
        # if (cnt == 48):
        #     exit()
        mx = max(mx, row['missclassification'] / row['error']    )
        sdcf += row['sdc rate'] / row['num_ops']
        sdco += row['sdc rate'] / row['layer area']
        if(row['type'] == "single"):
            sdc_e1 += row['missclassification'] / row['error']   * row['layer area']
            cnt1 += 1
        if(row['type'] == "small-box"):
            sdc_e2 += row['missclassification'] / row['error']  * row['layer area']  
            total += row['layer area']
            cnt2 += 1
        if(row['type'] == "medium-box"):
            sdc_e3 += row['missclassification'] / row['error']    * row['num_ops']
            cnt3 += 1
        sdc_e += row['missclassification'] / row['error']
print(sdc_e1 / total)
print(sdc_e2 / total)
print(sdc_e3 / total)
print(mx)