import utils.CThelper as CT

datasets = ['mayo', 'NBIA']
views = [64, 128]

for dataset in datasets:
    for view in views:
        CT.down_sample(dataset, view, 'FullViewNoiseless', train=True)
        CT.down_sample(dataset, view, 'FullViewNoiseless', train=False)
        CT.fbp_data(dataset, view, train=True)
        CT.fbp_data(dataset, view, train=False)
        
print("Done!")