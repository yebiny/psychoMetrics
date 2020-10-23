import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class eda():
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path)
        self.features = self.dataset.columns
        
    def get_label(self, feature):
        if feature=='age_group':
            label=['10s','20s','30s','40s','50s','60s','+70s']
        elif feature=='gender':
            label=['Male', 'Female']
        elif feature=='race':
            label=['Asian', 'Arab', 'Black', 'Indigenous Australian', 'Native American', 'White', 'Other']
        elif feature=='religion':
            label=['Agnostic', 'Atheist', 'Buddhist', 'Christian_Catholic', 'Christian_Mormon', 
                    'Christian_Protestant', 'Christian_Other', 'Hindu', 'Jewish', 'Muslim', 'Sikh', 'Other']
        return label
    
    def get_info(self, idx, show='y'):

        feature = self.features[idx]
        data = self.dataset[feature]
        if data.dtype==object:
            data = self.convert_object2int(feature, data)
        
        if show=='y':
            print(data[:2])
        
        return feature, data

    def convert_object2int(self, feature, data):
        label=self.get_label(feature)
        new_data = []
        for d in data:
            new_var = np.where(np.array(label)==d)[0][0]
            new_data.append(new_var)

        return np.array(new_data)
    
    def plot_hist(self, idx):
        feature, data = self.get_info(idx, show='n')
        
        plt.title(feature)
        if np.max(data)<10 and data.dtype==int:
            plt.hist(data, range=(0, np.max(data)+1), bins=np.max(data)+1, color='g', alpha=0.6, edgecolor='w',linewidth=1.3)
            plt.xticks([i for i in range(np.max(data)+1)])
            plt.yticks([])
            for i in range(0, np.max(data)+1):
                n = len(data[data==i])
                plt.text(x = i+0.1 , y = n+0.1, s = n, size = 9)
        else:
            print('test')
            plt.hist(data, color='g', alpha=0.6)


    def plot_multi_hist(self, idx_list, r=5):
        fig = plt.figure(figsize=(r*4,15))
        for idx in range(len(idx_list)):
            plt.subplot(3,r,idx+1)
            self.plot_hist(idx_list[idx])
        plt.show()

