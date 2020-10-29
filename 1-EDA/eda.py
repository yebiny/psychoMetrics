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
        
        data = np.array(data)

        return feature, data

    def convert_object2int(self, feature, data):
        label=self.get_label(feature)
        new_data = []
        for d in data:
            new_var = np.where(np.array(label)==d)[0][0]
            new_data.append(new_var)

        return np.array(new_data)
    
    def plot_hist(self, idx, vote=0):
        feature, data = self.get_info(idx, show='n')
        if np.max(data)>10:
            plt.hist(data, color='g', alpha=0.6, bins=(1000))
            plt.yscale('log')
        else: 
            data = np.int64(data)
        
        plt.title(feature)
        if vote==1:
            data1=data[self.dataset['voted']==1]
            data2=data[self.dataset['voted']==2]
            plt.hist([data1,data2], range=(0, np.max(data1)+1), bins=np.max(data1)+1,
                    color=['dodgerblue','orangered'], alpha=0.8)
        else:
             if np.max(data)<10:
                 plt.hist(data, range=(0, np.max(data)+1), bins=np.max(data)+1, color='g', alpha=0.6, edgecolor='w',linewidth=1.3)
                 plt.xticks([i for i in range(np.max(data)+1)])
                 plt.yticks([])
                 for i in range(0, np.max(data)+1):
                     n = len(data[data==i])
                     plt.text(x = i+0.1 , y = n+0.1, s = n, size = 9)


    def plot_multi_hist(self, idx_list, vote=0, r=5):
        fig = plt.figure(figsize=(r*4,20))
        for idx in range(len(idx_list)):
            plt.subplot(4,r,idx+1)
            self.plot_hist(idx_list[idx], vote=vote)
        plt.show()

