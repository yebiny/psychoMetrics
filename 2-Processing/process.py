import os, sys
import argparse
import numpy as np
import pandas as pd 


def qa_process(data):
    y=(data/5)-0.5
    y=np.expand_dims(y, -1)

    return y

def edu_tp_process(data):
    y=[]
    for d in data:
        if d==1: out=-1
        elif d==2: out=-0.66
        elif d==3: out=-0.33
        elif d==4 or d==0: out=0
        elif d==5: out=0.33
        elif d==6: out=0.66
        elif d==7: out=1
        else:
            print('*VAR ERRO*')
            break;
        y.append(out)
    y = np.array(y)
    y = np.expand_dims(y, -1)

    return y

def point_one_hot(data, point_label):
    processed_data=[]
    for var in data:
        if var==point_label:
            y=[1]
        elif var!=point_label:
            if var==0:
                y=[0.5]
            else:
                y=[0]
        processed_data.append(y)
    return np.array(processed_data)


def make_xset(dataset, use_features):
    xset=[]

    for feature in use_features:
        if type(feature)==str:
            print('*',feature)
            data = dataset[feature].to_numpy()

            if 'Q' and 'A' in feature: 
                data_processed=qa_process(data)
                print('qa_process: ', data.shape,'->',data_processed.shape)

            elif 'tp' in feature or feature=='education':
                data_processed=edu_tp_process(data)
                print('edu_tp_process: ', data.shape,'->',data_processed.shape)

        elif type(feature)==list:
            print('*',feature[0])
            data = dataset[feature[0]].to_numpy()
            
            data_processed=point_one_hot(data, feature[1])
            print('point_onehot process: ', data.shape,'->',data_processed.shape)

        xset.append(data_processed)
        print('------------------------------------------')
        
    xset=np.concatenate(xset, axis=1)
    
    return xset

def make_yset(dataset):
    yset=dataset['voted'].to_numpy()
    yset=np.expand_dims(yset, -1)
    yset[yset==2]=0
    print(len(yset[yset==0]), len(yset[yset==1] )) 
    return yset

def main():
    
    opt = argparse.ArgumentParser()
    opt.add_argument(dest='csv_file', type=str)
    args=opt.parse_args()

    features_used = [
    'QbA', 'QjA', 'QkA', 'QmA', 'QnA','QoA', 'QpA','QqA', 'QsA', 'QtA',
    'tp03', 'tp04','tp06', 'tp07', 'tp08', 'tp09','education',
    ['age_group', '10s'],
    ['married', 'Never married'],
    ['urban', 'Rural'],
    ['race', 'White'],
    
    ]
    print('# of features that used: %i '%len(features_used), features_used)
   
    dataset = pd.read_csv(args.csv_file)
    
    if 'train' in args.csv_file:
        x_data = make_xset(dataset, features_used)
        np.save('x_data', x_data)
        y_data = make_yset(dataset)
        np.save('y_data', y_data)
        print('\n * Final x shape: ', x_data.shape)
        print('\n * Final y shape: ', y_data.shape)
    
    elif 'test' in args.csv_file:
        x_data = make_xset(dataset, features_used)
        np.save('x_test', x_data)
        print('\n * Final x shape: ', x_data.shape)

    else: print('Wrong csv file. check csv file.')


if __name__ == '__main__':
    main()


