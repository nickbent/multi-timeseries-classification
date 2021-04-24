import pandas as pd
import torch
from torch.utils.data import DataLoader

def load_fault_map():
    map = {}
    map[0] = 'A/C Feed Ratio, B Composition Constante (Stream 4)'
    map[1] = 'B Composition, A/C Ration Constant (Stream 4)'
    map[2] = 'D Feed Temperature (Stream 2)'
    map[3] = 'Reactor Cooling Water Inlet Temperature'
    map[4] = 'Condenser Cooling Water Inlet Temperature'
    map[5] = 'A Feed Loss (Stream 1)'
    map[6] = 'C Header Pressure Loss - Reduced Availability (Stream 4)'
    map[7] = 'A, B, C Feed Composition (Stream 4)'
    map[8] = 'D Feed Temperature (Stream 2)'
    map[9] = 'C Feed Temperature (Stream 4)'
    map[10] = 'Reactor Cooling Water Inlet Temperature'
    map[11] = 'Condenser Cooling Water Inlet Temperature'
    map[12] = 'Reaction Kinetics'
    map[13] = 'Reactor Cooling Water Valve'
    map[15] = 'Unknown'
    map[16] = 'Unknown'
    map[17] = 'Unknown'
    map[18] = 'Unknown'
    map[19] = 'Unknown'
    map[20] = 'The valve for Stream 4 was fixed at the steady state position'
    return map

def load_dataset(filepath='../data/TEP_train.csv'):
    return pd.read_csv(filepath)

def data_Normalization(data):
    df_std = data.copy()
    for column in df_std.columns[3:-1]:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std

def data_simulationGroup_Preprocess(data_id):
    data=data_id.groupby(['simulationGroup'])
    data_dict=dict()
    for i,j in data:
        data_dict[i]=torch.tensor(j.drop(columns=['simulationGroup',"simulationRun", "sample"]).values)
    return data_dict

def data_preprocess(data):
    data_ids=data.groupby(['faultNumber'])
    data_dict=dict()
    for i,j in data_ids:
        data_dict[i]=data_simulationGroup_Preprocess(j.drop(columns=['faultNumber']))
    return data_dict

def pytorch_rolling_window(x, window_size, step_size=1):
    return x.unfold(0,window_size,step_size)

def split_dataset(data, simulationGroup_test, window_size, stepsize):
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    train=[]
    for i in data:
        if i!=simulationGroup_test:
            for j in data[i]:
                if data[i][j].shape[0]>window_size:
                    se=pytorch_rolling_window(data[i][j], window_size, step_size=stepsize)
                    y_train.append(torch.tensor([j]*se.shape[0]))
                    #Exemple se.shape (nombre d'extraits, nb de channels, window length)
                    x_train.append(se)
        else:
            for j in data[i]:
                if data[i][j].shape[0]>window_size:
                    se=pytorch_rolling_window(data[i][j], window_size, step_size=stepsize)
                    y_test.append(torch.tensor([j]*se.shape[0]))
                    x_test.append(se)
    x_train,y_train,x_test,y_test=torch.cat(x_train, dim=0),torch.cat(y_train, dim=0),torch.cat(x_test, dim=0),torch.cat(y_test, dim=0)
    idx_train=torch.randperm(x_train.shape[0])
    idx_test=torch.randperm(x_test.shape[0])
    return  x_train[idx_train],y_train[idx_train],x_test[idx_test],y_test[idx_test]

class dataset_TEP():
    def __init__(self):
        data = load_dataset()
        self.map_fault_id = load_fault_map()

        # Simulate the different participant by partitioning the simulation runs in 5 groups
        simulationGroup = []
        for i in data['simulationRun']:
            if i <= 100:
                simulationGroup.append(1)
            elif i > 100 and i<= 200:
                simulationGroup.append(2)
            elif i > 200 and i<= 300:
                simulationGroup.append(3)
            elif i > 300 and i<= 400:
                simulationGroup.append(4)
            elif i > 400 and i<= 500:
                simulationGroup.append(5)

        data['simulationGroup'] = simulationGroup
        self.data = data

    def load_as_ListOfTensor_with_all_attributes(self, fault_list, simulationGroup_test, window_size, stepsize):
        data = data_Normalization(self.data)
        data = data_preprocess(data)
        x_train,y_train,x_test,y_test = split_dataset(data, simulationGroup_test, window_size, stepsize)

        y_train2 = [fault_list.index(n) for n in y_train]
        y_test2 = [fault_list.index(n) for n in y_test]
        y_train2 = torch.LongTensor(y_train2)
        y_test2 = torch.LongTensor(y_test2)
                
        train = []
        test = []
        for i in range(len(x_train)):
            train.append((x_train[i].type(torch.FloatTensor),y_train2[i]))

        for i in range(len(x_test)):
            test.append((x_test[i].type(torch.FloatTensor),y_test2[i]))

        return train, test

    def load_as_DataLoader_with_all_attributes(self, fault_list, simulationGroup_test, window_size, stepsize, batch_size):        
        train, test = self.load_as_ListOfTensor_with_all_attributes(fault_list, simulationGroup_test, window_size, stepsize)
        train_data = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=None,
        pin_memory=False,
         )

        test_data=DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=None,
        pin_memory=False,
         )
        
        return train_data,test_data