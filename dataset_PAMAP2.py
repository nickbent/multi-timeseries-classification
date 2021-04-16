import pandas as pd
import torch
from torch.utils.data import DataLoader

def load_activity_map():
    map = {}
    map[0] = 'transient'
    map[1] = 'lying'
    map[2] = 'sitting'
    map[3] = 'standing'
    map[4] = 'walking'
    map[5] = 'running'
    map[6] = 'cycling'
    map[7] = 'Nordic_walking'
    map[9] = 'watching_TV'
    map[10] = 'computer_work'
    map[11] = 'car driving'
    map[12] = 'ascending_stairs'
    map[13] = 'descending_stairs'
    map[16] = 'vacuum_cleaning'
    map[17] = 'ironing'
    map[18] = 'folding_laundry'
    map[19] = 'house_cleaning'
    map[20] = 'playing_soccer'
    map[24] = 'rope_jumping'
    return map

def generate_three_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    return [x,y,z]

def generate_four_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    w = name +'_w'
    return [x,y,z,w]

def generate_cols_IMU(name):
    # temp
    temp = name+'_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name+'_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name+'_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name+'_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name+'_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name+'_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output

def load_IMU():
    output = ['time_stamp','activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output
    
def load_subjects(root='./PAMAP2_Dataset/Protocol/subject'):
    output = pd.DataFrame()
    cols = load_IMU()
    
    for i in range(101,110):
        path = root + str(i) +'.dat'
        subject = pd.read_table(path, header=None, sep='\s+')
        subject.columns = cols 
        subject['id'] = i
        output = output.append(subject, ignore_index=True)
    output.reset_index(drop=True, inplace=True)
    return output

def fix_data(data):
    data = data.drop(data[data['activity_id']==0].index)
    data = data.interpolate()
    # fill all the NaN values in a coulmn with the mean values of the column
    for colName in data.columns:
        data[colName] = data[colName].fillna(data[colName].mean())
    activity_mean = data.groupby(['activity_id']).mean().reset_index()
    return data

def data_Normalization(data):
    df_std = data.copy()
    for column in df_std.columns[2:-1]:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std



def data_id_Preprocess(data_id):
    data=data_id.groupby(['activity_id'])
    data_dict=dict()
    for i,j in data:
        data_dict[i]=torch.tensor(j.drop(columns=['activity_id',"time_stamp"]).values)
    return data_dict

#la sortie de cette fonction est un dictionnaire sur les id vers un dictionnaire sur les activity_id vers un tensor 
#contenant toutes les données
def data_Preprocess(data):
    data_ids=data.groupby(['id'])
    data_dict=dict()
    for i,j in data_ids:
        data_dict[i]=data_id_Preprocess(j.drop(columns=['id']))
    return data_dict

def pytorch_rolling_window(x, window_size, step_size=1):
    return x.unfold(0,window_size,step_size)

def split_dataset(data,id_test,stepsize):
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    train=[]
    for i in data:
        if i!=id_test:
            for j in data[i]:
                se=pytorch_rolling_window(data[i][j], 256, step_size=stepsize)
                y_train.append(torch.tensor([j]*se.shape[0]))
                #Exemple se.shape (nombre d'extraits, nb de channels, window length)
                x_train.append(se)
        else:
            for j in data[i]:
                se=pytorch_rolling_window(data[i][j], 256, step_size=stepsize)
                y_test.append(torch.tensor([j]*se.shape[0]))
                #Exemple se.shape (nombre d'extraits, nb de channels, window length)
                x_test.append(se)
    x_train,y_train,x_test,y_test=torch.cat(x_train, dim=0),torch.cat(y_train, dim=0),torch.cat(x_test, dim=0),torch.cat(y_test, dim=0)
    idx_train=torch.randperm(x_train.shape[0])
    idx_test=torch.randperm(x_test.shape[0])
    return  x_train[idx_train],y_train[idx_train],x_test[idx_test],y_test[idx_test]

class dataset_PAMAP2():
    def __init__(self,no_id_list):
        data = load_subjects()
        data = fix_data(data)
        
        #remove unused columns
        columns_dropped=[]
        for i in data.columns:
            if "orientation" in i:
                columns_dropped.append(i)
        data=data.drop(columns=columns_dropped)
        data= data[~data.id.isin(no_id_list)] #filter is not in pour enlever les sujets 8 et 9
        self.data=data
        self.map_activity_id=load_activity_map()
    
    def load_as_ListOfTensor_with_all_attributes(self,activity_list,id_test,stepsize):
        data_4_activities= self.data[self.data.activity_id.isin(activity_list)] #filter is in pour considérer que les 4 activités
        data_4_activities_normalized = data_Normalization(data_4_activities)
        data_processed=data_Preprocess(data_4_activities_normalized)
        x_train,y_train,x_test,y_test=split_dataset(data_processed,id_test,stepsize)
        
        y_train2=[activity_list.index(n) for n in y_train]
        y_test2=[activity_list.index(n) for n in y_test]
        y_train2=torch.LongTensor(y_train2)
        y_test2=torch.LongTensor(y_test2)
        
        train=[]
        test=[]
        for i in range(len(x_train)):
            train.append((x_train[i].type(torch.FloatTensor),y_train2[i])) #[2:5].type(torch.FloatTensor),y_train2[i]))

        for i in range(len(x_test)):
            test.append((x_test[i].type(torch.FloatTensor),y_test2[i]))#[2:5].type(torch.FloatTensor),y_test2[i]))
            
        return train,test
    
    def load_as_DataLoader_with_all_attributes(self,activity_list,id_test,stepsize,batch_size):        
        train,test=self.load_as_ListOfTensor_with_all_attributes(activity_list,id_test,stepsize)
        train_data=DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
         )

        test_data=DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
         )
        
        return train_data,test_data

    def load_as_ListOfTensor_with_specific_attributes(self,activity_list,id_test,stepsize,first_attribute_id,last_attribute_id):
        data_4_activities= self.data[self.data.activity_id.isin(activity_list)] #filter is in pour considérer que les 4 activités
        data_4_activities_normalized = data_Normalization(data_4_activities)
        data_processed=data_Preprocess(data_4_activities_normalized)
        x_train,y_train,x_test,y_test=split_dataset(data_processed,id_test,stepsize)
        
        y_train2=[activity_list.index(n) for n in y_train]
        y_test2=[activity_list.index(n) for n in y_test]
        y_train2=torch.LongTensor(y_train2)
        y_test2=torch.LongTensor(y_test2)
        
        train=[]
        test=[]
        for i in range(len(x_train)):
            train.append((x_train[i][first_attribute_id:last_attribute_id+1].type(torch.FloatTensor),y_train2[i]))

        for i in range(len(x_test)):
            test.append((x_test[i][first_attribute_id:last_attribute_id+1].type(torch.FloatTensor),y_test2[i]))
            
        return train,test
    
    def load_as_DataLoader_with_specific_attributes(self,activity_list,id_test,stepsize,batch_size,first_attribute_id,last_attribute_id):        
        train,test=self.load_as_ListOfTensor_with_specific_attributes(activity_list,id_test,stepsize,first_attribute_id,last_attribute_id)
        train_data=DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
         )

        test_data=DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
         )
        
        return train_data,test_data