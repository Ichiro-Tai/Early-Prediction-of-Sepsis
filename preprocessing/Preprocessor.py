import pandas as pd
import json
import math
import os
import pprint


class Preprocessor:
    def __init__(self, path_vital, path_grp, path_similar, path_master, path_label) -> None:
        self._path_label = path_label
        self._path_master = path_master
        self._path_similar = path_similar
        self._path_grp_index = path_grp
        self._path_vital = path_vital
        self.df_vital = pd.read_csv(self._path_vital)
        self.group_index_dict = self.read_json_file(self._path_grp_index)

        self.index_group_dict = None
        self.index_feature_list = None
        self.feature_index_dict = None
        self.patient_time_dict = None
        self.feature_value_order_dict = dict()
        self.patient_time_record_dict = dict()
        self.patient_master_dict = dict()
        self.patient_label_dict = dict()

    
    #generate patient_label_dict.json
    def gen_patient_label_dict(self, save_path=''):
        df_labels = pd.read_csv(self._path_label, header=None, skiprows=1)
        self.patient_label_dict = df_labels.set_index(0).iloc[:, -1].to_dict()
        if save_path:
            path = self.check_path(save_path, "patient_label_dict.json")
            with open(path, 'w') as outfile:
                json.dump(self.patient_label_dict, outfile, indent=4)


    #generate patient_master_dict.json
    def gen_patient_master_dict(self, save_path):
        df_master = pd.read_csv(self._path_master)
        m_set = set()
        for _, row in df_master.iterrows():
            data = list(row)[1:]
            for i, d in enumerate(data):
                m_set.add(str(i) + str(d))
        master_list = sorted(m_set)

        for i, row in df_master.iterrows():
            patient = row.iloc[0]
            feature = ['0'] * 43
            for j, d in enumerate(row.iloc[1:]):
                m = str(j) + str(d)
                idx = master_list.index(m)
                feature[idx] = '1'
            self.patient_master_dict[patient] = ''.join(feature)
        
        if save_path:
            path = self.check_path(save_path, "patient_master_dict.json")
            with open(path, 'w') as outfile:
                json.dump(self.patient_master_dict, outfile, indent=4)


    #generate patient_time_dict.json equivalent to gen_feature_time.py
    def gen_patient_time_dict(self, save_path=''):
        tmp = self.df_vital.groupby("adm_id")["event_time"].max()
        self.patient_time_dict = tmp.to_dict()  
        if save_path:
            path = self.check_path(save_path, "patient_time_dict.json")
            with open(path, 'w') as outfile:
                json.dump(self.patient_time_dict, outfile, indent=4)

    #gnerate feature_index_dict.json
    def gen_feature_index_dict(self, save_path=''):
        exclude_labels = ['adm_id', 'event_time']
        filtered_cols = [feat for feat in self.df_vital.columns if feat not in exclude_labels] 
        filtered_cols = [val.replace(',', ';') for val in filtered_cols]
        self.feature_index_dict = {feature: index for index, feature in enumerate(filtered_cols)}
        if save_path:
            path = self.check_path(save_path, "feature_index_dict.json")
            with open(path, 'w') as outfile:
                json.dump(self.feature_index_dict, outfile, indent=4)
        

    #generate index_feature_list (index_feature_list.json)
    def gen_index_feature_list(self, save_path=''):
        exclude_labels = ['adm_id', 'event_time']   
        self.index_feature_list = [feat for feat in self.df_vital.columns if feat not in exclude_labels]  
        self.index_feature_list = [string.replace(",", ";") for string in self.index_feature_list]
        if save_path:
            path = self.check_path(save_path, "index_feature_list.json")
            with open(path, 'w') as outfile:
                json.dump(self.index_feature_list, outfile, indent=4)


    #generate index_group_dict.json
    def gen_index_group_dict(self, save_path=''):
        self.index_group_dict = {str(value): int(key) for key, values in self.group_index_dict.items() for value in values}
        if save_path:
            path = self.check_path(save_path, "index_group_dict.json")
            with open(path, 'w') as outfile:
                json.dump(self.index_group_dict, outfile, indent=4)


    #generate feature_value_order_dict.json
    def gen_feature_value_order_dict(self, save_path=''):
        vital_dict = {}
        groups = self.read_json_file(self._path_similar)
        if self.index_feature_list is None:
            self.gen_index_feature_list()
        for col in self.df_vital.columns[1:]:
            col_values = self.df_vital[col].apply(lambda x: float(x) if x != '' else None).dropna().tolist()
            vital_dict[col] = col_values
        
        vital_dict = {key.replace(",", ";"): value for key, value in vital_dict.items()}
        for group in groups:
            min_group = min(group)  # Calculate the minimum value once for each group
            for k in group:
                if k != min_group:
                    k_feature = self.index_feature_list[k]
                    min_group_feature = self.index_feature_list[min_group]
                    # Combine the values of the k_feature and min_group_feature in vital_dict
                    vital_dict[min_group_feature] = vital_dict[min_group_feature] + vital_dict[k_feature]
                    # Remove the entry for k_feature from vital_dict
                    vital_dict.pop(k_feature)

        column_names = list(self.df_vital.columns[1:])
        column_names = [val.replace(",", ";") for val in column_names]
        for col in column_names:
            if col not in vital_dict:
                continue
            value_list = sorted(vital_dict[col])
            value_order_dict = dict()
            value_minorder_dict = dict()
            value_maxorder_dict = dict()
            for i_value, value in enumerate(value_list):
                if value not in value_minorder_dict:
                    value_minorder_dict[value] = i_value
                if value == value_list[-1]:
                    value_maxorder_dict[value] = len(value_list) - 1
                    break
                if value != value_list[i_value+1]:
                    value_maxorder_dict[value] = i_value
            for value in value_maxorder_dict:
                value_order_dict[value] = (value_maxorder_dict[value] + value_minorder_dict[value]) / 2.0 / len(value_list)
            self.feature_value_order_dict[col] = value_order_dict
        if save_path:
            path = self.check_path(save_path, "feature_value_order_dict.json")
            with open(path, 'w') as outfile:
                json.dump(self.feature_value_order_dict, outfile, indent=4)
    
    
    #generate patient_time_record_dict.json equivalent to gen_vital_feature.py
    def gen_patient_time_record_dict(self, save_path=''):
        patient_time_record_dict = dict()
        if self.feature_index_dict is None:
            self.gen_feature_index_dict()
        if not bool(self.feature_value_order_dict):
            self.gen_feature_value_order_dict()
        feature_value_order_dict2 = { str(self.feature_index_dict[k]):v for k,v in self.feature_value_order_dict.items()  if 'event' not in k}
        if self.index_group_dict is None:
            self.gen_index_group_dict()
        if self.patient_time_dict is None:
            self.gen_patient_time_dict()
            
        mx_time = -100
        for index, row in self.df_vital.iterrows():
            data = row.tolist()
            patient = data[0]
            time = data[1]
            mx_time = max(mx_time, time)
            if patient not in patient_time_record_dict:
                patient_time_record_dict[patient] = dict()
            if time not in patient_time_record_dict[patient]:
                patient_time_record_dict[patient][time] = dict()
            
            vs = dict()
            data = data[2:]
            for idx, val in enumerate(data):

                if math.isnan(val):
                    continue
                if str(idx) in self.index_group_dict:
                    idx = self.index_group_dict[str(idx)]
                value_order = feature_value_order_dict2[str(idx)]
                vs[idx] = value_order[val]
            patient_time_record_dict[patient][time].update(vs)

        for p, tr in patient_time_record_dict.items(): 
            self.patient_time_record_dict[p] = dict()
            for t, vs in tr.items():
                if mx_time > 0:
                    t = int(t - self.patient_time_dict[p] - 4)
                if t < - 102:
                    continue
                nvs = []
                for k in sorted(vs.keys()):
                    nvs.append([k, vs[k]])
                self.patient_time_record_dict[p][t] = nvs
        if save_path:
            path = self.check_path(save_path, "patient_time_record_dict.json")
            with open(path, 'w') as outfile:
                json.dump(self.patient_time_record_dict, outfile, indent=4)


    def check_path(self, p, json_name=''):
        path = ''
        if ".json" not in p:
            path = os.path.join(p, json_name)
        else:
            path = p
        return path

    def read_json_file(self, p):
        with open(p, 'r') as f:
            return json.loads(f.read())
    
def preprocess_data(path_vital, path_similar, path_grp_index, generated_data_save_path, path_master, path_label):
    prep = Preprocessor(path_vital, path_grp_index, path_similar, path_master, path_label)
    #if you dont pass generated_data_save_path json wont be generated but the memeber variable
    #list will be and it will be accessible for testing. 
    prep.gen_index_feature_list(generated_data_save_path)
    prep.gen_index_group_dict(generated_data_save_path)
    prep.gen_feature_index_dict(generated_data_save_path)
    prep.gen_patient_time_dict(generated_data_save_path)

    prep.gen_feature_value_order_dict(generated_data_save_path)
    prep.gen_patient_time_record_dict(generated_data_save_path)
    prep.gen_patient_master_dict(generated_data_save_path)
    prep.gen_patient_label_dict(generated_data_save_path)



if __name__ == "__main__":
    path_vital = "../data/vital.csv"
    path_similar = "../data/similar.json"
    path_grp_index = "../data/group_index_dict.json"
    generated_data_save_path = "../generated_data"
    path_master = "../data/master.csv"
    path_label = "../data/label.csv"
    pp = Preprocessor(path_vital, path_grp_index, path_similar, path_master, path_label)
    pp.gen_patient_label_dict("../generated_data")

