import pandas as pd




def generate_csv_file():
    path_master = "../data/master.csv"
    path_vitals = "../data/vital.csv"
    path_label = "../data/label.csv"
    df_master = pd.read_csv(path_master)
    race_dummy = pd.get_dummies(df_master.race, prefix='race')
    gender_dummy = pd.get_dummies(df_master.gender, prefix='gender')
    admission_type_dummy = pd.get_dummies(df_master.admission_type, prefix="admType")
    admission_source_dummy = pd.get_dummies(df_master.admission_source, prefix="admSource")
    care_seeting_dummy = pd.get_dummies(df_master.care_setting, prefix="careSetting")
    age_grp_dummy = pd.get_dummies(df_master.age_grp, prefix="ageGrp")
    df_master = pd.concat([df_master,gender_dummy,race_dummy, admission_type_dummy,admission_source_dummy,care_seeting_dummy,age_grp_dummy],axis = 1)
    df_master = df_master.drop(columns = ['gender','race','admission_type','admission_source','care_setting','age_grp'])
    df_label = pd.read_csv(path_label)
    df_vitals = pd.read_csv(path_vitals)
    df_vital_master = pd.merge(df_master, df_vitals, on='adm_id')
    df_processed = pd.merge(df_vital_master, df_label, on='adm_id')
    df_processed = df_processed.drop(columns=['adm_id'])
    df_processed.to_csv("../generated_data/other_alg_file1.csv")



if __name__ == "__main__":
    generate_csv_file()
