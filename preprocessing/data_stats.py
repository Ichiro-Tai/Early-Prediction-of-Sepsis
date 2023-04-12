import pandas as pd



class Data_stats:
    def __init__(self, path_master, path_labels) -> None:
        self._path_labels = path_labels
        self._path_master = path_master
        self.df_master = pd.read_csv(self._path_master)
        self.df_labels = pd.read_csv(self._path_labels)
        self.df_combined = pd.merge(self.df_master, self.df_labels, on='adm_id')
        #print(self.df_combined.head())
        #print(self.df_master.head())
        self.age_groups_count = {}    
        self.gender_counts = {}
        self.race_counts = {}
        self.risk_ratios = {}

        self.get_age_stats()
        self.get_gender_stats()
        self.get_race_stats()
        self.get_risk_ratios()


    def display_risk_ratios(self):
        pass

    def display_race_stats(self):
        total_people = sum(self.race_counts.values())
        for race, count in self.race_counts.items():
            percentage = round(count / total_people * 100, 2)
            print(f"{race}: {count} ({percentage}%)")


    def display_age_stats(self):
        total_count = sum(self.age_groups_count.values())
        for age_group, count in self.age_groups_count.items():
            if age_group[0] == '>=':
                group_str = f">={age_group[1]}"
            elif age_group[0] == '<':
                group_str = f"<{age_group[1]}"
            else:
                group_str = f"{age_group[0]}~{age_group[1]}"
            percentage = (count / total_count) * 100
            print(f"Age group {group_str}: {count} people ({percentage:.2f}%)")

    
    def display_gender_stats(self):
        total_people = sum(self.gender_counts.values())
        for gender, count in self.gender_counts.items():
            percentage = round(count / total_people * 100, 2)
            print(f"{gender}: {count} ({percentage}%)")


    def get_risk_ratios(self):
        sepsis_rate = self.df_combined['sepsis2'].mean()
        no_sepsis_rate = 1 - sepsis_rate
        overall_rr = sepsis_rate / no_sepsis_rate
        #gender
        male_sepsis_rate = self.df_combined[self.df_combined['gender'] == 'Male']['sepsis2'].mean()
        female_sepsis_rate = self.df_combined[self.df_combined['gender'] == 'Female']['sepsis2'].mean()
        male_rr = male_sepsis_rate / no_sepsis_rate
        female_rr = female_sepsis_rate / no_sepsis_rate
        self.risk_ratios["male_rr"] = male_rr
        self.risk_ratios["female_rr"] = female_rr
        #race
        white_sepsis_rate = self.df_combined[self.df_combined['race'] == 'Caucasian']['sepsis'].mean()
        black_sepsis_rate = self.df_combined[self.df_combined['race'] == 'African American']['sepsis'].mean()
        asian_sepsis_rate = self.df_combined[self.df_combined['race'] == 'Asian']['sepsis'].mean()
        other_sepsis_rate = self.df_combined[self.df_combined['race'] == 'Others/unknown']['sepsis'].mean()
        white_rr = white_sepsis_rate / no_sepsis_rate
        black_rr = black_sepsis_rate / no_sepsis_rate           
        asian_rr = asian_sepsis_rate / no_sepsis_rate
        other_rr = other_sepsis_rate / no_sepsis_rate
        self.risk_ratios['white_rr'] = white_rr
        self.risk_ratios['black_rr'] = black_rr
        self.risk_ratios['asian_rr'] = asian_rr
        self.risk_ratios['other_rr'] = other_rr
        #age





    def get_race_stats(self):
        race_counts = self.df_master['race'].value_counts()
        self.race_counts = race_counts.to_dict()


    def get_gender_stats(self):
        gender_counts = self.df_master['gender'].value_counts()
        self.gender_counts = gender_counts.to_dict()       


    def get_age_stats(self):
        for age_range_str in self.df_master['age_grp']:
            age_range = self.split_age_range(age_range_str)
            if age_range in self.age_groups_count:
                self.age_groups_count[age_range] += 1
            else:
                self.age_groups_count[age_range] = 1


    def split_age_range(self, age_range_str):
        if '~' in age_range_str:
            from_age, to_age = age_range_str.split('~')
            return (int(from_age), int(to_age))
        elif '>=' in age_range_str:
            age = int(age_range_str[2:])
            return ('>=', age)
        elif '<' in age_range_str:
            age = int(age_range_str[1:])
            return ('<', age)
        



if __name__ == "__main__":
    masterPath = "../data/master.csv"
    labelsPath = "../data/label.csv"
    sts = Data_stats(masterPath, labelsPath)
    #sts.display_age_stats()
    print("---------")
    #sts.display_gender_stats()
    print("---------------")
    #sts.display_race_stats()
