import pandas as pd
from collections import defaultdict
from tabulate import tabulate


class Data_stats:
    def __init__(self, path_master, path_labels) -> None:
        self._path_labels = path_labels
        self._path_master = path_master
        self.df_master = pd.read_csv(self._path_master)
        self.df_labels = pd.read_csv(self._path_labels)
        self.df_combined = pd.merge(self.df_master, self.df_labels, on='adm_id')
        self.age_groups_count = {}    
        self.gender_counts = {}
        self.race_counts = {}
        self.total_n = 0
        self.risk_ratios = defaultdict(dict)
        self.age_groups_strings = []

        self.calc_age_stats()
        self.calc_gender_stats()
        self.calc_race_stats()
        self.calc_risk_ratios()


    def display_all_data(self):
        self.table = [
            ["", f"People (n={self.total_n})", "Risk ratio"],
            ["Gender", "", ""],
            ["Female", self.gender_counts["Female"], "{:.2f}".format(self.risk_ratios['gender']['female_rr'])],
            ["Male", self.gender_counts["Male"], "{:.2f}".format(self.risk_ratios['gender']['male_rr'])],
            ["Race", "", ""],
            ["African American", self.race_counts["African American"], "{:.2f}".format(self.risk_ratios['race']['black_rr'])],
            ["Asian", self.race_counts["Asian"], "{:.2f}".format(self.risk_ratios['race']['asian_rr'])],
            ["Caucasian", self.race_counts["Caucasian"], "{:.2f}".format(self.risk_ratios['race']['white_rr'])],
            ["Other/unknown", self.race_counts["Others/unknown"], "{:.2f}".format(self.risk_ratios['race']['other_rr'])],
            ["Age", "", ""],
        ]
        self.risk_ratios['age'] = dict(sorted(self.risk_ratios['age'].items(), key=lambda x: x[0][0]))
        for a, b in self.risk_ratios['age'].items():
            key1, key2 = a
            age_grp_str = f"{key1}-{key2}"
            self.table.append([age_grp_str, self.age_groups_count[a], "{:.2f}".format(b)])
        print(tabulate(self.table, headers='firstrow'))

    def display_race_stats(self):
        total_people = sum(self.race_counts.values())
        for race, count in self.race_counts.items():
            percentage = round(count / total_people * 100, 2)
            print(f"{race}: {count} ({percentage}%)")

    def display_age_stats(self):
        total_count = sum(self.age_groups_count.values())
        for age_group, count in self.age_groups_count.items():
            group_str = f"{age_group[0]}-{age_group[1]}"
            percentage = (float(count) / total_count) * 100
            print(f"Age group {group_str}: {count} people ({percentage:.2f}%)")

    
    def display_gender_stats(self):
        total_people = sum(self.gender_counts.values())
        for gender, count in self.gender_counts.items():
            percentage = round(count / total_people * 100, 2)
            print(f"{gender}: {count} ({percentage}%)")


    def calc_risk_ratios(self):
        sepsis_rate = self.df_combined['sepsis2'].mean()
        no_sepsis_rate = 1 - sepsis_rate
        overall_rr = sepsis_rate / no_sepsis_rate
        #gender
        male_sepsis_rate = self.df_combined[self.df_combined['gender'] == 'Male']['sepsis2'].mean()
        female_sepsis_rate = self.df_combined[self.df_combined['gender'] == 'Female']['sepsis2'].mean()
        male_rr = male_sepsis_rate / no_sepsis_rate
        female_rr = female_sepsis_rate / no_sepsis_rate
        self.risk_ratios['gender']['male_rr'] = male_rr
        self.risk_ratios['gender']['female_rr'] = female_rr
        #race
        white_sepsis_rate = self.df_combined[self.df_combined['race'] == 'Caucasian']['sepsis2'].mean()
        black_sepsis_rate = self.df_combined[self.df_combined['race'] == 'African American']['sepsis2'].mean()
        asian_sepsis_rate = self.df_combined[self.df_combined['race'] == 'Asian']['sepsis2'].mean()
        other_sepsis_rate = self.df_combined[self.df_combined['race'] == 'Others/unknown']['sepsis2'].mean()
        white_rr = white_sepsis_rate / no_sepsis_rate
        black_rr = black_sepsis_rate / no_sepsis_rate           
        asian_rr = asian_sepsis_rate / no_sepsis_rate
        other_rr = other_sepsis_rate / no_sepsis_rate
        self.risk_ratios['race']['white_rr'] = white_rr
        self.risk_ratios['race']['black_rr'] = black_rr
        self.risk_ratios['race']['asian_rr'] = asian_rr
        self.risk_ratios['race']['other_rr'] = other_rr
        #age
        age_counts = self.df_combined.groupby('age_grp').size()
        sepsis_counts = self.df_combined.groupby('age_grp')['sepsis2'].sum().values
        overall_sepsis_count = self.df_combined['sepsis2'].sum()
        age_sepsis_rates = sepsis_counts / age_counts
        age_rrs = age_sepsis_rates / (overall_sepsis_count / len(self.df_combined))
        age_rrs_dict = age_rrs.to_dict()
        age_rrs_dict = {self.get_tuple_key(key): val for key, val in age_rrs_dict.items()}
        self.risk_ratios['age'] = age_rrs_dict

    def get_tuple_key(self, oldKey):
        if '~' in oldKey:
            from_age, to_age = oldKey.split('~')
            lowerAge = int(from_age)
            higherAge = int(to_age)
            if lowerAge > higherAge:
                lowerAge, higherAge = higherAge, lowerAge
            return (lowerAge, higherAge)
        elif '>=' in oldKey:
            age = int(oldKey[2:])
            return (age, 100)
        elif '<' in oldKey:
            age = int(oldKey[1:])
            return (0, age)

    def calc_race_stats(self):
        race_counts = self.df_master['race'].value_counts()
        self.race_counts = race_counts.to_dict()
        self.total_n = sum(self.race_counts.values())
        
    def calc_gender_stats(self):
        gender_counts = self.df_master['gender'].value_counts()
        self.gender_counts = gender_counts.to_dict()       

    def calc_age_stats(self):
        for age_range_str in self.df_master['age_grp']:
            age_range = self.split_age_range(age_range_str)
            if age_range in self.age_groups_count:
                self.age_groups_count[age_range] += 1
            else:
                self.age_groups_count[age_range] = 1

    def split_age_range(self, age_range_str):
        if age_range_str not in self.age_groups_strings:
            self.age_groups_strings.append(age_range_str)
        return self.get_tuple_key(age_range_str)


if __name__ == "__main__":
    masterPath = "../data/master.csv"
    labelsPath = "../data/label.csv"
    data_stats = Data_stats(masterPath, labelsPath)
    #data_stats.display_age_stats()
    #print("---------")
    #data_stats.display_gender_stats()
    #print("---------------")
    #data_stats.display_race_stats()
    #data_stats.display_risk_ratios()
    #data_stats.calc_risk_ratios()
    data_stats.display_all_data()

