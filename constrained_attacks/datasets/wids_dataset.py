import numpy as np
from sklearn.model_selection import train_test_split

from constrained_attacks.constraints.constraints import (
    Constraints,
    get_constraints_from_metadata,
)
from constrained_attacks.constraints.relation_constraint import Feature
from constrained_attacks.datasets.core import (
    MLC_DATA_PATH,
    DownloadFileDataset,
)
from constrained_attacks.datasets.processing import (
    get_numeric_categorical_preprocessor,
)


class WidsDataset(DownloadFileDataset):
    def __init__(
        self,
    ):

        data_path = f"{MLC_DATA_PATH}/wids/wids.csv"
        data_url = "https://uniluxembourg-my.sharepoint.com/:x:/g/personal/thibault_simonetto_uni_lu/ESKij8CqW6dAlIe3bW-6rk0BCyDZ7B1gLYI3OTLdlH2wGg?download=1"
        metadata_path = f"{MLC_DATA_PATH}/wids/wids_metadata.csv"
        metadata_url = "https://uniluxembourg-my.sharepoint.com/:x:/g/personal/thibault_simonetto_uni_lu/EWZgana5BVJNgvCiM1I-Y7MBhXOw7MrxyjGWMg2FRnMCBA?download=1"
        targets = ["hospital_death"]
        date = None
        date_format = None
        self.numerical_features = [
            "age",
            "bmi",
            "elective_surgery",
            "height",
            "pre_icu_los_days",
            "readmission_status",
            "weight",
            "apache_2_diagnosis",
            "apache_3j_diagnosis",
            "apache_post_operative",
            "arf_apache",
            "bun_apache",
            "creatinine_apache",
            "gcs_eyes_apache",
            "gcs_motor_apache",
            "gcs_unable_apache",
            "gcs_verbal_apache",
            "glucose_apache",
            "heart_rate_apache",
            "hematocrit_apache",
            "intubated_apache",
            "map_apache",
            "resprate_apache",
            "sodium_apache",
            "temp_apache",
            "ventilated_apache",
            "wbc_apache",
            "d1_diasbp_max",
            "d1_diasbp_min",
            "d1_diasbp_noninvasive_max",
            "d1_diasbp_noninvasive_min",
            "d1_heartrate_max",
            "d1_heartrate_min",
            "d1_mbp_max",
            "d1_mbp_min",
            "d1_mbp_noninvasive_max",
            "d1_mbp_noninvasive_min",
            "d1_resprate_max",
            "d1_resprate_min",
            "d1_spo2_max",
            "d1_spo2_min",
            "d1_sysbp_max",
            "d1_sysbp_min",
            "d1_sysbp_noninvasive_max",
            "d1_sysbp_noninvasive_min",
            "d1_temp_max",
            "d1_temp_min",
            "h1_diasbp_max",
            "h1_diasbp_min",
            "h1_diasbp_noninvasive_max",
            "h1_diasbp_noninvasive_min",
            "h1_heartrate_max",
            "h1_heartrate_min",
            "h1_mbp_max",
            "h1_mbp_min",
            "h1_mbp_noninvasive_max",
            "h1_mbp_noninvasive_min",
            "h1_resprate_max",
            "h1_resprate_min",
            "h1_spo2_max",
            "h1_spo2_min",
            "h1_sysbp_max",
            "h1_sysbp_min",
            "h1_sysbp_noninvasive_max",
            "h1_sysbp_noninvasive_min",
            "h1_temp_max",
            "h1_temp_min",
            "d1_bun_max",
            "d1_bun_min",
            "d1_calcium_max",
            "d1_calcium_min",
            "d1_creatinine_max",
            "d1_creatinine_min",
            "d1_glucose_max",
            "d1_glucose_min",
            "d1_hco3_max",
            "d1_hco3_min",
            "d1_hemaglobin_max",
            "d1_hemaglobin_min",
            "d1_hematocrit_max",
            "d1_hematocrit_min",
            "d1_platelets_max",
            "d1_platelets_min",
            "d1_potassium_max",
            "d1_potassium_min",
            "d1_sodium_max",
            "d1_sodium_min",
            "d1_wbc_max",
            "d1_wbc_min",
            "apache_4a_hospital_death_prob",
            "apache_4a_icu_death_prob",
            "aids",
            "cirrhosis",
            "diabetes_mellitus",
            "hepatic_failure",
            "immunosuppression",
            "leukemia",
            "lymphoma",
            "solid_tumor_with_metastasis",
        ]

        self.categorical_features = [
            "ethnicity",
            "gender",
            "hospital_admit_source",
            "icu_admit_source",
            "icu_stay_type",
            "icu_type",
            "apache_3j_bodysystem",
            "apache_2_bodysystem",
            "apache_3j_diagnosis_split0",
        ]

        super().__init__(
            targets,
            data_path,
            data_url,
            metadata_path,
            metadata_url,
            date,
            date_format,
        )

    def get_preprocessor(self):
        return get_numeric_categorical_preprocessor(
            numeric_features=self.numerical_features,
            categorical_features=self.categorical_features,
        )

    def _get_splits(self):
        x, y = self.get_x_y()

        merged = np.arange(len(x))
        i_train, i_test = train_test_split(
            merged,
            random_state=100,
            shuffle=True,
            stratify=y[merged],
            test_size=0.2,
        )
        i_train, i_val = train_test_split(
            i_train,
            random_state=200,
            shuffle=True,
            stratify=y[i_train],
            test_size=0.2,
        )
        return {"train": i_train, "val": i_val, "test": i_test}

    def get_relation_constraints(self):
        g_min_max = []
        for i in range(27, 88, 2):
            g_min_max.append(Feature(i + 1) <= Feature(i))
        return g_min_max

    def get_constraints(self) -> Constraints:
        metadata = self.get_metadata()
        col_filter = self.get_x_y_t()[0].columns.to_list()
        return get_constraints_from_metadata(
            metadata, self.get_relation_constraints(), col_filter
        )
