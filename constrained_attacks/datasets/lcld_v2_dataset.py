import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from constrained_attacks.constraints.constraints import (
    Constraints,
    get_constraints_from_metadata,
)
from constrained_attacks.constraints.relation_constraint import Constant as Co
from constrained_attacks.constraints.relation_constraint import Feature as Fe
from constrained_attacks.constraints.relation_constraint import SafeDivision
from constrained_attacks.datasets.core import (
    MLC_DATA_PATH,
    DownloadFileDataset,
)
from constrained_attacks.datasets.processing import (
    get_numeric_categorical_preprocessor,
)


class LcldV2TimeDataset(DownloadFileDataset):
    def __init__(
        self,
    ):

        data_path = f"{MLC_DATA_PATH}/lcld_v2/lcld_v2.csv"
        data_url = (
            "https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
            "thibault_simonetto_uni_lu/"
            "EeKpMXnQNo9CuRaLcHNjiX0B4Tf2H_HV3OKmlqvwbZZ-aA?download=1"
        )
        metadata_path = f"{MLC_DATA_PATH}/lcld_v2/lcld_v2_metadata.csv"
        metadata_url = (
            "https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
            "thibault_simonetto_uni_lu/"
            "EfIRH-UBrW1BiYwvSjlyZIIBFbOiALcxUzdZ5T11qhILlw?download=1"
        )
        targets = ["charged_off"]
        date = "issue_d"
        date_format = None
        self.numerical_features = [
            "loan_amnt",
            "term",
            "int_rate",
            "installment",
            "sub_grade",
            "emp_length",
            "annual_inc",
            "dti",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "total_acc",
            "mort_acc",
            "pub_rec_bankruptcies",
            "fico_score",
            "initial_list_status",
            "application_type",
            "month_of_year",
            "ratio_loan_amnt_annual_inc",
            "ratio_open_acc_total_acc",
            "month_since_earliest_cr_line",
            "ratio_pub_rec_month_since_earliest_cr_line",
            "ratio_pub_rec_bankruptcies_month_since_earliest_cr_line",
            "ratio_pub_rec_bankruptcies_pub_rec",
        ]
        self.categorical_features = [
            "home_ownership",
            "purpose",
            "verification_status",
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
        _, _, t = self.get_x_y_t()
        i_train = np.where(
            (pd.to_datetime("2013-01-01") <= t)
            & (t <= pd.to_datetime("2015-06-30"))
        )[0]
        i_val = np.where(
            (pd.to_datetime("2015-07-01") <= t)
            & (t <= pd.to_datetime("2015-12-31"))
        )[0]
        i_test = np.where(
            (pd.to_datetime("2016-01-01") <= t)
            & (t <= pd.to_datetime("2017-12-31"))
        )[0]
        return {"train": i_train, "val": i_val, "test": i_test}

    def get_relation_constraints(self):

        int_rate = Fe("int_rate") / Co(1200)
        term = Fe("term")
        installment = Fe("loan_amnt") * (
            (int_rate * ((Co(1) + int_rate) ** term))
            / ((Co(1) + int_rate) ** term - Co(1))
        )
        g1 = Fe("installment") == installment

        g2 = Fe("open_acc") <= Fe("total_acc")

        g3 = Fe("pub_rec_bankruptcies") <= Fe("pub_rec")

        g4 = (Fe("term") == Co(36)) | (Fe("term") == Co(60))

        g5 = Fe("ratio_loan_amnt_annual_inc") == (
            Fe("loan_amnt") / Fe("annual_inc")
        )

        g6 = Fe("ratio_open_acc_total_acc") == (
            Fe("open_acc") / Fe("total_acc")
        )

        # g7 was diff_issue_d_earliest_cr_line
        # g7 is not necessary in v2
        # issue_d and d_earliest cr_line are replaced
        # by month_since_earliest_cr_line

        g8 = Fe("ratio_pub_rec_month_since_earliest_cr_line") == (
            Fe("pub_rec") / Fe("month_since_earliest_cr_line")
        )

        g9 = Fe("ratio_pub_rec_bankruptcies_month_since_earliest_cr_line") == (
            Fe("pub_rec_bankruptcies") / Fe("month_since_earliest_cr_line")
        )

        g10 = Fe("ratio_pub_rec_bankruptcies_pub_rec") == SafeDivision(
            Fe("pub_rec_bankruptcies"), Fe("pub_rec"), Co(-1)
        )

        return [g1, g2, g3, g4, g5, g6, g8, g9, g10]

    def get_constraints(self) -> Constraints:
        metadata = self.get_metadata()
        col_filter = self.get_x_y_t()[0].columns.to_list()
        return get_constraints_from_metadata(
            metadata, self.get_relation_constraints(), col_filter
        )


class LcldV2IidDataset(LcldV2TimeDataset):
    def _get_splits(self):
        x, y = self.get_x_y()

        splits = super(LcldV2IidDataset, self)._get_splits()
        merged = np.concatenate(
            [splits.get("train"), splits.get("val"), splits.get("test")]
        )
        i_train, i_test = train_test_split(
            merged,
            random_state=100,
            shuffle=True,
            stratify=y[merged],
            test_size=splits.get("test").shape[0],
        )
        i_train, i_val = train_test_split(
            i_train,
            random_state=200,
            shuffle=True,
            stratify=y[i_train],
            test_size=splits.get("val").shape[0],
        )
        return {"train": i_train, "val": i_val, "test": i_test}
