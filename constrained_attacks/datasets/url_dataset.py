import numpy as np
from sklearn.model_selection import train_test_split

from constrained_attacks.constraints.constraints import (
    Constraints,
    get_constraints_from_metadata,
)
from constrained_attacks.constraints.relation_constraint import (
    Constant,
    Feature,
)
from constrained_attacks.datasets.core import (
    MLC_DATA_PATH,
    DownloadFileDataset,
)
from constrained_attacks.datasets.processing import (
    get_numeric_categorical_preprocessor,
)


class UrlDataset(DownloadFileDataset):
    def __init__(self):
        data_path = f"{MLC_DATA_PATH}/url/url.csv"
        data_url = (
            "https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
            "thibault_simonetto_uni_lu/"
            "Eb8cqp1UNTBOmpqPb3-Ajr4BJGvyjqvvaYP5M0dDM7k4pg?download=1"
        )
        metadata_path = f"{MLC_DATA_PATH}/url/url_metadata.csv"
        metadata_url = (
            "https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
            "thibault_simonetto_uni_lu/"
            "EeWjdreWcLRPiOo067kzmf0Bzmli13-nsCKsv6-IBXAAYg?download=1"
        )
        date = None
        date_format = None
        targets = ["is_phishing"]
        self.numerical_features = [
            "length_url",
            "length_hostname",
            "ip",
            "nb_dots",
            "nb_hyphens",
            "nb_at",
            "nb_qm",
            "nb_and",
            "nb_or",
            "nb_eq",
            "nb_underscore",
            "nb_tilde",
            "nb_percent",
            "nb_slash",
            "nb_star",
            "nb_colon",
            "nb_comma",
            "nb_semicolumn",
            "nb_dollar",
            "nb_space",
            "nb_www",
            "nb_com",
            "nb_dslash",
            "http_in_path",
            "https_token",
            "ratio_digits_url",
            "ratio_digits_host",
            "punycode",
            "port",
            "tld_in_path",
            "tld_in_subdomain",
            "abnormal_subdomain",
            "nb_subdomains",
            "prefix_suffix",
            "random_domain",
            "shortening_service",
            "path_extension",
            "nb_redirection",
            "nb_external_redirection",
            "length_words_raw",
            "char_repeat",
            "shortest_words_raw",
            "shortest_word_host",
            "shortest_word_path",
            "longest_words_raw",
            "longest_word_host",
            "longest_word_path",
            "avg_words_raw",
            "avg_word_host",
            "avg_word_path",
            "phish_hints",
            "domain_in_brand",
            "brand_in_subdomain",
            "brand_in_path",
            "suspecious_tld",
            "statistical_report",
            "whois_registered_domain",
            "domain_registration_length",
            "domain_age",
            "web_traffic",
            "dns_record",
            "google_index",
            "page_rank",
        ]

        self.categorical_features = []
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

    def get_relation_constraints(self):
        def apply_if_a_supp_zero_than_b_supp_zero(a: Feature, b: Feature):
            return (Constant(0) <= a) or (Constant(0) < b)

        g1 = Feature(1) <= Feature(0)

        intermediate_sum = Constant(0)
        for i in range(3, 18):
            intermediate_sum = intermediate_sum + Feature(i)
        intermediate_sum = intermediate_sum + (Constant(3) * Feature(19))

        g2 = intermediate_sum <= Feature(0)

        # g3: if x[:, 21] > 0 then x[:,3] > 0
        g3 = apply_if_a_supp_zero_than_b_supp_zero(Feature(21), Feature(3))

        # g4: if x[:, 23] > 0 then x[:,13] > 0
        g4 = apply_if_a_supp_zero_than_b_supp_zero(Feature(23), Feature(13))

        intermediate_sum = (
            Constant(3) * Feature(20)
            + Constant(4) * Feature(21)
            + Constant(2) * Feature(23)
        )
        g5 = intermediate_sum <= Feature(0)

        # g6: if x[:, 19] > 0 then x[:,25] > 0
        g6 = apply_if_a_supp_zero_than_b_supp_zero(Feature(19), Feature(25))

        # g7: if x[:, 19] > 0 then x[:,26] > 0
        # g7 = apply_if_a_supp_zero_than_b_supp_zero(19, 26)

        # g8: if x[:, 2] > 0 then x[:,25] > 0
        g8 = apply_if_a_supp_zero_than_b_supp_zero(Feature(2), Feature(25))

        # g9: if x[:, 2] > 0 then x[:,26] > 0
        # g9 = apply_if_a_supp_zero_than_b_supp_zero(2, 26)

        # g10: if x[:, 28] > 0 then x[:,25] > 0
        g10 = apply_if_a_supp_zero_than_b_supp_zero(Feature(28), Feature(25))

        # g11: if x[:, 31] > 0 then x[:,26] > 0
        g11 = apply_if_a_supp_zero_than_b_supp_zero(Feature(31), Feature(26))

        # x[:,38] <= x[:,37]
        g12 = Feature(38) <= Feature(37)

        g13 = (Constant(3) * Feature(20)) <= (Feature(0) + Constant(1))

        g14 = (Constant(4) * Feature(21)) <= (Feature(0) + Constant(1))

        g15 = (Constant(4) * Feature(2)) <= (Feature(0) + Constant(1))
        g16 = (Constant(2) * Feature(23)) <= (Feature(0) + Constant(1))

        return [
            g1,
            g2,
            g3,
            g4,
            g5,
            g6,
            g8,
            g10,
            g11,
            g12,
            g13,
            g14,
            g15,
            g16,
        ]

    def get_constraints(self) -> Constraints:
        metadata = self.get_metadata()
        col_filter = self.get_x_y_t()[0].columns.to_list()
        return get_constraints_from_metadata(
            metadata, self.get_relation_constraints(), col_filter
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
