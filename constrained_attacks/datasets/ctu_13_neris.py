from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from constrained_attacks.constraints.constraints import (
    Constraints,
    get_constraints_from_metadata,
)
from constrained_attacks.constraints.relation_constraint import Constant as Co
from constrained_attacks.constraints.relation_constraint import Feature as Fe
from constrained_attacks.constraints.relation_constraint import (
    MathOperation,
    SafeDivision,
)
from constrained_attacks.datasets.core import (
    MLC_DATA_PATH,
    DownloadFileDataset,
)
from constrained_attacks.datasets.processing import (
    get_numeric_categorical_preprocessor,
)


class Ctu13Neris(DownloadFileDataset):
    def __init__(
        self,
    ):

        data_path = f"{MLC_DATA_PATH}/ctu_13_neris/ctu_13_neris.csv"
        data_url = (
            "https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
            "thibault_simonetto_uni_lu/"
            "Ed5Wox3GpUtChu5ZYzMfACEB3PDRE3fMloUw04MSNiAkTQ?download=1"
        )
        metadata_path = (
            f"{MLC_DATA_PATH}/ctu_13_neris/ctu_13_neris_metadata.csv"
        )
        metadata_url = (
            "https://uniluxembourg-my.sharepoint.com/:x:/g/personal/"
            "thibault_simonetto_uni_lu/"
            "ESz5WJPXeF1Fv_i3vVk9cY8B-fqbrYuaaUy_e_iypmZFbQ?download=1"
        )
        targets = ["is_botnet"]
        date = "window_timestamp"
        date_format = None

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
            numeric_features=self.get_x_y()[0].columns,
            categorical_features=[],
        )

    def _get_splits(self):
        _, y = self.get_x_y()
        i = np.arange(len(y))
        i_train = i[:143046]
        i_test = i[143046:]
        i_train, i_val = train_test_split(
            i_train,
            random_state=1319,
            shuffle=True,
            stratify=y[i_train],
            test_size=0.2,
        )
        return {"train": i_train, "val": i_val, "test": i_test}

    def get_relation_constraints(self):

        x, _ = self.get_x_y()
        features = x.columns.to_list()

        def get_feature_family(
            features_l: List[str], family: str
        ) -> List[str]:
            return list(filter(lambda el: el.startswith(family), features_l))

        def sum_list_feature(features_l: List[str]) -> MathOperation:
            out = Fe(features_l[0])
            for el in features_l[1:]:
                out = out + Fe(el)
            return out

        def sum_feature_family(features_l: List[str], family: str):
            return sum_list_feature(get_feature_family(features_l, family))

        g1 = (
            sum_feature_family(features, "icmp_sum_s_")
            + sum_feature_family(features, "udp_sum_s_")
            + sum_feature_family(features, "tcp_sum_s_")
        ) == (
            sum_feature_family(features, "bytes_in_sum_s_")
            + sum_feature_family(features, "bytes_out_sum_s_")
        )

        g2 = (
            sum_feature_family(features, "icmp_sum_d_")
            + sum_feature_family(features, "udp_sum_d_")
            + sum_feature_family(features, "tcp_sum_d_")
        ) == (
            sum_feature_family(features, "bytes_in_sum_d_")
            + sum_feature_family(features, "bytes_out_sum_d_")
        )

        g_packet_size = []
        for e in ["s", "d"]:
            # -1 cause ignore last OTHER features
            bytes_outs = get_feature_family(features, f"bytes_out_sum_{e}_")[
                :-1
            ]
            pkts_outs = get_feature_family(features, f"pkts_out_sum_{e}_")[:-1]
            if len(bytes_outs) != len(pkts_outs):
                raise Exception("len(bytes_out) != len(pkts_out)")

            # Tuple of list to list of tuples
            for byte_out, pkts_out in list(zip(bytes_outs, pkts_outs)):
                g = SafeDivision(Fe(byte_out), Fe(pkts_out), Co(0.0)) <= Co(
                    1500
                )
                g_packet_size.append(g)

        g_min_max_sum = []
        for e_1 in ["bytes_out", "pkts_out", "duration"]:
            for port in [
                "1",
                "3",
                "8",
                "10",
                "21",
                "22",
                "25",
                "53",
                "80",
                "110",
                "123",
                "135",
                "138",
                "161",
                "443",
                "445",
                "993",
                "OTHER",
            ]:
                for e_2 in ["d", "s"]:
                    g_min_max_sum.extend(
                        [
                            Fe(f"{e_1}_max_{e_2}_{port}")
                            <= Fe(f"{e_1}_sum_{e_2}_{port}"),
                            Fe(f"{e_1}_min_{e_2}_{port}")
                            <= Fe(f"{e_1}_sum_{e_2}_{port}"),
                            Fe(f"{e_1}_min_{e_2}_{port}")
                            <= Fe(f"{e_1}_max_{e_2}_{port}"),
                        ]
                    )

        return [g1, g2] + g_packet_size + g_min_max_sum

    def get_constraints(self) -> Constraints:
        metadata = self.get_metadata()
        col_filter = self.get_x_y_t()[0].columns.to_list()
        return get_constraints_from_metadata(
            metadata, self.get_relation_constraints(), col_filter
        )
