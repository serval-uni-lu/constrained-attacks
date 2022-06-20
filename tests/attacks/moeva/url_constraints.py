from typing import List

from constrained_attacks.constraints.constraints import (
    get_constraints_from_file,
)
from constrained_attacks.constraints.relation_constraint import (
    BaseRelationConstraint,
    Constant,
    Feature,
)


def get_url_relation_constraints() -> List[BaseRelationConstraint]:
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


def get_url_constraints():
    return get_constraints_from_file(
        "./tests/resources/url/features.csv", get_url_relation_constraints()
    )
