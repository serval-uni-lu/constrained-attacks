from typing import Optional, Union

import numpy as np
import numpy.typing as npt


def get_feature_index(
    feature_names: Optional[npt.ArrayLike], feature_id: Union[int, str]
) -> int:
    if isinstance(feature_id, str):
        if feature_names is None:
            raise ValueError(
                f"Feature names not provided. "
                f"Impossible to convert {feature_id} to index"
            )

        feature_names = np.array(feature_names)
        index = np.where(feature_names == feature_id)[0]

        if len(index) <= 0:
            raise IndexError(f"{feature_id} is not in {feature_names}")

        return index[0]

    else:
        return feature_id
