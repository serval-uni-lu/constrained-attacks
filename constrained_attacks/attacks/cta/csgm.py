from constrained_attacks.attacks.cta.capgd import CAPGD
from constrained_attacks.utils.utils_sgm import register_hook_for_resnet, register_hook_for_densenet


class CSGM(CAPGD):
    r"""
    Same parameters as CAPG + an arhcitecture name and a gamma value
    """

    def __init__(self, *args, arch="", gamma=1, **kwargs):
        super(CSGM, self).__init__(*args, **kwargs)
        self.arch = arch
        self.gamma = gamma

        if "densenet" in arch:
            register_hook_for_densenet(self.model, arch=arch, gamma=gamma)
        elif "resnet" in arch:
            register_hook_for_resnet(self.model, arch=arch, gamma=gamma)
        else:
            raise ValueError('Current code only supports resnet/densenet. '
                             'You can extend this code to other architectures.')
