from torchattacks import LGV
from constrained_attacks.attacks.cta.capgd import CAPGD

class CLGV(LGV):
    r"""
        LGV attack in the paper 'LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity'
        [https://arxiv.org/abs/2207.13129]

        Arguments:
            model (nn.Module): initial model to attack.
            trainloader (torch.utils.data.DataLoader): data loader of the unnormalized train set. Must load data in [0, 1].
            Be aware that the batch size may impact success rate. The original paper uses a batch size of 256. A different
            batch-size might require to tune the learning rate.
            lr (float): constant learning rate to collect models. In the paper, 0.05 is best for ResNet-50. 0.1 seems best
            for some other architectures. (Default: 0.05)
            epochs (int): number of epochs. (Default: 10)
            nb_models_epoch (int): number of models to collect per epoch. (Default: 4)
            wd (float): weight decay of SGD to collect models. (Default: 1e-4)
            n_grad (int): number of models to ensemble at each attack iteration. 1 (default) is recommended for efficient
            iterative attacks. Higher numbers give generally better results at the expense of computations. -1 uses all
            models (should be used for single-step attacks like FGSM).
            verbose (bool): print progress. Install the tqdm package for better print. (Default: True)

        .. note:: If a list of models is not provided to `load_models()`, the attack will start by collecting models along
        the SGD trajectory for `epochs` epochs with the constant learning rate `lr`.

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height`
            and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.

        Examples::
            >>> attack = torchattacks.LGV(model, trainloader, lr=0.05, epochs=10, nb_models_epoch=4, wd=1e-4, n_grad=1, attack_class=BIM, eps=4/255, alpha=4/255/10, steps=50, verbose=True)
            >>> attack.collect_models()
            >>> attack.save_models('./models/lgv/')
            >>> adv_images = attack(images, labels)
        """

    def __init__(self, model, trainloader, lr=0.05, epochs=10, nb_models_epoch=4, wd=1e-4, n_grad=1, verbose=True,
                 attack_class=CAPGD,attack_attr={}):

        super(CLGV, self).__init__(model, trainloader, lr, epochs, nb_models_epoch, wd, n_grad, verbose,attack_class, **attack_attr )
