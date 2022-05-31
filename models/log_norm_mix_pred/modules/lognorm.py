import torch.distributions as D
from torch.distributions import Normal, MixtureSameFamily, Categorical, TransformedDistribution
import torch

class LogNormMix(TransformedDistribution):
    def __init__(self, locs: torch.Tensor, log_scales: torch.Tensor,
                        log_weights: torch.Tensor):
                        # mean_log_inter_time: float=0.0,
                        # std_log_inter_time: float=1.0):
        """
        Mixture of log-normal distributions.
        We model it in the following way (see Appendix D.2 in the paper):
        x ~ GaussianMixtureModel(locs, log_scales, log_weights)
        y = std_log_inter_time * x + mean_log_inter_time
        z = exp(y)
        Args:
            locs: Location parameters of the component distributions,
                shape (batch_size, seq_len, num_mix_components)
            log_scales: Logarithms of scale parameters of the component distributions,
                shape (batch_size, seq_len, num_mix_components)
            log_weights: Logarithms of mixing probabilities for the component distributions,
                shape (batch_size, seq_len, num_mix_components)
            # mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
            # std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        """
        self.locs = locs # (*,num_component)
        self.log_scales = log_scales
        self.log_weights = log_weights
        mixture_dist = Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        # if mean_log_inter_time == 0. and std_log_inter_time == 1.:
        #     transforms = []
        # else:
        #     transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        # self.mean_log_inter_time = mean_log_inter_time
        # self.std_log_inter_time = std_log_inter_time
        transforms = []
        transforms.append(D.ExpTransform())
        super().__init__(GMM, transforms)
    
    @property
    def mean(self):
        temp1 = torch.exp(self.locs + 0.5 * torch.exp(self.log_scales)) # (*, num_component)
        temp2 = temp1 * torch.exp(self.log_weights)
        # temp3 = temp2 * self.std_log_inter_time + self.mean_log_inter_time
        temp4 = temp2.sum(-1)# (*)
        assert not torch.any(temp4.isinf())
        return temp4

    
    # @property
    # def mean(self) -> torch.Tensor:
    #     """
    #     Compute the expected value of the distribution.
    #     Returns:
    #         mean: Expected value, shape (batch_size, seq_len)
    #     """
    #     a = 1
    #     b = 0
    #     loc = self.base_dist._component_distribution.loc
    #     variance = self.base_dist._component_distribution.variance
    #     log_weights = self.base_dist._mixture_distribution.logits
    #     temp = log_weights + a * loc + b + 0.5 * a**2 * variance
    #     temp_log_sum_exp = temp.logsumexp(-1)
    #     assert not torch.any(temp_log_sum_exp.exp().isinf())
    #     return temp_log_sum_exp.exp()
