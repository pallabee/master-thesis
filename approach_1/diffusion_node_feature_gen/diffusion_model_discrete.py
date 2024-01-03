import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb

from mlp_model import MLP
from noise_schedule import PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition, DiscreteUniformTransition
import diffusion_utils
from train_metrics import TrainLossDiscrete
from abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
import utils
import os

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Fdim = input_dims['Feat']
        self.Fdim_output = output_dims['Feat']

        self.dataset_info = dataset_infos
        self.train_loss = TrainLossDiscrete()

        self.val_nll = NLL()
        self.val_F_kl = SumExceptBatchKL()
        self.val_F_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_F_kl = SumExceptBatchKL()
        self.test_F_logp = SumExceptBatchMetric()

        self.model = MLP(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(f_classes=self.Fdim_output)
            f_limit = torch.ones(self.Fdim_output) / self.Fdim_output
            #f_limit = torch.rand(self.Fdim_output).to("cuda:0")

            self.limit_dist = utils.PlaceHolder(Feat=f_limit)
        elif cfg.model.transition == 'marginal':
            features = self.dataset_info.feature_types.float()
            f_marginals = features / torch.sum(features)
            #f_limit= torch.rand(self.Fdim_output).to("cuda:0")

            print(
                f"Marginal distribution of the classes: {f_marginals} for features")
            self.transition_model = MarginalUniformTransition( f_marginals=f_marginals)
            self.limit_dist = utils.PlaceHolder(Feat=f_marginals)

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
            #self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def training_step(self, data, i):
        dense = utils.to_dense(data.feature)
        noisy_data = self.apply_noise(dense.Feat.float())

        pred = self.forward(noisy_data)
        loss = self.train_loss(masked_pred_Feat= pred.Feat,
                               true_Feat = dense.Feat.float(),
                               log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Fdim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        #self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}")


    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_F_kl.reset()
        self.val_F_logp.reset()
       # self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense = utils.to_dense(data.feature)
        noisy_data = self.apply_noise(dense.Feat.float())

        pred = self.forward(noisy_data)
        nll = self.compute_val_loss(pred, noisy_data, dense.Feat.float(), test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(),
                   self.val_F_kl.compute() * self.T,
                   self.val_F_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/F_kl": metrics[1],
                       "val/F_logp": metrics[2]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- val/F_kl {metrics[1] :.2f} -- ",
                   f"Val F_logp: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        # if val_nll < self.best_val_nll:
        #     self.best_val_nll = val_nll
        # self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

    def kl_prior(self, Feat):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((Feat.size(0), 1), device=Feat.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probFeat = Feat @ Qtb.Feat
        bs, n, _ = probFeat.shape

        limit_Feat = self.limit_dist.Feat[None, None, :].expand(bs, n, -1).type_as(probFeat)

        kl_distance_Feat = F.kl_div(input=probFeat.log(), target=limit_Feat, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_Feat)

    def compute_Lt(self, Feat, pred, noisy_data, test):

        pred_probs_Feat = F.softmax(pred.Feat, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        #bs, n, d = Feat.shape
        prob_true = diffusion_utils.posterior_distributions(Feat=Feat,
                                                            F_t=noisy_data['F_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)

        prob_pred = diffusion_utils.posterior_distributions(Feat=pred_probs_Feat,

                                                            F_t=noisy_data['F_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)


        kl_f = (self.test_F_kl if test else self.val_F_kl)(prob_true.Feat, torch.log(prob_pred.Feat))
        return self.T *  kl_f

    def reconstruction_logp(self,t,Feat):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probFeat0 = Feat @ Q0.Feat # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probF=probFeat0)

        Feat0 = torch.FloatTensor(Feat.shape).to("cuda:0")
        for i in range(0, Feat0.shape[0]):
            for j in range(0, Feat0.shape[1]):
                Feat0[i][j] = Feat[i][j][sampled0.Feat[i][j]]
        assert (Feat.shape == Feat0.shape)

        sampled_0 = utils.PlaceHolder(Feat=Feat0)

        # Predictions
        noisy_data = { 'F_t': sampled_0.Feat,
                      't': torch.zeros(Feat0.shape[0], 1)}
        pred0 = self.forward(noisy_data)

        # Normalize predictions
        probFeat0 = F.softmax(pred0.Feat, dim=-1)

        return utils.PlaceHolder(Feat = probFeat0)

    def apply_noise(self, Feat):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(Feat.size(0), 1), device=Feat.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.Feat.sum(dim=2) - 1.) < 1e-4).all(), Qtb.Feat.sum(dim=2) - 1

        # Compute transition probabilities
        probF = Feat @ Qtb.Feat

        sampled_t = diffusion_utils.sample_discrete_features(probF=probF)

        F_t = torch.FloatTensor(Feat.shape).to("cuda:0")
        for i in range(0, F_t.shape[0]):
            for j in range(0, F_t.shape[1]):
                F_t[i][j] = Feat[i][j][sampled_t.Feat[i][j]]


        dir_path = os.path.dirname(os.path.realpath(__file__))
        print('sampled features',+ F_t)
        torch.save(F_t,dir_path + '/sampled_features/' + self.cfg.dataset.name+'/features.pt')
        #-----------------------For saving more samples of PubMed Disease class--------------------------------------
        # F_out = []
        # for i in range(0,20):
        #     sample_pubmed = diffusion_utils.sample_discrete_features(probF=probF)
        #     F_pubmed = torch.FloatTensor(Feat.shape).to("cuda:0")
        #     for i in range(0, F_pubmed.shape[0]):
        #         for j in range(0, F_pubmed.shape[1]):
        #             F_pubmed[i][j] = Feat[i][j][sample_pubmed.Feat[i][j]]
        #             F_out.append(F_pubmed[i][j].cpu().detach().numpy().tolist())
        #
        # #print('sampled features', + torch.stack(F_out))
        #
        # torch.save(torch.tensor(F_out), dir_path + '/sampled_features/' + self.cfg.dataset.name + '/features.pt')
        # -----------------------For saving more samples of PubMed Disease class---------------------------------------

        assert (Feat.shape == F_t.shape)

        z_t = utils.PlaceHolder(Feat=F_t)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'F_t': z_t.Feat}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, Feat, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        #kl_prior = self.kl_prior(Feat)

        # 3. Diffusion loss
        #loss_all_t = self.compute_Lt(Feat, pred, noisy_data,test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (Feat | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t,Feat)

        #Feature loss added
        feature_loss = self.val_F_logp(Feat * prob0.Feat.log())
        print('feature_loss',feature_loss)

        #Combine terms
        # nlls = kl_prior + loss_all_t - feature_loss
        # assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        #nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({
                # "kl prior": kl_prior.mean(),
                #       "Estimator loss terms": loss_all_t.mean(),
                       "feature_loss": feature_loss})
                       #'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return feature_loss

    def forward(self, noisy_data):

        return self.model(noisy_data['F_t'])



