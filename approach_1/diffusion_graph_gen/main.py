import graph_tool as gt
import os
import pathlib
import warnings
import shutil
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import utils
from abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model_discrete import DiscreteDenoisingDiffusion
from extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    for node_size in range(5,10):
        path = 'data/planar'
        if os.path.exists(path):
            shutil.rmtree(path)
        dataset_config = cfg["dataset"]

        if dataset_config["name"] in ['planar']:
            from spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
            from spectre_utils import PlanarSamplingMetrics
            from visualization import NonMolecularVisualization

            datamodule = SpectreGraphDataModule(cfg,node_size)
            sampling_metrics = PlanarSamplingMetrics(datamodule)

            dataset_infos = SpectreDatasetInfos(datamodule)
            train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
            visualization_tools = NonMolecularVisualization()

            if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features)

            model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                            'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                            'extra_features': extra_features, 'domain_features': domain_features}



        utils.create_folders(cfg)

        if cfg.model.type == 'discrete':
            model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        else:
            model = ''

        callbacks = []
        if cfg.train.save_model:
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                                  filename='{epoch}',
                                                  monitor='val/epoch_NLL',
                                                  save_top_k=5,
                                                  mode='min',
                                                  every_n_epochs=1)
            last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
            callbacks.append(last_ckpt_save)
            callbacks.append(checkpoint_callback)

        # if cfg.train.ema_decay > 0:
        #     ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        #     callbacks.append(ema_callback)

        name = cfg.general.name
        # if name == 'debug':
        #     print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

        use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
        trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                          strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                          accelerator='gpu' if use_gpu else 'cpu',
                          devices=cfg.general.gpus if use_gpu else 1,
                          max_epochs=cfg.train.n_epochs,
                          check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                          fast_dev_run=cfg.general.name == 'debug',
                          enable_progress_bar=False,
                          callbacks=callbacks,
                          log_every_n_steps=50 if name != 'debug' else 1,
                          logger = [])

        if not cfg.general.test_only:
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
            if cfg.general.name not in ['debug', 'test']:
                #trainer.test(model, datamodule=datamodule)
                pass
        else:
            pass
            # # Start by evaluating test_only_path
            # trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
            # if cfg.general.evaluate_all_checkpoints:
            #     directory = pathlib.Path(cfg.general.test_only).parents[0]
            #     print("Directory:", directory)
            #     files_list = os.listdir(directory)
            #     for file in files_list:
            #         if '.ckpt' in file:
            #             ckpt_path = os.path.join(directory, file)
            #             if ckpt_path == cfg.general.test_only:
            #                 continue
            #             print("Loading checkpoint", ckpt_path)
            #             trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
