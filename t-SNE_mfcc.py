import numpy as np
from torch.utils.data import DataLoader
from src.feature_extraction import *
from src.dataset import *
from config import cfg, update_config
import argparse
from loguru import logger

parser = argparse.ArgumentParser(description='Running audio classification')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

args = parser.parse_args()
update_config(cfg, args)

logger.info("Running model_inspection ...")
logger.info(f'MODEL path: {cfg.PATH.MODEL_PATH}')

mfcc_const = MFCC_params(cfg.FEATUREPARAMS.SAMPLING_RATE, cfg.FEATUREPARAMS.NUM_CEPSTRAL_COEFFICIENTS,
                         cfg.FEATUREPARAMS.HOP_LENGTH, cfg.FEATUREPARAMS.LEN_WINDOW)

if __name__ == '__main__':

    # data load
    data_raw = np.load(cfg.PATH.TEST_PATH)
    data_audio = data_raw[:, 0:-1]
    data_label = data_raw[:, -1]

    logger.info("data in - before MFCC")
    data_featureVector = audioProcessing(data_audio, mfcc_const)
    logger.info("data in - after MFCC")

    rawDataset = AudioDataset(data_featureVector, data_label, input_size=mfcc_const.n_mfcc)
    data_loader = DataLoader(rawDataset, batch_size=1, shuffle=False)
    
    # plot_tsne(features, data_label, class_names=cfg.HYPERPARAMS.LABEL_CLASS)
    features = np.mean(data_featureVector, axis=1)  # shape: (num_samples, n_mfcc)

    plot_tsne_3d(features, data_label, class_names=cfg.HYPERPARAMS.LABEL_CLASS)
    # plot_tsne_interactive(features, labels, indices, class_names=cfg.HYPERPARAMS.LABEL_CLASS)