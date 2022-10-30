# -*- coding: utf-8 -*-
# file: train_atepc_english.py
# time: 2021/6/8 0008
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC training script                                                  #
########################################################################################################################

from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional.dataset.dataset_manager import ABSADatasetList, DatasetItem
from pyabsa.functional import ATEPCConfigManager
from pyabsa import ATEPCCheckpointManager

config = ATEPCConfigManager.get_atepc_config_english()

config.model = ATEPCModelList.LCFS_ATEPC
config.evaluate_begin = 4
config.num_epoch = 5

dataset_path = DatasetItem("/home/shuyuan/shuyuan/PyABSA/integrated_datasets/atepc_datasets/100_1.CustomDataset")
checkpoint_path = ATEPCCheckpointManager.get_checkpoint(checkpoint='english')

aspect_extractor = Trainer(config=config,
                           dataset=dataset_path,
                           from_checkpoint=checkpoint_path,
                           checkpoint_save_mode=1,
                           auto_device=True
                           ).load_trained_model()


aspect_extractor.extract_aspect(
    ['the wine list is incredible and extensive and diverse , the food is all incredible and the staff was all very nice , ood at their jobs and cultured .',
     'the International Rice Research Institute (IRRI) based in the Philippines have been working on The tests for a long time and the International Rice Research Institute (IRRI) based in the Philippines would like to have The tests completed as soon as possible,', 
     'So the 75 percent falls far short of what a tobacco farmer needs. The Zimbabwe Tobacco Association (ZTA) are also of the view that tobacco is the biggest foreign currency earner yet The Zimbabwe Tobacco Association (ZTA) were awarded the lowest retention when other sectors have been given between 80 to 100 percent,']
)
