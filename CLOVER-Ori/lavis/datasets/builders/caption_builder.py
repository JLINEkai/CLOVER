"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder


from lavis.common.registry import registry



from lavis.datasets.datasets.pathvqa_dataset import PVQAdataset
from lavis.datasets.datasets.quilt_vqa_dataset import QuiltVQAdataset
from lavis.datasets.datasets.pmc_vqa_dataset import PMCVQAdataset
from lavis.datasets.datasets.sto_datasets import STOdataset

    
@registry.register_builder("pvqa")
class PVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PVQAdataset
    train_dataset_cls = PVQAdataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/pvqa/defaults_cap.yaml",
    }
    
@registry.register_builder("stovqa")
class STOBuilder(BaseDatasetBuilder):
    train_dataset_cls = STOdataset
    train_dataset_cls = STOdataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sto/defaults_cap.yaml",
    }
    
@registry.register_builder("quiltvqa")
class QuiltVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = QuiltVQAdataset
    train_dataset_cls = QuiltVQAdataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/quiltvqa/defaults_cap.yaml",
    }    
@registry.register_builder("pmcvqa")
class PMCVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PMCVQAdataset
    train_dataset_cls = PMCVQAdataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/pmcvqa/defaults_cap.yaml",
    }   


