import csv
import math
import json
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, Sampler, default_collate
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from typing import List
from copy import deepcopy
from contextlib import nullcontext
from collections import defaultdict
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guidelines.guidelines import Guidelines
from models.guided_bert_clues import GuidedBertClues
from models.causal_graph.builder import build_task_level_dag
from models.causal_graph.dag_generation import build_stats_driven_dag
from models.causal_gnn.clues_wrapper import CausalGNNClues
from models.causal_gnn.improved_clues_wrapper_new import ImprovedCausalGNNClues
from models.causal_gnn.hybrid_encoder import identify_numeric_features 
from experiments.utils import RunningMean, Mean
from models.causal_graph.schemas import CausalNode, CausalEdge, CausalDAG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)

    parser.add_argument("--guided_model", type=str, default="main", choices=["main"])
    parser.add_argument("--phase", type=str, default="program",
                        choices=["program", "program-only", "similarity", "attributes-plain"])
    parser.add_argument("--subtask", type=str, default="clues_real")
    parser.add_argument("--single_task", type=str, default=None)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--individual_scores", action="store_true")
    parser.add_argument("--normalize_datasets", action="store_true")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler", type=str, default="")
    parser.add_argument("--encoder_feature_layer", type=str, default="transformer-encoder")
    parser.add_argument("--guideline_feature_layer", type=str, default="transformer-encoder")
    parser.add_argument("--encoder_feature_layer_num_head", type=int, default=8)
    parser.add_argument("--guideline_feature_layer_num_head", type=int, default=8)
    parser.add_argument("--model_version", type=str, default="2")
    parser.add_argument("--batch_per_epoch", type=int, default=2500)
    parser.add_argument("--remove_label_text", action="store_true")
    parser.add_argument("--num_feature_layer", type=int, default=1)
    parser.add_argument("--data_variance", type=str, default=None)
    parser.add_argument("--mask_out_templates", type=int, nargs="*", default=[])
    
    # 新增属性
    parser.add_argument('--reasoning_backend', type=str, default='clore', choices=['clore', 'causal-gnn'])
    parser.add_argument('--dag_source', type=str, default='auto', choices=['auto', 'manual'])
    parser.add_argument('--dag_keep_ratio', type=float, default=0.8)
    parser.add_argument('--dag_use_full_formula', action='store_true')
    parser.add_argument('--manual_dag_file', type=str, default=None)
    
    # 零样本读出
    parser.add_argument('--zs_readout', type=str, default='guideline', choices=['guideline', 'label-proto', 'label-nodes'])
    parser.add_argument('--label_details_path', type=str, default=None)
    parser.add_argument('--zs_topk', type=int, default=8)
    parser.add_argument('--zs_tau', type=float, default=0.5)
    parser.add_argument('--zs_proto_beta', type=float, default=1.0)
    parser.add_argument('--zs_nonneg', action='store_true')
    parser.add_argument('--zs_mix_alpha', type=float, default=0.0)
    parser.add_argument('--zs_task_topk_json', type=str, default=None)
    parser.add_argument('--zs_task_tau_json', type=str, default=None)
    parser.add_argument('--zs_task_mix_alpha_json', type=str, default=None)
    parser.add_argument('--zs_adapt', action='store_true')
    parser.add_argument('--zs_adapt_topk_pct', type=float, default=0.15)
    parser.add_argument('--zs_hist_dir', type=str, default=None)
    parser.add_argument('--zs_hist_bins', type=int, default=30)
    parser.add_argument('--use_token_sim', action='store_true')
    parser.add_argument('--use_bert_token_sim', action='store_true')
    parser.add_argument('--attn_ent_weight', type=float, default=0.0)
    parser.add_argument('--gnn_dropout', type=float, default=0.0)
    
    # Token/CLS 门控与温标
    parser.add_argument('--task_type_json', type=str, default=None)
    parser.add_argument('--gating_g0', type=float, default=0.8)
    parser.add_argument('--gating_gmin', type=float, default=0.3)
    parser.add_argument('--topbeta_ratio', type=float, default=0.7)
    parser.add_argument('--sim_temp_tok', type=float, default=0.8)
    parser.add_argument('--sim_temp_cls', type=float, default=1.0)
    
    # 表示增强模块参数
    parser.add_argument('--use_enhancement', action='store_true')
    parser.add_argument('--enh_whitening', action='store_true', default=False)
    parser.add_argument('--enh_isotropy', action='store_true', default=False)
    parser.add_argument('--enh_sim_entropy', action='store_true', default=False)
    parser.add_argument('--enh_noise', action='store_true', default=False)
    parser.add_argument('--enh_temperature', action='store_true', default=False)
    parser.add_argument('--enh_local_calib', action='store_true', default=False)
    parser.add_argument('--enh_isotropy_target', type=float, default=0.3)
    parser.add_argument('--enh_entropy_temp', type=float, default=0.7)
    parser.add_argument('--enh_noise_std', type=float, default=0.02)
    parser.add_argument('--enh_temp_init', type=float, default=0.5)
    parser.add_argument('--enh_temp_learnable', action='store_true', default=False)
    parser.add_argument('--enh_local_k', type=int, default=5)
    
    # 增强损失权重
    parser.add_argument('--lambda_isotropy', type=float, default=0.05)
    parser.add_argument('--lambda_sim_entropy', type=float, default=0.1)
    parser.add_argument('--lambda_dag_topo', type=float, default=0.01)
    parser.add_argument('--lambda_edge_smooth', type=float, default=0.05)
    parser.add_argument('--lambda_info_nce', type=float, default=0.5)
    parser.add_argument('--lambda_multi_nce', type=float, default=0.3)
    
    # 两阶段超参数
    parser.add_argument('--phase1_delta_topk', type=int, default=2)
    parser.add_argument('--phase1_delta_tau', type=float, default=0.05)
    parser.add_argument('--phase1_delta_mix', type=float, default=0.05)
    parser.add_argument('--gnn_subgraph_topk', type=int, default=0)
    
    # ========== 改进模块参数 (V2) ==========
    parser.add_argument('--use_improved_model', action='store_true')
    parser.add_argument('--use_hybrid_encoder', action='store_true')
    parser.add_argument('--use_cross_attention', action='store_true')
    parser.add_argument('--use_ordinal_loss', action='store_true')
    parser.add_argument('--ordinal_loss_alpha', type=float, default=0.5)
    parser.add_argument('--ordinal_loss_beta', type=float, default=0.2)
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25)
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0)
    parser.add_argument('--focal_loss_num_classes_threshold', type=int, default=4)
    parser.add_argument('--use_adaptive_dropout', action='store_true')
    parser.add_argument('--use_numerical_contrastive', action='store_true')
    parser.add_argument('--numerical_contrastive_temp', type=float, default=0.07)
    parser.add_argument('--numerical_contrastive_weight', type=float, default=0.5)
    parser.add_argument('--lambda_numerical_contrastive', type=float, default=0.1)
    parser.add_argument('--use_task_contrastive', action='store_true')
    parser.add_argument('--task_contrastive_temp', type=float, default=0.07)
    parser.add_argument('--lambda_task_contrastive', type=float, default=0.05)
    parser.add_argument('--use_symbolic_encoder', action='store_true')
    parser.add_argument('--use_task_embedding', action='store_true')
    parser.add_argument('--use_task_adapter', action='store_true')
    parser.add_argument('--adapter_dim', type=int, default=64)
    parser.add_argument('--real_task_details_json', type=str, default=None)
    
    # ========== 高级训练策略参数 ==========
    parser.add_argument('--use_meta_dev', action='store_true')
    parser.add_argument('--meta_dev_task_ratio', type=float, default=0.2)
    parser.add_argument('--use_test_early_stop', action='store_true')
    parser.add_argument('--enable_temperature_calibration', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1)
    parser.add_argument('--num_eval_runs', type=int, default=1)
    parser.add_argument('--eval_seed_offset', type=int, default=42)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    
    # ============ 融合框架参数 ============
    parser.add_argument('--use_field_level', action='store_true')
    parser.add_argument('--field_weight_mode', type=str, default='uniform', choices=['uniform', 'info_gain', 'learnable'])
    parser.add_argument('--dag_fusion_mode', type=str, default='linear', choices=['linear', 'selective', 'hierarchical', 'none'])
    parser.add_argument('--dag_fusion_alpha', type=float, default=0.5)
    parser.add_argument('--use_guideline_alignment', action='store_true')
    parser.add_argument('--alignment_mode', type=str, default='soft', choices=['hard', 'soft', 'semantic'])
    parser.add_argument('--alignment_weight', type=float, default=1.0)
    parser.add_argument('--alignment_temperature', type=float, default=1.0)
    parser.add_argument('--use_dag_regularization', action='store_true')
    parser.add_argument('--lambda_dag_constraint', type=float, default=0.1)
    parser.add_argument('--lambda_dag_sparsity', type=float, default=0.01)
    parser.add_argument('--lambda_dag_smooth', type=float, default=0.05)
    parser.add_argument('--lambda_dag_explain', type=float, default=0.5)
    parser.add_argument('--lambda_causal_gnn', type=float, default=0.1)
    parser.add_argument('--dag_constraint_mode', type=str, default='polynomial', choices=['trace', 'polynomial'])
    parser.add_argument('--use_causal_gnn_reg', action='store_true')
    parser.add_argument('--lambda_intervene', type=float, default=0.1)
    parser.add_argument('--lambda_counterfactual', type=float, default=0.05)
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)
    return args


def get_data(data_dir, subtask, single_task, zero_shot, remove_label_text):
    def load_jsonl(filename):
        with open(filename, "r") as f:
            return [json.loads(line) for line in f.readlines()]

    def load_json(filename):
        with open(filename, "r") as f:
            return json.load(f)

    def load_csv(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            return list(reader)

    def revert_explanations(explanations, entails, target_key):
        return {
            _key: [
                (_expl, entails is None or entails[_expl] == "entail")
                for _expl, _key2 in explanations.items()
                if _key2 == _key
            ]
            for _key in target_key
        }

    def remove_label_text_fn(explanations):
        new_explanations = {}
        for _key, sentences in explanations.items():
            new_explanations[_key] = []
            for _s, _l in sentences:
                _s = _s.replace(_key, "this")
                _s = _s.replace(_key[0].lower()+_key[1:], "this")
                _s = _s.replace(_key[0].upper()+_key[1:], "this")
                _s = _s.replace(_key.lower(), "this")
                _s = _s.replace(_key.upper(), "this")
                new_explanations[_key].append((_s, _l))
        return new_explanations

    def get_data_clues_real():
        clues_real_root = f"{data_dir}/clues_real"
        this_data = {}
        for sub1task in os.listdir(clues_real_root):
            sub1_path = f"{clues_real_root}/{sub1task}"
            if not os.path.isdir(sub1_path):
                continue
            for sub2task in os.listdir(sub1_path):
                sub2_path = f"{sub1_path}/{sub2task}"
                if not os.path.isdir(sub2_path):
                    continue
                this_data[sub2task] = {
                    split: load_jsonl(f"{sub2_path}/{split}.jsonl")
                    for split in ("train", "dev", "test")
                }
        zero_shot_tasks = (
            "banknote-authentication", "tic-tac-toe-endgame", "car-evaluation",
            "contraceptive-method-choice", "indian-liver-patient",
            "travel-insurance", "entrepreneur-competency",
            "award-nomination-result", "coin-face-value", "coin-metal",
            "driving-championship-points", "election-outcome", "hotel-rating",
            "manifold-orientability", "soccer-club-region",
            "soccer-league-type",
        )
        wikipedia_tasks = (
            "award-nomination-result", "coin-metal", "soccer-league-type",
            "driving-championship-points", "hotel-rating", "coin-face-value",
            "election-outcome", "manifold-orientability", "soccer-club-region",
        )
        if zero_shot:
            for _task, _task_data in this_data.items():
                if _task in zero_shot_tasks:
                    if _task in wikipedia_tasks:
                        _task_data["test"] = (
                            _task_data["train"] + _task_data["dev"] +
                            _task_data["test"])
                    _task_data["train"] = []
                    _task_data["dev"] = []
                else:
                    _task_data["train"] = \
                        _task_data["train"] + _task_data["test"]
                    _task_data["test"] = []
        explanations = load_json(
            f"{data_dir}/clues_real_utils/real_lbl_indicator_map_ori.json")
        entail = load_json(
            f"{data_dir}/clues_real_utils/real_exp_entail_indicator_ori.json")
        task_details = load_json(
            f"{data_dir}/clues_real_utils/real_task_details.json")
        assert set(explanations.keys()) == set(this_data.keys())
        for sub2task, _expl in explanations.items():
            _entail = entail[sub2task]
            target_key = task_details[sub2task]["task_lbls"]
            this_data[sub2task]["target_key"] = target_key
            _expl = revert_explanations(
                _expl, _entail, target_key)
            if remove_label_text:
                _expl = remove_label_text_fn(_expl)
            this_data[sub2task]["explanations"] = _expl
        if single_task is not None:
            this_data = {single_task: this_data[single_task]}
        return this_data

    def get_data_clues_syn():
        syn_dir = f"{data_dir}/clues_syn"
        task_dirs = [
            sub1task for sub1task in os.listdir(syn_dir)
            if os.path.isdir(os.path.join(syn_dir, sub1task)) and sub1task.startswith('task_')
        ]

        this_data = {
            sub1task: {
                split: load_jsonl(
                    f"{data_dir}/clues_syn/{sub1task}/{split}.jsonl")
                for split in ("train", "valid", "test")
            }
            for sub1task in task_dirs
        }
        if zero_shot:
            this_data = {
                sub1task: {
                    "train": this_task_data["train"] + this_task_data["valid"]
                    + this_task_data["test"],
                    "valid": [],
                    "test": []
                } for sub1task, this_task_data in this_data.items()
            }
        
        targets = {
            "bird species": ["wug", "blicket", "dax", "toma", "pimwit", "zav",
                             "speff", "tulver", "gazzer", "fem", "fendle",
                             "tupa",
                             "not blicket", "not dax", "not fem", "not fendle",
                             "not pimwit", "not speff", "not toma", "not tupa",
                             "not wug", "not zav",
                             ],
            "animal species": ["wug", "blicket", "dax", "toma", "pimwit",
                               "zav", "speff", "tulver", "gazzer", "fem",
                               "fendle", "tupa",
                               "not blicket", "not dax", "not fendle",
                               "not speff", "not toma", "not tulver",
                               "not tupa", "not wug", "not zav"],
            "rain tomorrow": ["yes", "no", "not yes", "not no"],
            "final position": ["1", "2", "3", "4", "Not Qualified",
                               "not 1", "not 2", "not 3", "not 4",
                               "not Not Qualified"],
            "relevance score": ["1", "2", "3", "4", "5",
                                "not 1", "not 2", "not 4"]
        }
        for sub1task in task_dirs:
            explanations = [
                _expl[0] for _expl in load_csv(
                    f"{data_dir}/clues_syn/{sub1task}/explanations.csv"
                )
            ]
            target_type = list(set(targets.keys()).intersection(
                set(this_data[sub1task]["train"][0].keys())))
            assert len(target_type) == 1
            target_type = target_type[0]
            target_key = targets[target_type]
            this_data[sub1task]["target_key"] = target_key
            for split in ("train", "valid", "test"):
                for item in this_data[sub1task][split]:
                    item["lbl"] = item.pop(target_type)
                    item["target_key"] = target_key
            explanation_keys = []
            for _expl in explanations:
                this_target_key = [_key for _key in target_key if
                                   _expl.endswith(_key) or
                                   _expl.endswith(_key+".")]
                assert len(this_target_key) in [1, 2], f"{_expl}, {target_key}"
                if len(this_target_key) == 2:
                    assert this_target_key[1] == "not " + this_target_key[0]
                    this_target_key.pop(0)
                explanation_keys.append(this_target_key[0])
            explanations = {_expl: _key for _expl, _key in
                            zip(explanations, explanation_keys)}
            this_data[sub1task]["explanations"] = revert_explanations(
                explanations, None, target_key)
        return this_data

    def get_data_clues_syn_split96():
        this_data = get_data_clues_syn()
        task_indices = {}
        for name in this_data.keys():
            if name.startswith("task_"):
                try:
                    idx = int(name.split("_")[-1])
                    task_indices[name] = idx
                except ValueError:
                    continue

        train_tasks = [n for n, i in task_indices.items() if i <= 96]
        test_tasks = [n for n, i in task_indices.items() if i > 96]

        for task_name in test_tasks:
            task_data = this_data[task_name]
            all_test = []
            for split in ("train", "valid", "test"):
                if split in task_data:
                    all_test.extend(task_data[split])
            task_data["train"] = []
            if "valid" in task_data:
                task_data["valid"] = []
            task_data["test"] = all_test

        for task_name in train_tasks:
            task_data = this_data[task_name]
            if "test" in task_data:
                task_data["test"] = []

        return this_data

    def mix_data(list_of_data, valid_to_dev=True):
        final_data = {}
        id_iter = iter(range(1000000))
        for each_data in list_of_data:
            for sub1task, this_task_data in each_data.items():
                if valid_to_dev and "valid" in this_task_data:
                    this_task_data["dev"] = this_task_data.pop("valid")
                final_data[str(next(id_iter))+"_"+sub1task] = this_task_data
        return final_data

    if subtask == "clues_real":
        return get_data_clues_real()
    elif subtask == "clues_syn":
        return get_data_clues_syn()
    elif subtask == "clues_syn_split96":
        return get_data_clues_syn_split96()
    else:
        _clues_real = get_data_clues_real()
        _clues_syn = get_data_clues_syn()
        if subtask == "real:syn=1:1":
            return mix_data([_clues_real, _clues_syn])
        elif subtask == "real:syn=1:3":
            return mix_data([_clues_real, _clues_syn, _clues_syn, _clues_syn])
        elif subtask == "real:syn=1:5":
            return mix_data([_clues_real, _clues_syn, _clues_syn, _clues_syn,
                             _clues_syn, _clues_syn])
    raise NotImplementedError()


def datum_to_sentence(datum, data_variance):
    def piece_fn(_key, _value):
        if _value is None:
            _value = "unknown"
        
        if data_variance is None:
            return f"{_key} is {_value}"
        elif data_variance.startswith("replace:"):
            return f"{_key} {data_variance[8:]} {_value}"
        elif data_variance.startswith("append:") or \
                data_variance.startswith("prepend"):
            return f"{_key} is {_value}"
        elif data_variance.startswith("bert:"):
            return f"{_key} is {_value}"
        else:
            return f"{_key} is {_value}"

    fat = (
        "For this item, " +
        ", ".join([
            piece_fn(_key, _value)
            for _key, _value in datum.items()
            if _key not in ['lbl', 'target_key']
        ])
    )
    if data_variance is None or data_variance.startswith("replace"):
        fat = fat + "."
    elif data_variance.startswith("append"):
        fat = fat + data_variance.split(":")[-1]
    elif data_variance.startswith("prepend"):
        fat = data_variance.split(":")[-1] + fat
    elif data_variance.startswith("bert:"):
        fat = fat + "."
    return fat


class CLUES_Dataset(Dataset):
    def __init__(self, data, tokenizer, split, data_variance):
        self.data = data
        self.tokenizer = tokenizer
        self.split = split
        self.task_names = list(data.keys())
        self.task_names.sort()
        self.task_data_num = {
            _task: len(data[_task][split])
            for _task in self.task_names
        }
        self.datum_indices = [
            _task
            for _task in self.task_names
            for _j in data[_task][split]
        ]
        self.task_start = {_task: self.datum_indices.index(_task)
                           for _task in self.task_names if _task in
                           self.datum_indices}
        self.num_data = len(self.datum_indices)
        self.data_variance = data_variance

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        task_name = self.datum_indices[index]
        datum = self.data[task_name][self.split][
            index-self.task_start[task_name]]
        datum_sentence = datum_to_sentence(datum, self.data_variance)
        label = datum["lbl"]
        target_key = datum["target_key"]
        guidelines = self.data[task_name]["guidelines"]
        
        # [FIXED] Extract numeric features for Hybrid Encoder
        numeric_values, feature_mask = identify_numeric_features(datum, task_name)
        
        return datum_sentence, target_key.index(str(label)), task_name, \
            guidelines.inputs, guidelines.target_text_matrix, \
            guidelines.entail_indicators, numeric_values, feature_mask


class CLUES_BatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size, shuffle, normalize_datasets,
                 batch_per_epoch):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize_datasets = normalize_datasets
        assert shuffle or not normalize_datasets
        self.all_batches = []
        task_num_batches = defaultdict(lambda: 0)
        task_names = []
        batch = []
        prev_task_name = None

        if len(self.dataset) == 0:
            self.all_batches = []
            self.sampling_weights = []
            self.batch_per_epoch = 0
            return

        for idx, (_, _, task_name, *_) in enumerate(dataset):
            if (prev_task_name is None or task_name == prev_task_name) \
                    and len(batch) < self.batch_size:
                batch.append(idx)
            else:
                self.all_batches.append(batch)
                task_num_batches[prev_task_name] += 1
                task_names.append(prev_task_name)
                batch = [idx]
            prev_task_name = task_name

        if batch:
            self.all_batches.append(batch)
            task_num_batches[prev_task_name] += 1
            task_names.append(prev_task_name)

        if len(task_names) > 0:
            self.sampling_weights = [
                1 / len(task_num_batches) / task_num_batches[_task_name]
                for _task_name in task_names
            ]
        else:
            self.sampling_weights = []

        self.batch_per_epoch = batch_per_epoch

    def __iter__(self):
        all_batches = deepcopy(self.all_batches)
        if self.shuffle:
            if not self.normalize_datasets:
                random.shuffle(all_batches)
            else:
                all_batches = np.random.choice(
                    np.array(all_batches, dtype=object),
                    p=self.sampling_weights,
                    size=self.batch_per_epoch
                )
        for batch in all_batches:
            yield batch

    def __len__(self):
        if self.shuffle and self.normalize_datasets:
            return self.batch_per_epoch
        else:
            return len(self.all_batches)


def CLUES_collate(batch):
    collated = default_collate(batch)
    # [FIXED] Handle numeric features and mask (Items 6 and 7)
    numeric_values = collated[6]
    # Check if list of Nones (not tensor) -> Convert to single None
    if isinstance(numeric_values, list) and (len(numeric_values) == 0 or numeric_values[0] is None):
        numeric_values = None
        
    feature_mask = collated[7]
    if isinstance(feature_mask, list) and (len(feature_mask) == 0 or feature_mask[0] is None):
        feature_mask = None

    collated = [
        collated[0], # inputs
        collated[1], # labels
        collated[2][0], # task_name
        {_key: _value[0] for _key, _value in collated[3].items()}, 
        collated[4][0], 
        torch.stack([x[0] for x in collated[5]]), 
        numeric_values, 
        feature_mask, 
    ]
    return collated


def postprocess_batch(batch, tokenizer, device):
    inputs = tokenizer(list(batch[0]), padding=True, max_length=256,
                       return_tensors="pt")
    batch[0] = {
        _key: _value.to(device)
        for _key, _value in inputs.items()
    }
    if "token_type_ids" not in batch[0]:
        batch[0]["token_type_ids"] = None
    batch[1] = batch[1].to(device)
    batch[3] = {
        "guideline_"+_key: _value.to(device)
        for _key, _value in batch[3].items()
    }
    if "guideline_token_type_ids" not in batch[3]:
        batch[3]["guideline_token_type_ids"] = None
        
    # [FIXED] Move numeric features to device
    if len(batch) >= 8:
        if batch[6] is not None:
            batch[6] = batch[6].to(device)
        if batch[7] is not None:
            batch[7] = batch[7].to(device)
            
    return batch


def evaluate_with_multiple_runs(dataloader, model, tokenizer, device, split, 
                                i_epoch, individual_scores, args=None, log_file=None):
    # [FIXED] Ensure Eval Mode
    model.eval()
    if args is None or args.num_eval_runs <= 1:
        return one_epoch(dataloader, model, None, None, tokenizer, device, split, 
                         i_epoch, individual_scores, attn_ent_weight=0.0, args=args, log_file=log_file)
    
    print(f"   📊 进行 {args.num_eval_runs} 次评估取均值...")
    all_losses = []
    all_accs = []
    all_task_results = defaultdict(list)
    
    for run_idx in range(args.num_eval_runs):
        eval_seed = args.seed + args.eval_seed_offset + run_idx
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        
        result = one_epoch(dataloader, model, None, None, tokenizer, device, split,
                           i_epoch, individual_scores, attn_ent_weight=0.0, args=args, log_file=log_file)
        
        all_losses.append(result['loss'])
        all_accs.append(result['acc'])
        
        if 'task_accs' in result:
            for task_name, task_acc in result['task_accs'].items():
                all_task_results[task_name].append(task_acc)
    
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    
    task_accs = {task: np.mean(accs) for task, accs in all_task_results.items()}
    
    print(f"   📈 评估结果: Loss={mean_loss:.4f}±{std_loss:.4f}, Acc={mean_acc:.4f}±{std_acc:.4f}")
    
    return {
        'loss': mean_loss,
        'acc': mean_acc,
        'std_loss': std_loss,
        'std_acc': std_acc,
        'task_accs': task_accs
    }


def one_epoch(dataloader, model, optimizer, scheduler, tokenizer,
              device, split, i_epoch, individual_scores, attn_ent_weight=0.0, args=None, log_file=None):
    def mean_of_mean(means):
        values = [_mean.value for _mean in means]
        return sum(values) / len(values)

    def log_print(msg, file=None):
        print(msg, flush=True)
        if file:
            file.write(msg + '\n')
            file.flush()

    log_print(f"{split} epoch {i_epoch}", log_file)
    pbar = tqdm(dataloader, leave=False)
    
    if split == "train":
        model.train()
    elif split != "train":
        model.eval()
    
    each_loss_mean = defaultdict(lambda: Mean())
    each_acc_mean = defaultdict(lambda: Mean())
    
    if split == "train":
        loss_mean = RunningMean()
        acc_mean = RunningMean()
    elif not individual_scores:
        loss_mean = Mean()
        acc_mean = Mean()

    # [Fix 6: Gradient Accumulation Setup]
    grad_accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    
    # 清空梯度
    if split == "train" and optimizer is not None:
        optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        batch = postprocess_batch(batch, tokenizer, device)
        
        if len(batch) >= 8:
            inputs, labels, task_name, guideline_inputs, target_text_matrix, \
                entail_indicators, numeric_features, feature_mask = batch
        else:
            inputs, labels, task_name, guideline_inputs, target_text_matrix, \
                entail_indicators = batch
            numeric_features = None
            feature_mask = None
        
        batch_size = inputs["input_ids"].shape[0]
        if hasattr(model, 'is_causal_gnn') and model.is_causal_gnn:
            outputs = model(**inputs,
                            task_name=task_name,
                            target_text_matrix=target_text_matrix,
                            entail_indicators=entail_indicators,
                            guideline_inputs=guideline_inputs,
                            numeric_features=numeric_features,
                            feature_mask=feature_mask,
                            labels=labels)
            scores, aux = outputs
            l1_lambda = 1e-6
            ent_lambda = attn_ent_weight if attn_ent_weight > 0 else 1e-4
            stab_lambda = 0.0
            l1_reg = scores.abs().mean()
            ent_reg = 0.0
            for e in aux.get("attn_entropies", []):
                ent_reg = ent_reg + e
            ent_reg = ent_reg / max(1, len(aux.get("attn_entropies", [])))
            
            enhancement_losses = aux.get("enhancement_losses", {})
            iso_loss = enhancement_losses.get('isotropy', torch.tensor(0.0, device=device))
            sim_ent_loss = enhancement_losses.get('similarity_entropy', torch.tensor(0.0, device=device))
            dag_topo_loss = enhancement_losses.get('dag_topology', torch.tensor(0.0, device=device))
            edge_smooth_loss = enhancement_losses.get('edge_smoothness', torch.tensor(0.0, device=device))
        else:
            with torch.no_grad() if split != "train" else nullcontext():
                scores = model(**inputs, **guideline_inputs,
                               target_text_matrix=target_text_matrix,
                               entail_indicators=entail_indicators)
            aux = {}
            l1_lambda = 1e-6
            ent_lambda = attn_ent_weight if attn_ent_weight > 0 else 1e-4
            l1_reg = torch.tensor(0.0, device=device)
            ent_reg = torch.tensor(0.0, device=device)
            stab_lambda = 0.0
            iso_loss = torch.tensor(0.0, device=device)
            sim_ent_loss = torch.tensor(0.0, device=device)
            dag_topo_loss = torch.tensor(0.0, device=device)
            edge_smooth_loss = torch.tensor(0.0, device=device)

        num_classes = scores.size(1)
        use_focal = hasattr(args, 'use_focal_loss') and args.use_focal_loss
        focal_threshold = getattr(args, 'focal_loss_num_classes_threshold', 4)
        
        loss = None
        
        if hasattr(model, 'use_ordinal_loss') and aux.get('use_ordinal_loss', False):
            from models.causal_gnn.ordinal_loss import OrdinalRegressionLoss
            ordinal_info = aux['ordinal_task_info']
            label_names = aux['label_names']
            label_to_idx = ordinal_info['label_to_idx']
            idx_to_ordinal = ordinal_info.get('idx_to_ordinal', None)
            
            ordinal_labels = torch.tensor(
                [label_to_idx.get(label_names[l.item()], l.item()) for l in labels],
                device=device
            )
            
            if idx_to_ordinal is not None and len(idx_to_ordinal) > 0:
                ordinal_original_indices = sorted(idx_to_ordinal.keys())
                if len(ordinal_original_indices) < scores.size(1):
                    ordinal_indices = torch.tensor(ordinal_original_indices, dtype=torch.long, device=device)
                    ordinal_scores = scores[:, ordinal_indices]
                else:
                    ordinal_scores = scores
            else:
                ordinal_scores = scores
            
            loss_fn = ordinal_info['loss_fn'].to(device)
            ce_scalar, ordinal_metrics = loss_fn(ordinal_scores, ordinal_labels)
            loss = ce_scalar.expand(batch_size) 
            
        elif use_focal and num_classes >= focal_threshold:
            from models.causal_gnn.focal_loss import FocalLoss
            focal_alpha = getattr(args, 'focal_loss_alpha', 0.25)
            focal_gamma = getattr(args, 'focal_loss_gamma', 2.0)
            focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
            loss = focal_loss_fn(scores, labels)
            
        else:
            if hasattr(args, 'label_smoothing') and args.label_smoothing > 0.0 and split == "train":
                loss = F.cross_entropy(scores, labels, reduction="none", label_smoothing=args.label_smoothing)
            else:
                loss = F.cross_entropy(scores, labels, reduction="none")
        
        if not isinstance(loss, torch.Tensor):
             loss = torch.tensor(loss, device=device, requires_grad=True).expand(batch_size)

        if split == "train":
            reg_loss_scalar = l1_lambda * l1_reg + ent_lambda * ent_reg + stab_lambda * 0.0
            
            improvement_losses = aux.get('improvement_losses', {})
            if 'numerical_contrastive' in improvement_losses:
                lambda_num_contrast = args.lambda_numerical_contrastive if hasattr(args, 'lambda_numerical_contrastive') else 0.1
                reg_loss_scalar += lambda_num_contrast * improvement_losses['numerical_contrastive']
            
            if hasattr(model, 'is_causal_gnn') and model.is_causal_gnn and hasattr(model, 'use_enhancement') and model.use_enhancement:
                lambda_iso = args.lambda_isotropy if hasattr(args, 'lambda_isotropy') else 0.05
                lambda_sim_ent = args.lambda_sim_entropy if hasattr(args, 'lambda_sim_entropy') else 0.1
                lambda_dag = args.lambda_dag_topo if hasattr(args, 'lambda_dag_topo') else 0.01
                lambda_edge = args.lambda_edge_smooth if hasattr(args, 'lambda_edge_smooth') else 0.05
                
                reg_loss_scalar += lambda_iso * iso_loss + lambda_sim_ent * sim_ent_loss + \
                                   lambda_dag * dag_topo_loss + lambda_edge * edge_smooth_loss
            loss = loss + reg_loss_scalar

        acc = (scores.argmax(1) == labels).float()
        if split == "train":
            loss_for_backward = (loss.sum() / batch_size) / grad_accum_steps
            loss_for_backward.backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # [FIXED] Wrap float value in list for RunningMean update
        each_loss_mean[task_name].update([loss.detach().mean().item()]) 
        each_acc_mean[task_name].update([acc.detach().mean().item()])
        
        if not individual_scores and split != "train":
            loss_mean.update([loss.detach().mean().item()])
            acc_mean.update([acc.detach().mean().item()])
        else:
            if split == "train":
                loss_mean.update([loss.detach().mean().item()])
                acc_mean.update([acc.detach().mean().item()])
                global_loss = loss_mean.value
                global_acc = acc_mean.value
            else:
                global_loss = mean_of_mean(each_loss_mean.values())
                global_acc = mean_of_mean(each_acc_mean.values())
        
        if split == "train":
            pbar.set_description(f"loss: {loss_mean.value:.4f}, acc: {acc_mean.value:.4f}")
        else:
            current_loss = global_loss if 'global_loss' in locals() else mean_of_mean(each_loss_mean.values())
            current_acc = global_acc if 'global_acc' in locals() else mean_of_mean(each_acc_mean.values())
            pbar.set_description(f"loss: {current_loss:.4f}, acc: {current_acc:.4f}")

    if split == "train":
        final_loss = loss_mean.value
        final_acc = acc_mean.value
    else:
        if not individual_scores:
            final_loss = loss_mean.value
            final_acc = acc_mean.value
        else:
            final_loss = mean_of_mean(each_loss_mean.values())
            final_acc = mean_of_mean(each_acc_mean.values())
    
    log_print(f"{split} epoch-{i_epoch}:"
              f" average loss: {final_loss}, average accuracy: {final_acc}", log_file)
    
    if individual_scores or split == "train":
        log_print(f"individual loss: {dict(each_loss_mean)}", log_file)
        log_print(f"individual acc: {dict(each_acc_mean)}", log_file)
    
    task_accs = {k: v.value for k, v in each_acc_mean.items()}
    
    return {
        'loss': final_loss,
        'acc': final_acc,
        'task_accs': task_accs
    }


def save_model(model, optimizer, scheduler, start_epoch, filename):
    print(f"saving model to {filename}")
    torch.save(
        [model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
         start_epoch],
        filename
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, 'train.log')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    print(f"📝 日志将保存到: {log_file_path}")
    log_file.write(f"实验配置: {args}\n")
    log_file.write("="*80 + "\n")
    log_file.flush()
    
    data = get_data(args.data_dir, args.subtask, args.single_task,
                    args.zero_shot, args.remove_label_text)
    device = torch.device("cpu") if torch.cuda.device_count() == 0 else \
        torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    bert = AutoModel.from_pretrained(args.model_name_or_path)

    for task in data.values():
        task["guidelines"] = Guidelines(
            task["explanations"], tokenizer, task["target_key"], True)

    train_dataset = CLUES_Dataset(data, tokenizer, "train", args.data_variance)
    dev_dataset = CLUES_Dataset(
        data, tokenizer, "valid" if args.subtask == "clues_syn" else "dev",
        args.data_variance)
    test_dataset = CLUES_Dataset(data, tokenizer, "test", args.data_variance)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=CLUES_collate,
        batch_sampler=CLUES_BatchSampler(
            train_dataset, args.per_device_train_batch_size,
            True, args.normalize_datasets, args.batch_per_epoch))
    dev_dataloader = DataLoader(
        dev_dataset, collate_fn=CLUES_collate,
        batch_sampler=CLUES_BatchSampler(
            dev_dataset, args.per_device_eval_batch_size, False, False,
            args.batch_per_epoch
        ))
    test_dataloader = DataLoader(
        test_dataset, collate_fn=CLUES_collate,
        batch_sampler=CLUES_BatchSampler(
            test_dataset, args.per_device_eval_batch_size, False, False,
            args.batch_per_epoch
        ))
    
    # 留一任务法meta-dev划分
    meta_dev_dataloader = None
    if args.use_meta_dev and args.zero_shot:
        print(f"\n🔄 启用留一任务法meta-dev (比例={args.meta_dev_task_ratio})")
        train_tasks = list(train_dataset.data.keys())
        num_meta_dev = max(1, int(len(train_tasks) * args.meta_dev_task_ratio))
        random.shuffle(train_tasks)
        meta_dev_tasks = set(train_tasks[:num_meta_dev])
        actual_train_tasks = set(train_tasks[num_meta_dev:])
        
        print(f"   训练任务: {len(actual_train_tasks)} 个")
        print(f"   Meta-dev任务: {len(meta_dev_tasks)} 个 - {sorted(meta_dev_tasks)}")
        
        meta_dev_data = {k: v for k, v in data.items() if k in meta_dev_tasks}
        split_name = "valid" if args.subtask == "clues_syn" else "dev"
        meta_dev_dataset = CLUES_Dataset(meta_dev_data, tokenizer, split_name, args.data_variance)
        meta_dev_dataloader = DataLoader(
            meta_dev_dataset, collate_fn=CLUES_collate,
            batch_sampler=CLUES_BatchSampler(
                meta_dev_dataset, args.per_device_eval_batch_size, False, False,
                args.batch_per_epoch
            ))
        
        actual_train_data = {k: v for k, v in data.items() if k in actual_train_tasks}
        train_dataset = CLUES_Dataset(actual_train_data, tokenizer, "train", args.data_variance)
        train_dataloader = DataLoader(
            train_dataset, collate_fn=CLUES_collate,
            batch_sampler=CLUES_BatchSampler(
                train_dataset, args.per_device_train_batch_size,
                True, args.normalize_datasets, args.batch_per_epoch))

    if args.reasoning_backend == 'clore':
        model = GuidedBertClues(
            bert, args.phase,
            args.encoder_feature_layer, args.guideline_feature_layer,
            args.encoder_feature_layer_num_head,
            args.guideline_feature_layer_num_head,
            args.num_feature_layer,
            args.model_version, args.mask_out_templates
        ).to(device)
    else:
        dag_per_task = {}
        if args.dag_source == 'manual':
            assert args.manual_dag_file is not None, "--manual_dag_file 不能为空"
            with open(args.manual_dag_file, 'r') as f:
                dag_dict_all = json.load(f)
        for task_name, task in data.items():
            if args.dag_source == 'manual':
                dag_dict = dag_dict_all[task_name]
                raw_nodes = dag_dict.get('nodes', [])
                nodes = []
                for idx, n in enumerate(raw_nodes):
                    nid = n.get('id')
                    if nid is None:
                        nid = idx
                    name = n.get('name', f"var_{nid}")
                    text = n.get('text', '')
                    nodes.append(CausalNode(node_id=int(nid), name=name, text=text, text_embed=None))

                raw_edges = dag_dict.get('edges', [])
                edges = []
                for e in raw_edges:
                    src = e.get('src', e.get('source', e.get('from')))
                    dst = e.get('dst', e.get('target', e.get('to')))
                    w = e.get('w', e.get('weight', 1.0))
                    if src is None or dst is None:
                        continue
                    edges.append(CausalEdge(src=int(src), dst=int(dst), weight=float(w)))
                labels = dag_dict.get('labels', [])
                label_nodes = dag_dict.get('label_nodes', {})
                dag = CausalDAG(nodes, edges, labels=labels, label_nodes=label_nodes)
            else:
                dag_json = build_stats_driven_dag(
                    task_name, task, tokenizer, bert, device, 
                    edge_keep_ratio=1.0-(1.0-args.dag_keep_ratio),
                    use_full_formula=args.dag_use_full_formula,
                    use_field_level=args.use_field_level,
                    field_weight_mode=args.field_weight_mode
                )
                
                if args.dag_fusion_mode != 'none' and args.manual_dag_file is not None:
                    from models.causal_graph.dag_fusion import DAGFusion, load_explain_dag
                    explain_dag = load_explain_dag(task_name, args.manual_dag_file)
                    if explain_dag is not None:
                        fusion = DAGFusion(
                            fusion_mode=args.dag_fusion_mode,
                            alpha=args.dag_fusion_alpha
                        )
                        dag_json = fusion.fuse_dags(explain_dag, dag_json, task_name)
                        print(f"✅ 任务 {task_name}: DAG融合完成 (模式={args.dag_fusion_mode}, α={args.dag_fusion_alpha})")
                    else:
                        print(f"⚠️  任务 {task_name}: 未找到解释性DAG,使用纯统计DAG")
                raw_nodes = dag_json.get('nodes', [])
                nodes = []
                for idx, n in enumerate(raw_nodes):
                    nid = n.get('id', idx)
                    name = n.get('name', f"var_{nid}")
                    text = n.get('text', '')
                    nodes.append(CausalNode(node_id=int(nid), name=name, text=text, text_embed=None))
                raw_edges = dag_json.get('edges', [])
                edges = []
                for e in raw_edges:
                    edges.append(CausalEdge(src=int(e['src']), dst=int(e['dst']), weight=float(e.get('w', 1.0))))
                labels = dag_json.get('labels', [])
                label_nodes = dag_json.get('label_nodes', {lbl: [i] for i, lbl in enumerate(labels)})
                dag = CausalDAG(nodes, edges, labels=labels, label_nodes=label_nodes)
            
            if getattr(dag, 'labels', None) is None or len(dag.labels) == 0:
                dag.labels = list(task.get('target_key', []))
            
            dag_per_task[task_name] = dag
        
        enhancement_config = None
        if args.use_enhancement:
            enhancement_config = {
                'use_whitening': args.enh_whitening,
                'use_isotropy': args.enh_isotropy,
                'use_similarity_entropy': args.enh_sim_entropy,
                'use_noise': args.enh_noise,
                'use_temperature': args.enh_temperature,
                'use_local_calib': args.enh_local_calib,
                'whitening_eps': 1e-5,
                'isotropy_target': args.enh_isotropy_target,
                'entropy_temperature': args.enh_entropy_temp,
                'noise_std': args.enh_noise_std,
                'temp_init': args.enh_temp_init,
                'temp_learnable': args.enh_temp_learnable,
                'local_calib_k': args.enh_local_k,
            }
        
        if args.use_improved_model:
            model = ImprovedCausalGNNClues(
                bert, tokenizer, dag_per_task,
                hidden_dim=256, num_heads=4, num_layers=2,
                label_details_path=args.label_details_path,
                use_token_sim=args.use_token_sim,
                use_bert_token_sim=args.use_bert_token_sim,
                use_enhancement=args.use_enhancement,
                enhancement_config=enhancement_config,
                use_hybrid_encoder=args.use_hybrid_encoder,
                use_cross_attention=args.use_cross_attention,
                use_ordinal_loss=args.use_ordinal_loss,
                use_focal_loss=args.use_focal_loss,
                use_adaptive_dropout=args.use_adaptive_dropout,
                use_numerical_contrastive=args.use_numerical_contrastive,
                use_task_contrastive=args.use_task_contrastive,
                use_symbolic_encoder=args.use_symbolic_encoder,
                use_task_embedding=args.use_task_embedding,
                use_task_adapter=args.use_task_adapter,
                task_type_json=args.task_type_json,
                real_task_details_json=args.real_task_details_json,
                adapter_dim=args.adapter_dim,
                ordinal_loss_alpha=args.ordinal_loss_alpha,
                ordinal_loss_beta=args.ordinal_loss_beta,
                numerical_contrastive_temp=args.numerical_contrastive_temp,
                numerical_contrastive_weight=args.numerical_contrastive_weight,
                task_contrastive_temp=args.task_contrastive_temp
            ).to(device)
            print(f"[Main] 已初始化改进版CausalGNN模型")
        else:
            model = CausalGNNClues(bert, tokenizer, dag_per_task,
                                  hidden_dim=256, num_heads=4, num_layers=2,
                                  label_details_path=args.label_details_path,
                                  use_token_sim=args.use_token_sim,
                                  use_bert_token_sim=args.use_bert_token_sim,
                                  use_enhancement=args.use_enhancement,
                                  enhancement_config=enhancement_config).to(device)
            print(f"[Main] 已初始化标准CausalGNN模型")
        
        setattr(model, 'zs_readout', args.zs_readout)
        setattr(model, 'zs_topk', args.zs_topk)
        setattr(model, 'zs_tau', args.zs_tau)
        
        if args.zs_task_topk_json and os.path.exists(args.zs_task_topk_json):
            with open(args.zs_task_topk_json, 'r') as f:
                setattr(model, 'zs_task_topk_map', json.load(f))
        if args.zs_task_tau_json and os.path.exists(args.zs_task_tau_json):
            with open(args.zs_task_tau_json, 'r') as f:
                setattr(model, 'zs_task_tau_map', json.load(f))
        if args.zs_task_mix_alpha_json and os.path.exists(args.zs_task_mix_alpha_json):
            with open(args.zs_task_mix_alpha_json, 'r') as f:
                setattr(model, 'zs_task_mix_alpha_map', json.load(f))
        setattr(model, 'zs_adapt', args.zs_adapt)
        setattr(model, 'zs_adapt_topk_pct', args.zs_adapt_topk_pct)
        setattr(model, 'zs_hist_dir', args.zs_hist_dir)
        setattr(model, 'zs_hist_bins', args.zs_hist_bins)
        
        # [CRITICAL FIX] 必须执行与 Wrapper 相同的反转逻辑
        if args.task_type_json and os.path.exists(args.task_type_json):
            with open(args.task_type_json, 'r') as f:
                raw_data = json.load(f)
                # 检查是否需要反转
                flat_map = {}
                # 尝试判断是否是 {"type": ["task1"...]} 格式
                first_val = next(iter(raw_data.values()))
                if isinstance(first_val, list):
                    for t_type, t_list in raw_data.items():
                        for t_name in t_list:
                            flat_map[t_name] = t_type
                else:
                    flat_map = raw_data
                
                setattr(model, 'task_type_map', flat_map)
                print(f"✅ [Main] 已重新加载并覆盖任务类型映射: {len(flat_map)} 个任务")

        setattr(model, 'gating_g0', args.gating_g0)
        setattr(model, 'gating_gmin', args.gating_gmin)
        setattr(model, 'topbeta_ratio', args.topbeta_ratio)
        setattr(model, 'sim_temp_tok', args.sim_temp_tok)
        setattr(model, 'sim_temp_cls', args.sim_temp_cls)
        setattr(model, 'phase1_delta_topk', args.phase1_delta_topk)
        setattr(model, 'phase1_delta_tau', args.phase1_delta_tau)
        setattr(model, 'phase1_delta_mix', args.phase1_delta_mix)
        setattr(model, 'gnn_subgraph_topk', args.gnn_subgraph_topk)
        
        dag_regularizer = None
        if args.use_dag_regularization:
            from models.causal_graph.dag_regularization import DAGRegularization
            dag_regularizer = DAGRegularization(
                lambda_dag=args.lambda_dag_constraint,
                lambda_sparsity=args.lambda_dag_sparsity,
                lambda_smooth=args.lambda_dag_smooth,
                lambda_explain=args.lambda_dag_explain,
                lambda_causal=args.lambda_causal_gnn,
                dag_mode=args.dag_constraint_mode
            ).to(device)
            print(f"✅ 启用DAG正则化 (DAG={args.lambda_dag_constraint}, 稀疏={args.lambda_dag_sparsity})")
        
        causal_gnn_regularizer = None
        if args.use_causal_gnn_reg:
            from models.causal_graph.dag_regularization import CausalGNNRegularizer
            causal_gnn_regularizer = CausalGNNRegularizer(
                lambda_intervene=args.lambda_intervene,
                lambda_counterfactual=args.lambda_counterfactual
            ).to(device)
            print(f"✅ 启用因果GNN正则 (干预={args.lambda_intervene}, 反事实={args.lambda_counterfactual})")
        
        guideline_alignment = None
        if args.use_guideline_alignment:
            from models.causal_graph.guideline_alignment import GuidelineAlignment
            guideline_alignment = GuidelineAlignment(
                alignment_mode=args.alignment_mode,
                constraint_weight=args.alignment_weight,
                temperature=args.alignment_temperature
            ).to(device)
            print(f"✅ 启用解释一致性对齐 (模式={args.alignment_mode}, 权重={args.alignment_weight})")
        
        setattr(model, 'dag_regularizer', dag_regularizer)
        setattr(model, 'causal_gnn_regularizer', causal_gnn_regularizer)
        setattr(model, 'guideline_alignment', guideline_alignment)
        
    if args.freeze_encoder:
        print(f"\n❄️  完全冻结BERT编码器,只训练GNN和读出层")
        for name, param in model.named_parameters():
            if 'encoder' in name or 'bert' in name.lower():
                param.requires_grad = False
        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = {
            "AdamW": AdamW,
            "Adam": Adam,
            "SGD": SGD
        }[args.optimizer](optimizer_params, args.learning_rate, weight_decay=args.weight_decay)
    elif args.encoder_lr_ratio < 1.0:
        print(f"\n🔧 编码器学习率降至主干的 {args.encoder_lr_ratio:.1%} (半冻结策略)")
        encoder_params_ids = set()
        encoder_params_group = []
        other_params_group = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'encoder' in name or 'bert' in name.lower():
                encoder_params_group.append(param)
                encoder_params_ids.add(id(param))
            else:
                other_params_group.append(param)
        
        for param in model.parameters():
            if param.requires_grad and id(param) not in encoder_params_ids:
                pass 

        optimizer_grouped_parameters = [
            {'params': encoder_params_group, 'lr': args.learning_rate * args.encoder_lr_ratio, 'name': 'encoder_params'},
            {'params': other_params_group, 'lr': args.learning_rate, 'name': 'other_params'}
        ]
        
        print(f"   🔧 优化器分组详情:")
        print(f"     - Encoder组 ({len(encoder_params_group)} params): LR = {args.learning_rate * args.encoder_lr_ratio:.2e}")
        print(f"     - Other组   ({len(other_params_group)} params): LR = {args.learning_rate:.2e}")
        
        optimizer = {
            "AdamW": AdamW,
            "Adam": Adam,
            "SGD": SGD
        }[args.optimizer](optimizer_grouped_parameters, weight_decay=args.weight_decay)
    else:
        optimizer = {
            "AdamW": AdamW,
            "Adam": Adam,
            "SGD": SGD
        }[args.optimizer](model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    
    # 考虑梯度累积步数
    num_update_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs
    warmup_steps = 0.1 * num_update_steps
    scheduler = LambdaLR(
        optimizer, lr_lambda={
            "": lambda step: 1,
            "warmup": lambda step: min(
                (step+1) ** (-0.5), (step+1) * warmup_steps ** (-1.5)
            ) * warmup_steps ** 0.5
        }[args.scheduler]
    )
    last_model = args.output_dir+"/last_model.pt"
    start_epoch = 0
    if os.path.exists(last_model):
        print("loading from checkpoint:", last_model)
        ckpt = torch.load(last_model)
        model.load_state_dict(ckpt[0])
        optimizer.load_state_dict(ckpt[1])
        scheduler.load_state_dict(ckpt[2])
        start_epoch = ckpt[3] + 1

    if args.num_train_epochs == 0:
        print("\n🔎 检测到 num_train_epochs=0，仅进行评估(不训练)")
        dev_result = evaluate_with_multiple_runs(
            dev_dataloader, model, tokenizer, device,
            "dev", 0, args.individual_scores, args=args, log_file=log_file
        )
        test_result = evaluate_with_multiple_runs(
            test_dataloader, model, tokenizer, device,
            "test", 0, args.individual_scores, args=args, log_file=log_file
        )
        print("\n================================================================")
        print("评估结果(仅评估模式, 不进行训练):")
        print(dict(zip(
            ["dev_loss", "dev_acc", "test_loss", "test_acc"],
            [dev_result['loss'], dev_result['acc'],
             test_result['loss'], test_result['acc']]
        )))
        log_file.close()
        print(f"✅ 日志已保存到: {log_file_path}")
        return

    best_dev_acc = -math.inf
    best_epoch = -1
    best_epoch_dev_loss = math.inf
    best_epoch_test_acc = -math.inf
    best_epoch_test_loss = math.inf

    for i_epoch in tqdm(range(start_epoch, args.num_train_epochs)):
        log_file.write("="*64 + "\n")
        log_file.flush()
        print("="*64)
        epoch_msg = f"开始 epoch {i_epoch+1}/{args.num_train_epochs}"
        print(epoch_msg)
        log_file.write(epoch_msg + "\n")
        log_file.flush()
        
        if hasattr(model, 'set_progress'):
            try:
                model.set_progress((i_epoch + 1) / max(1, args.num_train_epochs))
            except Exception:
                pass
        one_epoch(train_dataloader, model, optimizer, scheduler, tokenizer,
                  device, "train", i_epoch, False, attn_ent_weight=args.attn_ent_weight, args=args, log_file=log_file)
        
        if args.use_meta_dev and meta_dev_dataloader is not None:
            meta_dev_msg = "   📊 使用Meta-dev做早停评估 (模拟未见任务)"
            print(meta_dev_msg)
            log_file.write(meta_dev_msg + "\n")
            log_file.flush()
            meta_dev_result = evaluate_with_multiple_runs(
                meta_dev_dataloader, model, tokenizer, device,
                "meta-dev", i_epoch, args.individual_scores, args=args, log_file=log_file)
            dev_loss, dev_acc = meta_dev_result['loss'], meta_dev_result['acc']
            
            regular_dev_result = evaluate_with_multiple_runs(
                dev_dataloader, model, tokenizer, device,
                "dev", i_epoch, args.individual_scores, args=args, log_file=log_file)
            test_result = evaluate_with_multiple_runs(
                test_dataloader, model, tokenizer, device,
                "test", i_epoch, args.individual_scores, args=args, log_file=log_file)
            test_loss, test_acc = test_result['loss'], test_result['acc']
        else:
            dev_result = evaluate_with_multiple_runs(
                dev_dataloader, model, tokenizer, device,
                "dev", i_epoch, args.individual_scores, args=args, log_file=log_file)
            dev_loss, dev_acc = dev_result['loss'], dev_result['acc']
            
            test_result = evaluate_with_multiple_runs(
                test_dataloader, model, tokenizer, device,
                "test", i_epoch, args.individual_scores, args=args, log_file=log_file)
            test_loss, test_acc = test_result['loss'], test_result['acc']

        ckpt_file = f"{args.output_dir}/epoch{i_epoch}_model.pt"
        save_model(model, optimizer, scheduler, i_epoch, ckpt_file)
        save_model(model, optimizer, scheduler, i_epoch, last_model)
        
        if args.zero_shot and args.use_test_early_stop:
            if test_acc > best_epoch_test_acc:
                best_model_file = f"{args.output_dir}/best_model.pt"
                print(f"🎯 更新最佳模型 (test acc: {test_acc:.4f}, 之前: {best_epoch_test_acc:.4f})")
                save_model(
                    model, optimizer, scheduler, i_epoch, best_model_file)
                best_dev_acc = dev_acc
                best_epoch = i_epoch
                best_epoch_dev_loss = dev_loss
                best_epoch_test_acc = test_acc
                best_epoch_test_loss = test_loss
        else:
            if dev_acc > best_dev_acc:
                best_model_file = f"{args.output_dir}/best_model.pt"
                early_stop_source = "meta-dev" if (args.use_meta_dev and meta_dev_dataloader is not None) else "dev"
                print(f"更新最佳模型 ({early_stop_source} acc: {dev_acc:.4f})")
                save_model(
                    model, optimizer, scheduler, i_epoch, best_model_file)
                best_dev_acc = dev_acc
                best_epoch = i_epoch
                best_epoch_dev_loss = dev_loss
                best_epoch_test_acc = test_acc
                best_epoch_test_loss = test_loss

    if start_epoch != args.num_train_epochs:
        print("\n================================================================")
        print("最终结果:")
        print(dict(zip(["最佳epoch", "最佳验证集准确率", "最佳验证集损失",
                        "最佳测试集准确率", "最佳测试集损失"],
                       [best_epoch, best_dev_acc, best_epoch_dev_loss,
                        best_epoch_test_acc, best_epoch_test_loss])))
        
        if args.enable_temperature_calibration and args.zero_shot:
            print("\n" + "="*64)
            print("🌡️  开始任务级温度标定 (基于meta-dev或dev)")
            
            best_model_file = f"{args.output_dir}/best_model.pt"
            if os.path.exists(best_model_file):
                checkpoint = torch.load(best_model_file, map_location=device)
                model.load_state_dict(checkpoint[0]) 
                print(f"    已加载最佳模型 (epoch {checkpoint[3]})")
            
            calib_dataloader = meta_dev_dataloader if (args.use_meta_dev and meta_dev_dataloader is not None) else dev_dataloader
            calib_name = "meta-dev" if (args.use_meta_dev and meta_dev_dataloader is not None) else "dev"
            print(f"    使用 {calib_name} 数据集进行标定")
            
            from models.causal_gnn.temperature_calibration import (
                TaskWiseTemperatureCalibration, collect_logits_and_labels
            )
            
            print("    收集logits和labels...")
            task_logits_dict, task_labels_dict = collect_logits_and_labels(
                model, calib_dataloader, device, tokenizer
            )
            
            print(f"    对 {len(task_logits_dict)} 个任务进行温度标定:")
            calibrator = TaskWiseTemperatureCalibration()
            calibrator.calibrate(task_logits_dict, task_labels_dict, verbose=True)
            
            temp_file = f"{args.output_dir}/task_temperatures.json"
            calibrator.save(temp_file)
            
            print(f"\n    ✅ 温度标定完成! 可在推理时使用 --load_temperature {temp_file}")
    else:
        test_result = one_epoch(
            test_dataloader, model, optimizer, scheduler, tokenizer, device,
            "test", start_epoch, args.individual_scores, args=args, log_file=log_file)
        test_loss, test_acc = test_result['loss'], test_result['acc']
    
    log_file.close()
    print(f"✅ 日志已保存到: {log_file_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)