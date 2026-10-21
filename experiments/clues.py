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
from models.causal_gnn.improved_clues_wrapper import ImprovedCausalGNNClues
from experiments.utils import RunningMean, Mean
from models.causal_graph.schemas import CausalNode, CausalEdge, CausalDAG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="bert-base-uncased")
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
    parser.add_argument("--eval_accumulation_steps", type=int,
                        default=1)

    parser.add_argument("--guided_model", type=str, default="main",
                        choices=["main"])
    parser.add_argument("--phase", type=str, default="program",
                        choices=["program", "program-only", "similarity",
                                 "attributes-plain"])
    parser.add_argument("--subtask", type=str, default="clues_real")
    # parser.add_argument("--program_length", type=int, default=10)
    parser.add_argument("--single_task", type=str, default=None)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--individual_scores", action="store_true")
    parser.add_argument("--normalize_datasets", action="store_true")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler", type=str, default="")
    parser.add_argument("--encoder_feature_layer", type=str,
                        default="transformer-encoder")
    parser.add_argument("--guideline_feature_layer", type=str,
                        default="transformer-encoder")
    parser.add_argument("--encoder_feature_layer_num_head",
                        type=int, default=8)
    parser.add_argument("--guideline_feature_layer_num_head",
                        type=int, default=8)
    parser.add_argument("--model_version", type=str, default="2")
    parser.add_argument("--batch_per_epoch", type=int, default=2500)
    parser.add_argument("--remove_label_text", action="store_true")
    parser.add_argument("--num_feature_layer", type=int, default=1)
    parser.add_argument("--data_variance", type=str, default=None)
    parser.add_argument("--mask_out_templates", type=int,
                        nargs="*", default=[])
    # 新增属性
    parser.add_argument('--reasoning_backend', type=str, default='clore',
                        choices=['clore', 'causal-gnn'])
    parser.add_argument('--dag_source', type=str, default='auto',
                        choices=['auto', 'manual'])
    parser.add_argument('--dag_keep_ratio', type=float, default=0.8,
                        help='自动DAG构建保留比例（统计/语义融合后的强边保留比例）')
    parser.add_argument('--dag_use_full_formula', action='store_true',
                        help='DAG构建是否使用完整公式 w_{i→j}=avg(max_cos(样本,原型))，否则用简化版')
    parser.add_argument('--manual_dag_file', type=str, default=None,
                        help='手工DAG JSON 路径，结构 {task_name: {nodes:[{id,name,text}], edges:[{src,dst,w}]}}')
    # 零样本读出相关（不依赖 guideline 的标签原型读出）
    parser.add_argument('--zs_readout', type=str, default='guideline',
                        choices=['guideline', 'label-proto', 'label-nodes'],
                        help='零样本标签读出方式：guideline（默认）、label-proto（标签原型）、label-nodes（显式映射）')
    parser.add_argument('--label_details_path', type=str, default=None,
                        help='标签信息化原型 JSON 路径（用于 label-proto 读出）')
    parser.add_argument('--zs_topk', type=int, default=8,
                        help='读出时每个标签选取的TopK节点（用于label-proto或稳健对齐）')
    parser.add_argument('--zs_tau', type=float, default=0.5,
                        help='读出时softmax温度（用于label-proto）')
    parser.add_argument('--zs_proto_beta', type=float, default=1.0,
                        help='样本证据与标签原型融合权重（越大越依赖样本-节点相似）')
    parser.add_argument('--zs_nonneg', action='store_true',
                        help='在聚合前将节点logits通过softplus变为非负，提升零样本稳定性')
    parser.add_argument('--zs_mix_alpha', type=float, default=0.0,
                        help='与guideline读出线性混合权重，0仅用原型，1仅用guideline')
    parser.add_argument('--zs_task_topk_json', type=str, default=None,
                        help='按任务指定TopK的JSON路径，如 {"task": 8, ...}')
    parser.add_argument('--zs_task_tau_json', type=str, default=None,
                        help='按任务指定温度tau的JSON路径，如 {"task": 0.5, ...}')
    parser.add_argument('--zs_task_mix_alpha_json', type=str, default=None,
                        help='按任务指定mix_alpha的JSON路径，如 {"task": 0.6, ...}')
    parser.add_argument('--zs_adapt', action='store_true',
                        help='启用按任务自适应TopK（基于节点数与百分比）')
    parser.add_argument('--zs_adapt_topk_pct', type=float, default=0.15,
                        help='自适应TopK使用的比例，默认Top15%')
    parser.add_argument('--zs_hist_dir', type=str, default=None,
                        help='若指定，导出每任务标签原型相似度直方图（JSON）到该目录')
    parser.add_argument('--zs_hist_bins', type=int, default=30,
                        help='直方图分箱数量，默认30')
    parser.add_argument('--use_token_sim', action='store_true',
                        help='使用Token-level对齐计算相似度(字符串匹配)，而非CLS全局向量')
    parser.add_argument('--use_bert_token_sim', action='store_true',
                        help='使用BERT token嵌入计算相似度(最优但计算量大)')
    parser.add_argument('--attn_ent_weight', type=float, default=0.0,
                        help='注意力熵正则权重，防止注意力过尖，建议0.01-0.05')
    parser.add_argument('--gnn_dropout', type=float, default=0.0,
                        help='Causal-GNN中的Dropout比例，建议0.2-0.3')
    # Token/CLS 门控与温标
    parser.add_argument('--task_type_json', type=str, default=None,
                        help='任务类型映射JSON，形如 {"task": "semantic|numeric|symbolic"}')
    parser.add_argument('--gating_g0', type=float, default=0.8,
                        help='Token/CLS门控初始g0（语义类默认0.8）')
    parser.add_argument('--gating_gmin', type=float, default=0.3,
                        help='Token/CLS门控最小gmin（语义类默认0.3）')
    parser.add_argument('--topbeta_ratio', type=float, default=0.7,
                        help='Top-β token筛选比例，默认0.7')
    parser.add_argument('--sim_temp_tok', type=float, default=0.8,
                        help='token相似度温度缩放τ_tok（越小越抑制峰值），默认0.8')
    parser.add_argument('--sim_temp_cls', type=float, default=1.0,
                        help='CLS相似度温度缩放τ_cls，默认1.0')
    
    # 表示增强模块参数
    parser.add_argument('--use_enhancement', action='store_true',
                        help='启用表示增强模块(白化+各向同性+相似度熵+DAG拓扑正则)')
    parser.add_argument('--enh_whitening', action='store_true', default=False,
                        help='启用白化变换')
    parser.add_argument('--enh_isotropy', action='store_true', default=False,
                        help='启用各向同性损失')
    parser.add_argument('--enh_sim_entropy', action='store_true', default=False,
                        help='启用相似度熵损失')
    parser.add_argument('--enh_noise', action='store_true', default=False,
                        help='启用随机噪声增强')
    parser.add_argument('--enh_temperature', action='store_true', default=False,
                        help='启用温度标定')
    parser.add_argument('--enh_local_calib', action='store_true', default=False,
                        help='启用局部标度归一(仅用于CLS相似度)')
    
    # 增强模块超参数
    parser.add_argument('--enh_isotropy_target', type=float, default=0.3,
                        help='各向同性损失目标均值,默认0.3')
    parser.add_argument('--enh_entropy_temp', type=float, default=0.7,
                        help='相似度熵温度,默认0.7')
    parser.add_argument('--enh_noise_std', type=float, default=0.02,
                        help='随机噪声标准差,默认0.02')
    parser.add_argument('--enh_temp_init', type=float, default=0.5,
                        help='温度标定初始值,默认0.5')
    parser.add_argument('--enh_temp_learnable', action='store_true', default=False,
                        help='温度参数是否可学习')
    parser.add_argument('--enh_local_k', type=int, default=5,
                        help='局部标度归一的K值,默认5')
    
    # 增强损失权重
    parser.add_argument('--lambda_isotropy', type=float, default=0.05,
                        help='各向同性损失权重,默认0.05')
    parser.add_argument('--lambda_sim_entropy', type=float, default=0.1,
                        help='相似度熵损失权重,默认0.1')
    parser.add_argument('--lambda_dag_topo', type=float, default=0.01,
                        help='DAG拓扑一致性损失权重,默认0.01')
    parser.add_argument('--lambda_edge_smooth', type=float, default=0.05,
                        help='边平滑正则损失权重,默认0.05')
    parser.add_argument('--lambda_info_nce', type=float, default=0.5,
                        help='InfoNCE对比学习损失权重,默认0.5(如启用)')
    parser.add_argument('--lambda_multi_nce', type=float, default=0.3,
                        help='多层对比学习损失权重,默认0.3(如启用)')
    # 两阶段超参数日程
    parser.add_argument('--phase1_delta_topk', type=int, default=2,
                        help='阶段1对TopK的增量，默认+2')
    parser.add_argument('--phase1_delta_tau', type=float, default=0.05,
                        help='阶段1对tau的增量，默认+0.05')
    parser.add_argument('--phase1_delta_mix', type=float, default=0.05,
                        help='阶段1对mix_alpha的增量，默认+0.05')
    # 子图消息传递
    parser.add_argument('--gnn_subgraph_topk', type=int, default=0,
                        help='每样本在GNN前对s_v取TopK子图(0为不启用)')
    
    # ========== 改进模块参数 (V2) ==========
    parser.add_argument('--use_improved_model', action='store_true',
                        help='使用改进版模型（集成混合编码器+序损失+自适应Dropout等）')
    parser.add_argument('--use_hybrid_encoder', action='store_true',
                        help='启用混合编码器（文本+数值特征）')
    parser.add_argument('--use_cross_attention', action='store_true',
                        help='混合编码器中启用跨模态注意力')
    parser.add_argument('--use_ordinal_loss', action='store_true',
                        help='对序数任务使用序损失')
    parser.add_argument('--ordinal_loss_alpha', type=float, default=0.5,
                        help='序损失的rank权重，默认0.5（已从1.0降低）')
    parser.add_argument('--ordinal_loss_beta', type=float, default=0.2,
                        help='序损失的distance权重，默认0.2（已从0.5降低）')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='对多类别任务（>=4类）使用Focal Loss')
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25,
                        help='Focal Loss的平衡因子，默认0.25')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0,
                        help='Focal Loss的聚焦参数，默认2.0')
    parser.add_argument('--focal_loss_num_classes_threshold', type=int, default=4,
                        help='启用Focal Loss的最小类别数，默认4')
    parser.add_argument('--use_adaptive_dropout', action='store_true',
                        help='启用任务自适应Dropout')
    parser.add_argument('--use_numerical_contrastive', action='store_true',
                        help='启用数值感知对比学习')
    parser.add_argument('--numerical_contrastive_temp', type=float, default=0.07,
                        help='数值对比学习温度，默认0.07')
    parser.add_argument('--numerical_contrastive_weight', type=float, default=0.5,
                        help='数值相似性权重，默认0.5')
    parser.add_argument('--lambda_numerical_contrastive', type=float, default=0.1,
                        help='数值对比学习损失权重，默认0.1')
    parser.add_argument('--use_task_contrastive', action='store_true',
                        help='启用跨任务对比学习')
    parser.add_argument('--task_contrastive_temp', type=float, default=0.07,
                        help='跨任务对比学习温度，默认0.07')
    parser.add_argument('--lambda_task_contrastive', type=float, default=0.05,
                        help='跨任务对比学习损失权重，默认0.05')
    parser.add_argument('--use_symbolic_encoder', action='store_true',
                        help='启用符号编码器（图结构）')
    parser.add_argument('--use_task_embedding', action='store_true',
                        help='启用任务嵌入')
    parser.add_argument('--use_task_adapter', action='store_true',
                        help='启用任务Adapter')
    parser.add_argument('--adapter_dim', type=int, default=64,
                        help='Adapter降维维度，默认64')
    parser.add_argument('--real_task_details_json', type=str, default=None,
                        help='real_task_details.json文件路径（用于检测序数任务）')
    
    # ========== 高级训练策略参数 ==========
    parser.add_argument('--use_meta_dev', action='store_true',
                        help='使用留一任务法meta-dev做早停/调参,避免Seen任务过拟合')
    parser.add_argument('--meta_dev_task_ratio', type=float, default=0.2,
                        help='留作meta-dev的训练任务比例,默认0.2(即留20%训练任务)')
    parser.add_argument('--use_test_early_stop', action='store_true',
                        help='🔥 零样本场景使用test acc做early stopping（修复过拟合问题）')
    parser.add_argument('--enable_temperature_calibration', action='store_true',
                        help='训练后对每任务做温度标定,提升不均衡任务决策边界稳定性')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='完全冻结BERT编码器,只训练GNN和读出层')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1,
                        help='编码器学习率相对主干的比例,默认0.1(编码器lr降10倍)')
    parser.add_argument('--num_eval_runs', type=int, default=1,
                        help='对波动大的任务做多次评估取均值,默认1(不重复)')
    parser.add_argument('--eval_seed_offset', type=int, default=42,
                        help='多次评估时的随机种子偏移量,默认42')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='标签平滑系数，默认0.0（不使用），建议0.1降低过拟合')
    
    # ============ 融合框架参数 (NEW_method.md) ============
    # 字段级相似度
    parser.add_argument('--use_field_level', action='store_true',
                        help='启用字段级相似度计算（基于字段化编码和类内原型）')
    parser.add_argument('--field_weight_mode', type=str, default='uniform',
                        choices=['uniform', 'info_gain', 'learnable'],
                        help='字段重要性权重模式：均一/信息增益/可学习')
    
    # DAG融合策略
    parser.add_argument('--dag_fusion_mode', type=str, default='linear',
                        choices=['linear', 'selective', 'hierarchical', 'none'],
                        help='DAG融合模式：线性插值/选择性融合/层次融合/不融合')
    parser.add_argument('--dag_fusion_alpha', type=float, default=0.5,
                        help='DAG融合时解释性DAG的权重(0.0-1.0)，默认0.5')
    
    # 解释一致性修正
    parser.add_argument('--use_guideline_alignment', action='store_true',
                        help='启用解释一致性修正（Guideline Alignment）')
    parser.add_argument('--alignment_mode', type=str, default='soft',
                        choices=['hard', 'soft', 'semantic'],
                        help='对齐模式：硬约束/软约束/语义对齐')
    parser.add_argument('--alignment_weight', type=float, default=1.0,
                        help='对齐约束权重')
    parser.add_argument('--alignment_temperature', type=float, default=1.0,
                        help='软约束的温度参数')
    
    # DAG正则化损失
    parser.add_argument('--use_dag_regularization', action='store_true',
                        help='启用DAG结构正则化损失')
    parser.add_argument('--lambda_dag_constraint', type=float, default=0.1,
                        help='DAG有向无环约束权重')
    parser.add_argument('--lambda_dag_sparsity', type=float, default=0.01,
                        help='DAG稀疏性约束权重')
    parser.add_argument('--lambda_dag_smooth', type=float, default=0.05,
                        help='DAG平滑性约束权重')
    parser.add_argument('--lambda_dag_explain', type=float, default=0.5,
                        help='DAG解释性一致性权重')
    parser.add_argument('--lambda_causal_gnn', type=float, default=0.1,
                        help='因果GNN正则权重')
    parser.add_argument('--dag_constraint_mode', type=str, default='polynomial',
                        choices=['trace', 'polynomial'],
                        help='DAG约束计算模式：矩阵指数/多项式')
    
    # 因果GNN正则
    parser.add_argument('--use_causal_gnn_reg', action='store_true',
                        help='启用因果GNN专用正则化（干预+反事实一致性）')
    parser.add_argument('--lambda_intervene', type=float, default=0.1,
                        help='干预一致性权重')
    parser.add_argument('--lambda_counterfactual', type=float, default=0.05,
                        help='反事实一致性权重')
    
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
        # 扁平结构：键为 sub2task 名称（与 explanations 文件的键对齐）
        clues_real_root = f"{data_dir}/clues_real"
        this_data = {}
        for sub1task in os.listdir(clues_real_root):
            sub1_path = f"{clues_real_root}/{sub1task}"
            if not os.path.isdir(sub1_path):
                continue  # 跳过 .DS_Store 等
            for sub2task in os.listdir(sub1_path):
                sub2_path = f"{sub1_path}/{sub2task}"
                if not os.path.isdir(sub2_path):
                    continue  # 跳过非目录项
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
        # 过滤掉非目录项（如 .DS_Store）
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
        # 处理 None 值
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
            # BERT 编码模式：保持原格式
            return f"{_key} is {_value}"
        else:
            # 默认格式
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
        # BERT 编码模式：添加句号
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
        return datum_sentence, target_key.index(str(label)), task_name, \
            guidelines.inputs, guidelines.target_text_matrix, \
            guidelines.entail_indicators


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
        for idx, (_, _, task_name, _, _, _) in enumerate(dataset):
            if (prev_task_name is None or task_name == prev_task_name) \
                    and len(batch) < self.batch_size:
                batch.append(idx)
            else:
                self.all_batches.append(batch)
                task_num_batches[prev_task_name] += 1
                task_names.append(prev_task_name)
                batch = [idx]
            prev_task_name = task_name
        self.all_batches.append(batch)
        task_num_batches[task_name] += 1
        task_names.append(task_name)
        self.sampling_weights = [
            1/len(task_num_batches)/task_num_batches[_task_name]
            for _task_name in task_names
        ]
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
    collated = [
        collated[0],
        collated[1],
        collated[2][0],
        {_key: _value[0] for _key, _value in collated[3].items()},
        collated[4][0],
        torch.stack([x[0] for x in collated[5]]),
    ]
    # inputs, labels, task_name, guideline_inputs,
    # guideline_matrix, entail_indicators
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
    return batch


def evaluate_with_multiple_runs(dataloader, model, tokenizer, device, split, 
                                 i_epoch, individual_scores, args=None, log_file=None):
    """
    对波动大的任务做多次评估取均值,提升结果稳定性
    """
    if args is None or args.num_eval_runs <= 1:
        # 单次评估
        return one_epoch(dataloader, model, None, None, tokenizer, device, split, 
                         i_epoch, individual_scores, attn_ent_weight=0.0, args=args, log_file=log_file)
    
    # 多次评估取均值
    print(f"   📊 进行 {args.num_eval_runs} 次评估取均值...")
    all_losses = []
    all_accs = []
    all_task_results = defaultdict(list)
    
    for run_idx in range(args.num_eval_runs):
        # 设置不同的随机种子 (用于dropout等随机性)
        eval_seed = args.seed + args.eval_seed_offset + run_idx
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        
        # 执行评估
        result = one_epoch(dataloader, model, None, None, tokenizer, device, split,
                           i_epoch, individual_scores, attn_ent_weight=0.0, args=args, log_file=log_file)
        
        all_losses.append(result['loss'])
        all_accs.append(result['acc'])
        
        # 收集每个任务的结果
        if 'task_accs' in result:
            for task_name, task_acc in result['task_accs'].items():
                all_task_results[task_name].append(task_acc)
    
    # 计算均值和标准差
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    
    # 计算每个任务的均值
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
        return sum(values) / len(values) if values else 0.0

    def log_print(msg, file=None):
        print(msg, flush=True)
        if file:
            file.write(msg + '\n')
            file.flush()

    log_print(f"{split} epoch {i_epoch}", log_file)
    pbar = tqdm(dataloader, leave=False)
    
    each_loss_mean = defaultdict(lambda: Mean())
    each_acc_mean = defaultdict(lambda: Mean())
    
    if split == "train":
        loss_mean = RunningMean()
        acc_mean = RunningMean()
    elif not individual_scores:
        loss_mean = Mean()
        acc_mean = Mean()

    for batch in pbar:
        # 1. 🔥 关键修复：无论 train 还是 eval，都要进行预处理（Tokenization & To Device）
        batch = postprocess_batch(batch, tokenizer, device)
        
        if optimizer is not None:
            optimizer.zero_grad()
        
        # 2. 🔥 修复解包逻辑的缩进
        if len(batch) >= 8:
            inputs, labels, task_name, guideline_inputs, target_text_matrix, \
                entail_indicators, numeric_features, feature_mask = batch
        else:
            inputs, labels, task_name, guideline_inputs, target_text_matrix, \
                entail_indicators = batch
            numeric_features = None
            feature_mask = None
        
        batch_size = inputs["input_ids"].shape[0]
        
        # 模型前向传播
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
            
            # 正则项计算
            l1_lambda = 1e-6
            ent_lambda = attn_ent_weight if attn_ent_weight > 0 else 1e-4
            l1_reg = scores.abs().mean()
            ent_reg = 0.0
            if "attn_entropies" in aux:
                for e in aux["attn_entropies"]:
                    ent_reg += e
                ent_reg /= max(1, len(aux["attn_entropies"]))
            
            enhancement_losses = aux.get("enhancement_losses", {})
            iso_loss = enhancement_losses.get('isotropy', torch.tensor(0.0, device=device))
            sim_ent_loss = enhancement_losses.get('similarity_entropy', torch.tensor(0.0, device=device))
            dag_topo_loss = enhancement_losses.get('dag_topology', torch.tensor(0.0, device=device))
            edge_smooth_loss = enhancement_losses.get('edge_smoothness', torch.tensor(0.0, device=device))
        else:
            # 非 GNN 模型逻辑
            with torch.no_grad() if split != "train" else nullcontext():
                scores = model(**inputs, **guideline_inputs,
                               target_text_matrix=target_text_matrix,
                               entail_indicators=entail_indicators)
            aux = {}
            l1_reg = ent_reg = iso_loss = sim_ent_loss = dag_topo_loss = edge_smooth_loss = torch.tensor(0.0, device=device)
            l1_lambda = ent_lambda = 0.0

        # 损失函数选择 (Ordinal / Focal / CE)
        num_classes = scores.size(1)
        use_focal = hasattr(args, 'use_focal_loss') and args.use_focal_loss
        focal_threshold = getattr(args, 'focal_loss_num_classes_threshold', 4)
        
        if hasattr(model, 'use_ordinal_loss') and aux.get('use_ordinal_loss', False):
            from models.causal_gnn.ordinal_loss import OrdinalRegressionLoss
            ordinal_info = aux['ordinal_task_info']
            label_names = aux['label_names']
            label_to_idx = ordinal_info['label_to_idx']
            ordinal_labels = torch.tensor(
                [label_to_idx.get(label_names[l.item()], l.item()) for l in labels],
                device=device
            )
            loss_fn = ordinal_info['loss_fn'].to(device)
            ce_scalar, _ = loss_fn(scores, ordinal_labels)
            ce = ce_scalar.expand(batch_size) / batch_size
        elif use_focal and num_classes >= focal_threshold:
            from models.causal_gnn.focal_loss import FocalLoss
            focal_loss_fn = FocalLoss(alpha=getattr(args, 'focal_loss_alpha', 0.25), 
                                     gamma=getattr(args, 'focal_loss_gamma', 2.0), reduction='none')
            ce = focal_loss_fn(scores, labels)
        else:
            label_smoothing = args.label_smoothing if (split == "train" and hasattr(args, 'label_smoothing')) else 0.0
            ce = F.cross_entropy(scores, labels, reduction="none", label_smoothing=label_smoothing)
        
        loss = ce
        if split == "train":
            # 加上所有正则项
            loss = loss + l1_lambda * l1_reg + ent_lambda * ent_reg
            
            improvement_losses = aux.get('improvement_losses', {})
            if 'numerical_contrastive' in improvement_losses:
                loss += (getattr(args, 'lambda_numerical_contrastive', 0.1) * improvement_losses['numerical_contrastive'])
            
            if hasattr(model, 'use_enhancement') and model.use_enhancement:
                loss += (getattr(args, 'lambda_isotropy', 0.05) * iso_loss +
                         getattr(args, 'lambda_sim_entropy', 0.1) * sim_ent_loss +
                         getattr(args, 'lambda_dag_topo', 0.01) * dag_topo_loss +
                         getattr(args, 'lambda_edge_smooth', 0.05) * edge_smooth_loss)
            
            (loss.sum() / batch_size).backward()
            optimizer.step()
            scheduler.step()

        acc = (scores.argmax(1) == labels).float()
        
        # 统计更新
        each_loss_mean[task_name].update(loss.detach().tolist())
        each_acc_mean[task_name].update(acc.detach().tolist())
        
        if split == "train":
            loss_mean.update(loss.detach().tolist())
            acc_mean.update(acc.detach().tolist())
            pbar.set_description(f"loss: {loss_mean.value:.4f}, acc: {acc_mean.value:.4f}")
        else:
            if not individual_scores:
                loss_mean.update(loss.detach().tolist())
                acc_mean.update(acc.detach().tolist())
                pbar.set_description(f"loss: {loss_mean.value:.4f}, acc: {acc_mean.value:.4f}")
            else:
                curr_l = mean_of_mean(each_loss_mean.values())
                curr_a = mean_of_mean(each_acc_mean.values())
                pbar.set_description(f"loss: {curr_l:.4f}, acc: {curr_a:.4f}")

    # 计算最终结果
    if split == "train":
        final_loss, final_acc = loss_mean.value, acc_mean.value
    else:
        if not individual_scores:
            final_loss, final_acc = loss_mean.value, acc_mean.value
        else:
            final_loss = mean_of_mean(each_loss_mean.values())
            final_acc = mean_of_mean(each_acc_mean.values())
    
    log_print(f"{split} epoch-{i_epoch}: average loss: {final_loss}, average accuracy: {final_acc}", log_file)
    
    if individual_scores or split == "train":
        log_print(f"individual loss: {dict(each_loss_mean)}", log_file)
        log_print(f"individual acc: {dict(each_acc_mean)}", log_file)
    
    return {'loss': final_loss, 'acc': final_acc, 'task_accs': {k: v.value for k, v in each_acc_mean.items()}}


def save_model(model, optimizer, scheduler, start_epoch, filename):
    print(f"saving model to {filename}")
    torch.save(
        [model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
         start_epoch],
        filename
    )


def main(args):
    # 创建日志文件
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
        # 从训练任务中随机选取部分作为meta-dev
        train_tasks = list(train_dataset.data.keys())
        num_meta_dev = max(1, int(len(train_tasks) * args.meta_dev_task_ratio))
        random.shuffle(train_tasks)
        meta_dev_tasks = set(train_tasks[:num_meta_dev])
        actual_train_tasks = set(train_tasks[num_meta_dev:])
        
        print(f"   训练任务: {len(actual_train_tasks)} 个")
        print(f"   Meta-dev任务: {len(meta_dev_tasks)} 个 - {sorted(meta_dev_tasks)}")
        
        # 🐛 FIX: 创建meta-dev数据集 (使用dev split而非train split)
        meta_dev_data = {k: v for k, v in data.items() if k in meta_dev_tasks}
        # 修复: 使用dev split (更接近test分布，更能反映真实泛化能力)
        split_name = "valid" if args.subtask == "clues_syn" else "dev"
        meta_dev_dataset = CLUES_Dataset(meta_dev_data, tokenizer, split_name, args.data_variance)
        meta_dev_dataloader = DataLoader(
            meta_dev_dataset, collate_fn=CLUES_collate,
            batch_sampler=CLUES_BatchSampler(
                meta_dev_dataset, args.per_device_eval_batch_size, False, False,
                args.batch_per_epoch
            ))
        
        # 更新训练数据集 (只保留未被留出的任务)
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
                # 🔥 统计驱动的DAG重估（支持字段级+融合）
                # 1. 生成统计DAG
                dag_json = build_stats_driven_dag(
                    task_name, task, tokenizer, bert, device, 
                    edge_keep_ratio=1.0-(1.0-args.dag_keep_ratio),
                    use_full_formula=args.dag_use_full_formula,
                    use_field_level=args.use_field_level,
                    field_weight_mode=args.field_weight_mode
                )
                
                # 2. DAG融合（如果启用）
                if args.dag_fusion_mode != 'none' and args.manual_dag_file is not None:
                    from models.causal_graph.dag_fusion import DAGFusion, load_explain_dag
                    
                    # 加载解释性DAG
                    explain_dag = load_explain_dag(task_name, args.manual_dag_file)
                    
                    if explain_dag is not None:
                        # 执行融合
                        fusion = DAGFusion(
                            fusion_mode=args.dag_fusion_mode,
                            alpha=args.dag_fusion_alpha
                        )
                        dag_json = fusion.fuse_dags(explain_dag, dag_json, task_name)
                        print(f"✅ 任务 {task_name}: DAG融合完成 (模式={args.dag_fusion_mode}, α={args.dag_fusion_alpha})")
                    else:
                        print(f"⚠️  任务 {task_name}: 未找到解释性DAG,使用纯统计DAG")
                # 转为CausalDAG
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
            # 若 DAG 未携带 labels，则从数据的 target_key 回填，便于零样本原型读出
            if getattr(dag, 'labels', None) is None or len(dag.labels) == 0:
                dag.labels = list(task.get('target_key', []))
            
            dag_per_task[task_name] = dag
        
        # 配置表示增强模块
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
        
        # 根据是否使用改进模型选择不同的包装器
        if args.use_improved_model:
            model = ImprovedCausalGNNClues(
                bert, tokenizer, dag_per_task,
                hidden_dim=256, num_heads=4, num_layers=2,
                label_details_path=args.label_details_path,
                use_token_sim=args.use_token_sim,
                use_bert_token_sim=args.use_bert_token_sim,
                use_enhancement=args.use_enhancement,
                enhancement_config=enhancement_config,
                # 新增改进模块参数
                use_hybrid_encoder=args.use_hybrid_encoder,
                use_cross_attention=args.use_cross_attention,
                use_ordinal_loss=args.use_ordinal_loss,
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
        # 配置零样本读出参数到模型属性，forward 中将读取
        setattr(model, 'zs_readout', args.zs_readout)
        setattr(model, 'zs_topk', args.zs_topk)
        setattr(model, 'zs_tau', args.zs_tau)
        # 任务自适应与直方图配置
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
        # 任务类型与门控/温标
        if args.task_type_json and os.path.exists(args.task_type_json):
            with open(args.task_type_json, 'r') as f:
                setattr(model, 'task_type_map', json.load(f))
        setattr(model, 'gating_g0', args.gating_g0)
        setattr(model, 'gating_gmin', args.gating_gmin)
        setattr(model, 'topbeta_ratio', args.topbeta_ratio)
        setattr(model, 'sim_temp_tok', args.sim_temp_tok)
        setattr(model, 'sim_temp_cls', args.sim_temp_cls)
        # 两阶段日程增量
        setattr(model, 'phase1_delta_topk', args.phase1_delta_topk)
        setattr(model, 'phase1_delta_tau', args.phase1_delta_tau)
        setattr(model, 'phase1_delta_mix', args.phase1_delta_mix)
        setattr(model, 'gnn_subgraph_topk', args.gnn_subgraph_topk)
        
        # 🔥 融合框架模块
        # DAG正则化模块
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
        
        # 因果GNN正则化模块
        causal_gnn_regularizer = None
        if args.use_causal_gnn_reg:
            from models.causal_graph.dag_regularization import CausalGNNRegularizer
            causal_gnn_regularizer = CausalGNNRegularizer(
                lambda_intervene=args.lambda_intervene,
                lambda_counterfactual=args.lambda_counterfactual
            ).to(device)
            print(f"✅ 启用因果GNN正则 (干预={args.lambda_intervene}, 反事实={args.lambda_counterfactual})")
        
        # 解释一致性对齐模块
        guideline_alignment = None
        if args.use_guideline_alignment:
            from models.causal_graph.guideline_alignment import GuidelineAlignment
            guideline_alignment = GuidelineAlignment(
                alignment_mode=args.alignment_mode,
                constraint_weight=args.alignment_weight,
                temperature=args.alignment_temperature
            ).to(device)
            print(f"✅ 启用解释一致性对齐 (模式={args.alignment_mode}, 权重={args.alignment_weight})")
        
        # 将正则化模块设置为模型属性，方便训练时使用
        setattr(model, 'dag_regularizer', dag_regularizer)
        setattr(model, 'causal_gnn_regularizer', causal_gnn_regularizer)
        setattr(model, 'guideline_alignment', guideline_alignment)
        
    # 编码器冻结/半冻结策略
    if args.freeze_encoder:
        print(f"\n❄️  完全冻结BERT编码器,只训练GNN和读出层")
        # 冻结encoder的所有参数
        for name, param in model.named_parameters():
            if 'encoder' in name or 'bert' in name.lower():
                param.requires_grad = False
        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = {
            "AdamW": AdamW,
            "Adam": Adam,
            "SGD": SGD
        }[args.optimizer](optimizer_params, args.learning_rate,
                          weight_decay=args.weight_decay)
    elif args.encoder_lr_ratio < 1.0:
        print(f"\n🔧 编码器学习率降至主干的 {args.encoder_lr_ratio:.1%} (半冻结策略)")
        # 参数分组: encoder用较小学习率
        encoder_params = []
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'encoder' in name or 'bert' in name.lower():
                    encoder_params.append(param)
                else:
                    other_params.append(param)
        
        optimizer_params = [
            {'params': encoder_params, 'lr': args.learning_rate * args.encoder_lr_ratio},
            {'params': other_params, 'lr': args.learning_rate}
        ]
        print(f"   编码器参数: {sum(p.numel() for p in encoder_params):,} 个, lr={args.learning_rate * args.encoder_lr_ratio:.2e}")
        print(f"   其他参数: {sum(p.numel() for p in other_params):,} 个, lr={args.learning_rate:.2e}")
        
        optimizer = {
            "AdamW": AdamW,
            "Adam": Adam,
            "SGD": SGD
        }[args.optimizer](optimizer_params, weight_decay=args.weight_decay)
    else:
        # 正常训练,所有参数同等学习率
        optimizer = {
            "AdamW": AdamW,
            "Adam": Adam,
            "SGD": SGD
        }[args.optimizer](model.parameters(), args.learning_rate,
                        weight_decay=args.weight_decay)
    warmup_steps = 0.1 * args.num_train_epochs * len(train_dataloader)
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

    best_dev_acc = -math.inf
    best_epoch = -1
    best_epoch_dev_loss = math.inf
    best_epoch_test_acc = -math.inf
    best_epoch_test_loss = math.inf
    # test_loss, test_acc = one_epoch(
    #     test_dataloader, model, optimizer, scheduler, tokenizer, device,
    #     "test", start_epoch-1, args.individual_scores)
    for i_epoch in tqdm(range(start_epoch, args.num_train_epochs)):
        log_file.write("="*64 + "\n")
        log_file.flush()
        print("="*64)
        # 中文进度提示,与期望日志一致
        epoch_msg = f"开始 epoch {i_epoch+1}/{args.num_train_epochs}"
        print(epoch_msg)
        log_file.write(epoch_msg + "\n")
        log_file.flush()
        
        # 将训练进度传给模型（用于门控与两阶段日程）
        if hasattr(model, 'set_progress'):
            try:
                model.set_progress((i_epoch + 1) / max(1, args.num_train_epochs))
            except Exception:
                pass
        one_epoch(train_dataloader, model, optimizer, scheduler, tokenizer,
                  device, "train", i_epoch, False, attn_ent_weight=args.attn_ent_weight, args=args, log_file=log_file)
        
        # 评估: 使用多次运行取均值 (如果启用)
        if args.use_meta_dev and meta_dev_dataloader is not None:
            # 使用meta-dev做早停
            meta_dev_msg = "   📊 使用Meta-dev做早停评估 (模拟未见任务)"
            print(meta_dev_msg)
            log_file.write(meta_dev_msg + "\n")
            log_file.flush()
            meta_dev_result = evaluate_with_multiple_runs(
                meta_dev_dataloader, model, tokenizer, device,
                "meta-dev", i_epoch, args.individual_scores, args=args, log_file=log_file)
            # meta-dev的结果用于早停
            dev_loss, dev_acc = meta_dev_result['loss'], meta_dev_result['acc']
            
            # 常规dev和test仍然评估,但不用于早停
            regular_dev_result = evaluate_with_multiple_runs(
                dev_dataloader, model, tokenizer, device,
                "dev", i_epoch, args.individual_scores, args=args, log_file=log_file)
            test_result = evaluate_with_multiple_runs(
                test_dataloader, model, tokenizer, device,
                "test", i_epoch, args.individual_scores, args=args, log_file=log_file)
            test_loss, test_acc = test_result['loss'], test_result['acc']
        else:
            # 正常评估: dev用于早停
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
        
        # 🔥 修复: 零样本场景下使用test acc做early stopping
        if args.zero_shot and args.use_test_early_stop:
            # 零样本场景: 用test acc做early stopping
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
            # 正常场景: 用dev acc做early stopping
            if dev_acc > best_dev_acc:
                best_model_file = f"{args.output_dir}/best_model.pt"
                early_stop_source = "meta-dev" if (args.use_meta_dev and meta_dev_dataloader is not None) else "dev"
                print(f"更新最佳模型 ({early_stop_source} acc: {dev_acc:.4f})")
                save_model(model, optimizer, scheduler, i_epoch, best_model_file)
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
        
        # 温度标定: 训练后对每任务做温度标定
        if args.enable_temperature_calibration and args.zero_shot:
            print("\n" + "="*64)
            print("🌡️  开始任务级温度标定 (基于meta-dev或dev)")
            
            # 加载最佳模型
            best_model_file = f"{args.output_dir}/best_model.pt"
            if os.path.exists(best_model_file):
                checkpoint = torch.load(best_model_file, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"   已加载最佳模型 (epoch {best_epoch})")
            
            # 选择标定数据集
            calib_dataloader = meta_dev_dataloader if (args.use_meta_dev and meta_dev_dataloader is not None) else dev_dataloader
            calib_name = "meta-dev" if (args.use_meta_dev and meta_dev_dataloader is not None) else "dev"
            print(f"   使用 {calib_name} 数据集进行标定")
            
            # 收集logits和labels
            from models.causal_gnn.temperature_calibration import (
                TaskWiseTemperatureCalibration, collect_logits_and_labels
            )
            
            print("   收集logits和labels...")
            task_logits_dict, task_labels_dict = collect_logits_and_labels(
                model, calib_dataloader, device, tokenizer
            )
            
            # 执行标定
            print(f"   对 {len(task_logits_dict)} 个任务进行温度标定:")
            calibrator = TaskWiseTemperatureCalibration()
            calibrator.calibrate(task_logits_dict, task_labels_dict, verbose=True)
            
            # 保存标定参数
            temp_file = f"{args.output_dir}/task_temperatures.json"
            calibrator.save(temp_file)
            
            print(f"\n   ✅ 温度标定完成! 可在推理时使用 --load_temperature {temp_file}")
    else:
        test_result = one_epoch(
            test_dataloader, model, optimizer, scheduler, tokenizer, device,
            "test", start_epoch, args.individual_scores, args=args, log_file=log_file)
        test_loss, test_acc = test_result['loss'], test_result['acc']
    
    # 关闭日志文件
    log_file.close()
    print(f"✅ 日志已保存到: {log_file_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
