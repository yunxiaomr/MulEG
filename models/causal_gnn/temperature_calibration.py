"""
任务级温度标定模块
用于训练后对每个任务进行温度/Platt标定,提升不均衡任务的决策边界稳定性
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling: 对logits进行温度缩放
    
    T>1: 软化分布,降低过度自信
    T<1: 尖化分布,增强确定性
    """
    def __init__(self, init_temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)
    
    def forward(self, logits):
        """
        Args:
            logits: (batch_size, num_classes)
        Returns:
            calibrated_logits: (batch_size, num_classes)
        """
        return logits / self.temperature
    
    def fit(self, logits, labels, max_iter=50, lr=0.01):
        """
        在验证集上优化温度参数
        
        Args:
            logits: (N, num_classes) tensor
            labels: (N,) tensor
            max_iter: 最大迭代次数
            lr: 学习率
        """
        logits = logits.detach()
        labels = labels.detach()
        
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return self.temperature.item()


class TaskWiseTemperatureCalibration:
    """
    任务级温度标定: 为每个任务学习最优温度参数
    """
    def __init__(self):
        self.task_temperatures = {}  # {task_name: temperature}
        self.task_scalers = {}       # {task_name: TemperatureScaler}
    
    def calibrate(self, task_logits_dict, task_labels_dict, verbose=True):
        """
        对每个任务进行温度标定
        
        Args:
            task_logits_dict: {task_name: logits_tensor (N, num_classes)}
            task_labels_dict: {task_name: labels_tensor (N,)}
            verbose: 是否打印详细信息
        """
        for task_name in task_logits_dict:
            if task_name not in task_labels_dict:
                continue
            
            logits = task_logits_dict[task_name]
            labels = task_labels_dict[task_name]
            
            if len(logits) == 0:
                continue
            
            # 为该任务创建温度缩放器
            scaler = TemperatureScaler()
            optimal_temp = scaler.fit(logits, labels)
            
            self.task_temperatures[task_name] = optimal_temp
            self.task_scalers[task_name] = scaler
            
            if verbose:
                # 计算标定前后的准确率和NLL
                with torch.no_grad():
                    before_probs = torch.softmax(logits, dim=-1)
                    after_probs = torch.softmax(scaler(logits), dim=-1)
                    
                    before_acc = (logits.argmax(dim=-1) == labels).float().mean().item()
                    after_acc = (scaler(logits).argmax(dim=-1) == labels).float().mean().item()
                    
                    criterion = nn.CrossEntropyLoss()
                    before_nll = criterion(logits, labels).item()
                    after_nll = criterion(scaler(logits), labels).item()
                
                print(f"   📈 {task_name:30s} | T={optimal_temp:.3f} | "
                      f"Acc: {before_acc:.4f}→{after_acc:.4f} | "
                      f"NLL: {before_nll:.4f}→{after_nll:.4f}")
    
    def apply(self, logits, task_name):
        """
        对指定任务的logits应用温度标定
        
        Args:
            logits: (batch_size, num_classes) tensor
            task_name: 任务名称
        Returns:
            calibrated_logits: (batch_size, num_classes) tensor
        """
        if task_name not in self.task_scalers:
            # 未标定的任务,返回原始logits
            return logits
        
        return self.task_scalers[task_name](logits)
    
    def get_temperature(self, task_name):
        """获取任务的温度参数"""
        return self.task_temperatures.get(task_name, 1.0)
    
    def save(self, filepath):
        """保存温度参数到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.task_temperatures, f, indent=2)
        print(f"✅ 温度标定参数已保存到: {filepath}")
    
    def load(self, filepath):
        """从文件加载温度参数"""
        import json
        with open(filepath, 'r') as f:
            self.task_temperatures = json.load(f)
        
        # 重建scalers
        for task_name, temp in self.task_temperatures.items():
            scaler = TemperatureScaler(init_temperature=temp)
            scaler.temperature.requires_grad = False
            self.task_scalers[task_name] = scaler
        
        print(f"✅ 已加载 {len(self.task_temperatures)} 个任务的温度标定参数")


def collect_logits_and_labels(model, dataloader, device, tokenizer):
    """
    在验证集上收集每个任务的logits和labels,用于温度标定
    
    Returns:
        task_logits_dict: {task_name: logits_tensor}
        task_labels_dict: {task_name: labels_tensor}
    """
    model.eval()
    task_logits = defaultdict(list)
    task_labels = defaultdict(list)
    
    with torch.no_grad():
        for batch in dataloader:
            samples, labels, mask, guideline_inputs = batch
            task_names = [s['task'] for s in samples]
            
            # Forward pass
            outputs = model(
                samples=samples,
                labels=labels,
                mask=mask,
                guideline_inputs=guideline_inputs,
                tokenizer=tokenizer
            )
            
            # 收集每个任务的logits和labels
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']  # (batch_size, num_classes)
            else:
                # 兼容旧接口
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            
            for i, task_name in enumerate(task_names):
                task_logits[task_name].append(logits[i].cpu())
                task_labels[task_name].append(labels[i].cpu())
    
    # 拼接
    task_logits_dict = {
        task: torch.stack(logits_list) if len(logits_list) > 0 else torch.empty(0)
        for task, logits_list in task_logits.items()
    }
    task_labels_dict = {
        task: torch.stack(labels_list) if len(labels_list) > 0 else torch.empty(0)
        for task, labels_list in task_labels.items()
    }
    
    return task_logits_dict, task_labels_dict

