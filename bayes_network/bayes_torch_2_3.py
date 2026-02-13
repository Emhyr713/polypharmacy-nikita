# bayes_torch.py
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from itertools import combinations

def log_to_file(filename: str, message: str, add_timestamp: bool = True, init: bool = False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if add_timestamp else ""
    prefix = f"[{timestamp}] " if add_timestamp else ""
    mode = "w" if init else "a"
    with open(filename, mode, encoding="utf-8") as f:
        f.write(f"{prefix}{message}\n")


def load_json(logpath: str, filename: str) -> Any:
    log_to_file(logpath, f"Loading JSON: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    log_to_file(logpath, f"Loaded JSON: {filename}")
    return data


def save_json(obj: Any, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def _configs_matrix(n_parents: int, device: torch.device) -> torch.Tensor:
    if n_parents == 0:
        return torch.empty((0, 0), dtype=torch.uint8, device=device)
    n_conf = 1 << n_parents
    ints = torch.arange(n_conf, device=device, dtype=torch.long)
    bin_mat = ((ints[:, None] >> torch.arange(n_parents - 1, -1, -1, device=device)) & 1).to(torch.float32)
    return bin_mat  # shape (n_conf, n_parents)


class BayesianTorchModel(nn.Module):
    def __init__(self, 
                 graph_data: dict, 
                 prob_data: Optional[dict] = None, 
                 device: Optional[torch.device] = None, 
                 logfile_path: Optional[str] = None,
                 max_parents_for_dependent: int = 5):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.logfile_path = logfile_path
        self.max_parents_for_dependent = max_parents_for_dependent

        self.node_ids = [n["id"] for n in graph_data["nodes"] if n.get("label","") != 'group']
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.idx_to_id = {i: nid for nid, i in self.id_to_idx.items()}
        self.node_meta = {self.id_to_idx[n["id"]]:{
                                                    "name": n["name"],
                                                    "label": n.get("label","")}
                                                    for n in graph_data["nodes"]
                                                    if n.get("label","") != 'group'}

        parent_map = {nid: [] for nid in self.node_ids}
        child_map = {nid: [] for nid in self.node_ids}
        for link in graph_data["links"]:
            if self.id_to_idx.get(link["target"]) is not None and self.id_to_idx.get(link["source"]) is not None:
                parent_map[link["target"]].append(link["source"])
                child_map[link["source"]].append(link["target"])


        self.parents_idx = {self.id_to_idx[nid]: [self.id_to_idx[p] for p in parent_map[nid]] for nid in self.node_ids}
        self.children_idx = {self.id_to_idx[nid]: [self.id_to_idx[c] for c in child_map[nid]] for nid in self.node_ids}


        self.n_nodes = len(self.node_ids)
        self.sorted_idx = self._topological_sort()

        self.cpt_parents_count = [len(self.parents_idx[i]) for i in range(self.n_nodes)]
        
        # Определяем, какие узлы являются зависимыми от совместного распределения родителей
        # Узел считается зависимым, если количество родителей <= max_parents_for_dependent
        self.dependent_node = [cnt <= max_parents_for_dependent if max_parents_for_dependent > 0 else True 
                              for cnt in self.cpt_parents_count]
        
        # Логируем информацию о независимых узлах
        if self.logfile_path:
            independent_nodes = [self.node_meta[i]["name"] for i, is_dep in enumerate(self.dependent_node) 
                                if not is_dep]
            if independent_nodes:
                log_to_file(self.logfile_path, 
                           f"Независимые узлы (вероятность = произведение вероятностей родителей): {', '.join(independent_nodes)}")
        
        self.configs_per_node = [1 << k if k > 0 and self.dependent_node[i] else 0 
                                for i, k in enumerate(self.cpt_parents_count)]
        
        # configs_matrix содержит матрицы конфигураций только для зависимых узлов
        self.configs_matrix = []
        for i in range(self.n_nodes):
            if self.dependent_node[i] and self.cpt_parents_count[i] > 0:
                self.configs_matrix.append(_configs_matrix(self.cpt_parents_count[i], self.device))
            else:
                self.configs_matrix.append(torch.empty((0, 0), device=self.device))

        if prob_data:
            log_to_file(self.logfile_path, f"Начальные веса загружены из файла: {self.device}")
        else:
            log_to_file(self.logfile_path, f"Начальные веса сгенерированы: {self.device}")

        self.logit_params = nn.ParameterList()
        for i in range(self.n_nodes):
            n_conf = self.configs_per_node[i]
            node_label = self.node_meta[i].get("label", "")
            
            # Для независимых узлов с родителями не создаем параметры
            if not self.dependent_node[i] and self.cpt_parents_count[i] > 0:
                # Независимый узел с родителями: не создаем параметры, вероятность будет вычисляться динамически
                self.logit_params.append(None)
                continue
                
            if node_label == "prepare":
                probs_t = torch.tensor([0.0], dtype=torch.float32, device=self.device)
                logits = torch.log(probs_t.clamp(1e-6,1-1e-6)/(1-probs_t.clamp(1e-6,1-1e-6)))
            elif prob_data and self.idx_to_id[i] in prob_data:
                cpt = prob_data[self.idx_to_id[i]]
                if n_conf > 0:
                    probs = [float(cpt.get(",".join(map(str, [(conf_int >> (self.cpt_parents_count[i]-1-b))&1 
                                                               for b in range(self.cpt_parents_count[i])])), 0.5)) 
                            for conf_int in range(n_conf)]
                else:
                    probs = [0.5]  # Для независимых узлов без родителей (корневых)
                probs_t = torch.tensor(probs, dtype=torch.float32, device=self.device)
                logits = torch.log(probs_t.clamp(1e-6,1-1e-6)/(1-probs_t.clamp(1e-6,1-1e-6)))
            else:
                if n_conf > 0:
                    init_p = torch.rand(n_conf, device=self.device)*0.8 + 0.1
                    # init_p = torch.rand(n_conf, device=self.device)*0.35 + 0.15
                    logits = torch.log(init_p/(1-init_p))
                else:
                    # Для независимых узлов без родителей создаем один параметр
                    init_p = torch.rand(1, device=self.device)*0.8 + 0.1
                    # init_p = torch.rand(1, device=self.device)*0.35 + 0.15
                    logits = torch.log(init_p/(1-init_p))
            self.logit_params.append(nn.Parameter(logits) if not isinstance(logits, nn.Parameter) else logits)

        self.to(self.device)

    def _topological_sort(self) -> List[int]:
        indeg = {i: len(self.parents_idx[i]) for i in range(self.n_nodes)}
        children_map = {i: list(self.children_idx[i]) for i in range(self.n_nodes)}
        zero = [i for i in range(self.n_nodes) if indeg[i] == 0]
        res = []
        while zero:
            v = zero.pop(0)
            res.append(v)
            for c in children_map.get(v, []):
                indeg[c] -= 1
                if indeg[c] == 0:
                    zero.append(c)
        if len(res) != self.n_nodes:
            raise ValueError("Graph contains a cycle; topological sort not possible.")
        return res

    def forward(self, evidence: Optional[Dict[int, float]] = None) -> torch.Tensor:
        """
        Поддержка двух режимов:
        - evidence: dict[int, float] -> поведение как ранее, возвращается тензор (n_nodes,)
        - evidence: torch.Tensor shape (B, n_nodes) -> батчевый режим, возвращается тензор (B, n_nodes)
        """
        eps = 1e-6
        min_val = torch.log(torch.tensor(eps, device=self.device))
        max_val = torch.log(torch.tensor(1.0 - eps, device=self.device))

        # Батчевый режим: evidence передаётся как тензор (B, n_nodes)
        if isinstance(evidence, torch.Tensor):
            evidence = evidence.to(self.device)
            if evidence.dim() == 1:
                evidence = evidence.unsqueeze(0)
            B = evidence.shape[0]
            log_marginals = torch.zeros((B, self.n_nodes), device=self.device, dtype=torch.float32)

            for idx in self.sorted_idx:
                k = self.cpt_parents_count[idx]
                
                # Для независимых узлов с родителями вычисляем вероятность как произведение вероятностей родителей
                if not self.dependent_node[idx] and k > 0:
                    parent_indices = self.parents_idx[idx]
                    if len(parent_indices) > 0:
                        # Получаем вероятности родителей (уже в log-пространстве)
                        parent_log_probs = log_marginals[:, parent_indices]  # shape (B, k)
                        # Преобразуем из log-пространства в обычные вероятности, перемножаем и возвращаем в log-пространство
                        parent_probs = torch.exp(parent_log_probs.clamp(min=min_val, max=max_val))
                        prob_product = torch.prod(parent_probs, dim=1)  # shape (B,)
                        log_prob_product = torch.log(prob_product.clamp(min=eps, max=1-eps))
                        log_marginals[:, idx] = log_prob_product
                    else:
                        # Независимый узел без родителей (корневой)
                        if self.logit_params[idx] is not None:
                            logit = self.logit_params[idx]
                            log_p = F.logsigmoid(logit)
                            log_marginals[:, idx] = log_p.expand(B)
                        else:
                            log_marginals[:, idx] = torch.log(torch.tensor(0.5, device=self.device)).expand(B)
                else:
                    # Зависимые узлы или независимые без родителей
                    logit = self.logit_params[idx]
                    log_p_cpt = F.logsigmoid(logit)  # shape (n_conf,)

                    # Если у узла нет родителей
                    if k == 0:
                        log_marginal_default = log_p_cpt.expand(B)  # (B,)
                        log_marginals[:, idx] = log_marginal_default
                    else:
                        parent_indices = self.parents_idx[idx]
                        parent_log_probs = log_marginals[:, parent_indices]  # shape (B, k)
                        parent_log_probs = parent_log_probs.clamp(min=min_val, max=max_val)

                        configs = self.configs_matrix[idx]  # shape (n_conf, k)
                        log1m_parent = torch.log1p(-torch.exp(parent_log_probs).clamp(max=1-eps))

                        # (B, n_conf, k)
                        log_probs_given_config = configs.unsqueeze(0) * parent_log_probs.unsqueeze(1) + (1 - configs.unsqueeze(0)) * log1m_parent.unsqueeze(1)
                        log_prod = torch.sum(log_probs_given_config, dim=2)  # (B, n_conf)

                        log_marginal = torch.logsumexp(log_prod + log_p_cpt.unsqueeze(0), dim=1)  # (B,)
                        log_marginals[:, idx] = log_marginal

                # Обрабатываем явно заданные наблюдения
                mask = evidence[:, idx] >= 0.0
                if mask.any():
                    vals = evidence[:, idx].clamp(0.0, 1.0)
                    log_marginals[mask, idx] = torch.log(vals[mask] + eps)

            return torch.exp(log_marginals)  # shape (B, n_nodes)

    def export_prob_data(self) -> Dict[str, Dict[str, float]]:
        prob_data = {}
        for i in range(self.n_nodes):
            # Для независимых узлов с родителями возвращаем пустой словарь
            if not self.dependent_node[i] and self.cpt_parents_count[i] > 0:
                prob_data[self.idx_to_id[i]] = {}
                continue
                
            if self.logit_params[i] is not None:
                p = torch.sigmoid(self.logit_params[i]).detach().cpu().numpy().tolist()
            else:
                p = [0.5]  # Значение по умолчанию
                
            n_conf = self.configs_per_node[i]
            mapping = {}
            
            # Для независимых узлов без родителей
            if not self.dependent_node[i] and self.cpt_parents_count[i] == 0:
                mapping[""] = round(float(p[0]), 6)
            # Для зависимых узлов
            else:
                for conf_int in range(n_conf):
                    key = "" if n_conf == 1 else ",".join(str((conf_int >> (self.cpt_parents_count[i]-1-b))&1) 
                                                         for b in range(self.cpt_parents_count[i]))
                    mapping[key] = round(float(p[conf_int]), 6)
                    
            prob_data[self.idx_to_id[i]] = mapping
        return prob_data


class BayesianTrainer:
    def __init__(self, 
                 graph_path: str, 
                 orlov_path: str, 
                 prob_data = None, 
                 logfile_path: str = "opt_calc.txt", 
                 device: Optional[torch.device] = None,
                 max_parents_for_dependent: int = 5):
        self.logfile_path = logfile_path
        self.graph_json = load_json(logfile_path, graph_path)
        self.orlov_raw = load_json(logfile_path, orlov_path)
        if prob_data:
            self.prob_data = load_json(logfile_path, prob_data)
        else: 
            self.prob_data = None

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.max_parents_for_dependent = max_parents_for_dependent

        log_to_file(self.logfile_path, "", init=True)
        log_to_file(self.logfile_path, f"Training on device: {self.device}")
        log_to_file(self.logfile_path, f"Порог для независимых узлов: {max_parents_for_dependent}")

        self.model = BayesianTorchModel(self.graph_json, 
                                        prob_data=self.prob_data, 
                                        device=self.device, 
                                        logfile_path=logfile_path,
                                        max_parents_for_dependent=max_parents_for_dependent)
        
        self.name_to_idx = {self.model.node_meta[i]["name"]: i for i in range(self.model.n_nodes)}
        self.orlov: Dict[int, List[Tuple[int, float]]] = {}
        for drug_name, se_list in self.orlov_raw.items():
            if drug_name not in self.name_to_idx:
                continue
            d_idx = self.name_to_idx[drug_name]
            entries = [(self.name_to_idx[se_name], float(p)) for se_name, p in se_list if se_name in self.name_to_idx]
            if entries:
                self.orlov[d_idx] = entries

        self.drug_indices = list(self.orlov.keys())
        self.device = self.model.device
        # Создаем лок для синхронизации логирования
        self.log_lock = threading.Lock()


    def _compute_loss(
        self,
        max_combinations: Optional[int] = 1000,
        random_combinations: bool = True,
        min_drugs_per_combo: int = 1,
        max_drugs_per_combo: Optional[int] = 10,
        target_scale: float = 3.0,
        singleton_weight: float = 3.0
    ) -> torch.Tensor:
        if not self.orlov:
            return torch.tensor(0.0, device=self.device)

        n_drugs = len(self.drug_indices)
        if max_drugs_per_combo is None:
            max_drugs_per_combo = n_drugs

        # Ограничиваем диапазон разумными значениями
        min_r = max(1, min_drugs_per_combo)
        max_r = min(max_drugs_per_combo, n_drugs)

        all_combinations = []
        for r in range(min_r, max_r + 1):
            for combo in combinations(self.drug_indices, r):
                all_combinations.append(list(combo))

        # Если комбинаций слишком много — ограничиваем
        if max_combinations is not None and len(all_combinations) > max_combinations:
            if random_combinations:
                selected_combinations = random.sample(all_combinations, max_combinations)
            else:
                selected_combinations = all_combinations[:max_combinations]
        else:
            selected_combinations = all_combinations

        if not selected_combinations:
            return torch.tensor(0.0, device=self.device)

        losses = []
        combination_weights = []
        batch_size = 2048

        for start in range(0, len(selected_combinations), batch_size):
            batch_combos = selected_combinations[start:start + batch_size]
            B = len(batch_combos)
            evidence_tensor = torch.full((B, self.model.n_nodes), -1.0, device=self.device, dtype=torch.float32)

            for bi, combo in enumerate(batch_combos):
                for d_idx in combo:
                    evidence_tensor[bi, d_idx] = 1.0

            marginals_batch = self.model(evidence_tensor)

            for bi, combo in enumerate(batch_combos):
                target_dict = {}
                for drug_idx in combo:
                    if drug_idx not in self.orlov:
                        continue
                    for se_idx, se_prob in self.orlov[drug_idx]:
                        if se_idx not in target_dict:
                            target_dict[se_idx] = []
                        target_dict[se_idx].append(se_prob)

                # print("target_dict[se_idx]",  target_dict)

                for se_idx in target_dict:
                    probs = torch.tensor(target_dict[se_idx], device=self.device)
                    
                    # Суммирование
                    # raw_sum = torch.sum(probs)
                    # target = (raw_sum / target_scale).clamp(0.0, 1.0)

                    # ИЛИ через логистическу сумму (сигмоиду)
                    target = torch.sigmoid(torch.sum(torch.logit(probs)))

                    target_dict[se_idx] = target.item()

                    # print("probs:", probs, "raw_sum", raw_sum, "target", target)

                if not target_dict:
                    continue

                sorted_se_indices = sorted(target_dict.keys())
                targets = torch.tensor([target_dict[se_idx] for se_idx in sorted_se_indices], device=self.device)
                preds = marginals_batch[bi, sorted_se_indices]

                weight = 1

                # combo_size = len(combo)
                # weight = 1.0 / math.sqrt(combo_size)

                # Вес для одиночных препаратов
                # weight = 1.0 / math.sqrt(combo_size)
                # if combo_size == 1:
                #     weight *= singleton_weight

                mse = torch.sum((preds - targets) ** 2)
                if mse.item() > 0:
                    losses.append(mse * weight)

                # bce = F.binary_cross_entropy(preds, targets, reduction='sum')
                # if bce.item() > 0:
                #     losses.append(bce * weight)

        if not losses:
            return torch.tensor(0.0, device=self.device)

        total_weight = sum(combination_weights)
        weighted_loss = sum(losses) / total_weight if total_weight > 0 else sum(losses)
        return weighted_loss


    # В методе train_model нужно изменить вызов _compute_loss
    def train_model(
            self,
            epochs: int = 500,
            lr: float = 1e-2,
            log_interval: int = 10,
            intermediate_file: str = "intermediate_probabilities_opt.json",
            final_file: str = "probabilities_opt.json",
            optimizer_name: str = "AdamW",
            scheduler_name: str = "CosineAnnealing",
            weight_decay: float = 1e-4,
            grad_clip: Optional[float] = 1.0,
            max_combinations: Optional[int] = None,
            random_combinations: bool = True,
            max_drug_in_comb: int = 10, 
            target_scale: int = 1
        ):
        self.model.to(self.device)
        log_to_file(self.logfile_path, "", init=True)
        log_to_file(self.logfile_path, f"Training on device: {self.device}")
        if max_combinations is not None:
            log_to_file(self.logfile_path, f"Using max {max_combinations} combinations, random selection: {random_combinations}")

        # Создаем оптимизатор только для параметров, которые не None
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "radam":
            from torch_optimizer import RAdam
            optimizer = RAdam(params_to_optimize, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)

        if scheduler_name.lower() == "cosineannealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name.lower() == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_name.lower() == "reducelronplateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        else:
            scheduler = None

        # "Обрезка" вероятностей
        def prob_to_logit(p):
            return math.log(p / (1 - p))

        LOGIT_MIN = prob_to_logit(0.05)   # ≈ -2.944
        LOGIT_MAX = prob_to_logit(0.8)    # = 0.0

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            
            # current_max_drugs = None

            # progress = (epoch - 1) / max(1, epochs - 1)
            # current_max_drugs = max(1, min(max_drug_in_comb, int(1 + progress * (max_drug_in_comb - 1))))

            loss_list = []

            for i_comb in range(1, max_drug_in_comb):
                loss = self._compute_loss(
                    max_combinations=max_combinations,
                    random_combinations=random_combinations,
                    max_drugs_per_combo=i_comb,
                    min_drugs_per_combo=i_comb,
                    target_scale=target_scale,
                    singleton_weight=1
                )
                if loss.requires_grad:
                    loss_list.append(loss)

            if not loss_list:
                total_loss = torch.tensor(0.0, device=self.device)
            else:
                total_loss = torch.stack(loss_list).sum()

            # l2_loss = torch.sum(torch.stack([torch.sum(p ** 2) for p in self.model.logit_params]))
            # total_loss = total_loss + weight_decay * l2_loss

            total_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, grad_clip)

            optimizer.step()

            # with torch.no_grad():
            #     for p in self.model.logit_params:
            #         if p is not None:
            #             p.clamp_(LOGIT_MIN, LOGIT_MAX)

            if scheduler:
                scheduler.step()

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                current_lr = optimizer.param_groups[0]['lr']
                # log_msg = f"Epoch {epoch}/{epochs} Loss: {total_loss.item():.8f} Raw Loss_min: {loss_min.item():.8f} Raw Loss_max: {loss_max.item():.8f} LR: {current_lr:.6f}"
                log_msg = f"Epoch {epoch}/{epochs} Loss: {total_loss.item():.8f} LR: {current_lr:.6f}"

                # log_msg += f" MaxDrugs: {current_max_drugs}"
                log_to_file(self.logfile_path, log_msg)
                
                prob_data = self.model.export_prob_data()
                os.makedirs(os.path.dirname(intermediate_file) or ".", exist_ok=True)
                self.save_probabilities(prob_data, intermediate_file)



            # optimizer.step()

        final_prob_data = self.model.export_prob_data()
        os.makedirs(os.path.dirname(final_file) or ".", exist_ok=True)
        self.save_probabilities(final_prob_data, final_file)
        log_to_file(self.logfile_path, f"Training finished. Probabilities saved to {final_file}")
        return final_prob_data, total_loss.item()
    
    def save_probabilities(self, prob_data, path: str):
        save_json(prob_data, path)


if __name__ == "__main__":
    DIR = "bayes_network/"
    # graph_path = "bayes_network/data/graphs/TEST.json"
    GRAPH_NAME = "10_6"
    graph_path = f"{DIR}data/graphs/graphs_{GRAPH_NAME}.json"
    orlov_path = f"{DIR}data/Orlov.json"
    logfile = f"{DIR}data/opt_calc_{GRAPH_NAME}.txt"
    # prob_data = None
    # prob_data = f"{DIR}data/intermediate_probabilities_opt.json"
    prob_data = f"{DIR}data/probabilities_opt_{GRAPH_NAME}_pytorch_del_3_comb_1.json"
    intermediate_file = f"{DIR}data/intermediate_probabilities_opt.json"
    
    # Устанавливаем порог для независимых узлов
    max_parents_for_dependent = 10  # Можно изменить это значение
    
    trainer = BayesianTrainer(graph_path, orlov_path, prob_data=None, 
                             logfile_path=logfile, device='cpu',
                             max_parents_for_dependent=max_parents_for_dependent)
    
    final_probs, final_loss = trainer.train_model(
        epochs=150,
        lr=0.07,
        log_interval=10,
        intermediate_file=intermediate_file,
        final_file=f"{DIR}data/probabilities_opt_{GRAPH_NAME}_pytorch_2_3_comb_3_v2.json",
        optimizer_name="AdamW",
        # optimizer_name="Adam",
        scheduler_name="cosineannealing",
        weight_decay=1e-3,
        grad_clip=1,
        max_combinations=7500,
        max_drug_in_comb=3,
        random_combinations = True,
        target_scale = 1
    )

    print("Final loss:", final_loss)