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
    def __init__(self, graph_data: dict, prob_data: Optional[dict] = None, device: Optional[torch.device] = None, logfile_path=None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.logfile_path = logfile_path

        # Узлы
        self.node_ids = [n["id"] for n in graph_data["nodes"] if n.get("label","") != 'group']
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.idx_to_id = {i: nid for nid, i in self.id_to_idx.items()}
        self.node_meta = {
            self.id_to_idx[n["id"]]: {"name": n["name"], "label": n.get("label","")}
            for n in graph_data["nodes"] if n.get("label","") != 'group'
        }

        self.n_nodes = len(self.node_ids)

        # Родители / дети
        parent_map = {nid: [] for nid in self.node_ids}
        child_map = {nid: [] for nid in self.node_ids}
        for link in graph_data["links"]:
            src, tgt = link["source"], link["target"]
            if src in self.id_to_idx and tgt in self.id_to_idx:
                parent_map[tgt].append(src)
                child_map[src].append(tgt)

        # parents_idx: список списков
        self.parents_idx = [
            [self.id_to_idx[p] for p in parent_map[nid]] for nid in self.node_ids
        ]
        self.children_idx = [
            [self.id_to_idx[c] for c in child_map[nid]] for nid in self.node_ids
        ]

        # CPT и конфигурации
        self.max_exact_parents = 10
        self.cpt_parents_count = [len(p) for p in self.parents_idx]
        self.configs_per_node = [1 if k == 0 else (1 << k) if k <= self.max_exact_parents else 1
                                 for k in self.cpt_parents_count]
        self.configs_matrix = [
            _configs_matrix(k, self.device) if 0 < k <= self.max_exact_parents else
            torch.empty((1, 0), device=self.device) if k == 0 else
            None
            for k in self.cpt_parents_count
        ]

        # topological sort
        self.sorted_idx = self._topological_sort()

        # Инициализация параметров
        self.node_to_param_idx = {}
        self.logit_params = nn.ParameterList()
        param_idx = 0

        for i in range(self.n_nodes):
            k = self.cpt_parents_count[i]
            n_conf = self.configs_per_node[i]
            node_id = self.idx_to_id[i]

            # Узел с > max_exact_parents — аппроксимация
            if k > self.max_exact_parents:
                self.node_to_param_idx[i] = None
                continue

            # Корневой узел
            if k == 0:
                logits = torch.zeros(1, device=self.device)

            # Узел с точным CPT
            else:
                if prob_data and node_id in prob_data:
                    cpt = prob_data[node_id]
                    if len(cpt) != n_conf:
                        raise ValueError(f"CPT size mismatch for node {node_id}: {len(cpt)} vs {n_conf}")

                    probs = []
                    for conf_int in range(n_conf):
                        key = ",".join(str((conf_int >> (k - 1 - b)) & 1) for b in range(k))
                        probs.append(float(cpt[key]))
                    probs_t = torch.tensor(probs, device=self.device).clamp(1e-6, 1 - 1e-6)
                    logits = torch.log(probs_t / (1 - probs_t))
                else:
                    init_p = torch.rand((n_conf,), device=self.device) * 0.8 + 0.1
                    logits = torch.log(init_p / (1 - init_p))

            self.logit_params.append(nn.Parameter(logits))
            self.node_to_param_idx[i] = param_idx
            param_idx += 1

        # Логирование
        if prob_data:
            log_to_file(self.logfile_path, f"Начальные веса загружены с устройства {self.device}")
        else:
            log_to_file(self.logfile_path, f"Начальные веса сгенерированы на устройстве {self.device}")

        self.to(self.device)

        # Проверка
        for i in range(self.n_nodes):
            if self.node_to_param_idx[i] is not None:
                p = self.logit_params[self.node_to_param_idx[i]]
                assert p.numel() == self.configs_per_node[i], (
                    f"Init mismatch node={self.idx_to_id[i]} p={p.numel()} conf={self.configs_per_node[i]}"
                )

        # Отладка
        for i in range(self.n_nodes):
            print(f"INIT node={self.idx_to_id[i]} k={self.cpt_parents_count[i]} n_conf={self.configs_per_node[i]}")



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

    def forward(self, evidence_tensor: torch.Tensor):
        """
        evidence_tensor: shape = (B, n_nodes), values: 0/1 или -1 for unknown
        Returns:
            marginals: shape = (B, n_nodes), values in [0,1]
        """
        B = evidence_tensor.size(0)
        marginals = torch.zeros(B, self.n_nodes, device=self.device)

        for i in self.sorted_idx:
            k = self.cpt_parents_count[i]
            n_conf = self.configs_per_node[i]
            node_id = self.idx_to_id[i]

            # Индексы родителей
            parents = self.parents_idx[i]

            # Корневой узел или аппроксимация
            if k == 0 or self.node_to_param_idx[i] is None:
                logit_param = self.logit_params[self.node_to_param_idx[i]] if self.node_to_param_idx[i] is not None else torch.zeros(1, device=self.device)
                prob = torch.sigmoid(logit_param)

                # Expand по батчу
                if prob.numel() == 1:
                    marginals[:, i] = prob.expand(B)
                else:
                    marginals[:, i] = prob.unsqueeze(0).repeat(B, 1)[:, 0]

                continue

            # Узел с родителями и точный CPT
            # Берем значения родителей из батча
            if len(parents) > 0:
                parent_vals = evidence_tensor[:, parents]  # shape = (B, k)
            else:
                parent_vals = torch.zeros(B, 0, device=self.device, dtype=torch.long)

            # Определяем конфигурацию CPT для каждого элемента батча
            conf_indices = torch.zeros(B, dtype=torch.long, device=self.device)
            for idx, p in enumerate(parents):
                conf_indices += parent_vals[:, idx].long() << (k - 1 - idx)

            # Получаем logit для каждой конфигурации
            logit_param = self.logit_params[self.node_to_param_idx[i]]  # shape = (n_conf,)
            logit_param = logit_param.unsqueeze(0).expand(B, n_conf)  # shape = (B, n_conf)

            # Берем нужный конфигурационный логит
            batch_logits = logit_param[torch.arange(B, device=self.device), conf_indices]
            marginals[:, i] = torch.sigmoid(batch_logits)

        return marginals



    def export_prob_data(self) -> Dict[str, Dict[str, float]]:
        prob_data = {}

        for i in range(self.n_nodes):
            node_id = self.idx_to_id[i]
            k = self.cpt_parents_count[i]
            n_conf = self.configs_per_node[i]
            param_idx = self.node_to_param_idx[i]

            # 🔴 Узлы с аппроксимацией
            if param_idx is None:
                prob_data[node_id] = {
                    "approximation": "independent_parents"
                }
                continue

            logits = self.logit_params[param_idx]
            p = torch.sigmoid(logits).detach().cpu().tolist()

            # 🧨 ЖЁСТКАЯ ПРОВЕРКА (если снова всплывёт — сразу видно где)
            if len(p) != n_conf:
                raise RuntimeError(
                    f"EXPORT ERROR node={node_id}: "
                    f"len(p)={len(p)} n_conf={n_conf} k={k}"
                )

            mapping = {}
            for conf_int in range(n_conf):
                if n_conf == 1:
                    key = ""
                else:
                    key = ",".join(
                        str((conf_int >> (k - 1 - b)) & 1)
                        for b in range(k)
                    )
                mapping[key] = round(float(p[conf_int]), 6)

            prob_data[node_id] = mapping

        return prob_data



class BayesianTrainer:
    def __init__(self, graph_path: str, orlov_path: str, prob_data = None, logfile_path: str = "opt_calc.txt", device: Optional[torch.device] = None):
        self.logfile_path = logfile_path
        self.graph_json = load_json(logfile_path, graph_path)
        self.orlov_raw = load_json(logfile_path, orlov_path)
        if prob_data:
            self.prob_data = load_json(logfile_path, prob_data)
        else: 
            self.prob_data = None

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        log_to_file(self.logfile_path, "", init=True)
        log_to_file(self.logfile_path, f"Training on device: {self.device}")

        self.model = BayesianTorchModel(self.graph_json, prob_data=self.prob_data, device=self.device, logfile_path = logfile_path)
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
        batch_size = 64


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
                    raw_sum = torch.sum(probs)
                    target = (raw_sum / target_scale).clamp(0.0, 1.0)
                    target_dict[se_idx] = target.item()

                    # print("probs:", probs, "raw_sum", raw_sum, "target", target)

                if not target_dict:
                    continue

                sorted_se_indices = sorted(target_dict.keys())
                targets = torch.tensor([target_dict[se_idx] for se_idx in sorted_se_indices], device=self.device)
                preds = marginals_batch[bi, sorted_se_indices]

                combo_size = len(combo)
                weight = 1
                # Вес для одиночных препаратов
                # weight = 1.0 / math.sqrt(combo_size)
                # if combo_size == 1:
                #     weight *= singleton_weight

                mse = torch.sum((preds - targets) ** 2)
                if mse.item() > 0:
                    losses.append(mse * weight)
                    # combination_weights.append(weight)

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
            max_drug_in_comb: int = 10
        ):
        self.model.to(self.device)
        log_to_file(self.logfile_path, "", init=True)
        log_to_file(self.logfile_path, f"Training on device: {self.device}")
        if max_combinations is not None:
            log_to_file(self.logfile_path, f"Using max {max_combinations} combinations, random selection: {random_combinations}")

        if optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "radam":
            from torch_optimizer import RAdam
            optimizer = RAdam(self.model.parameters(), lr=lr, weight_decay=weight_decay/3)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler_name.lower() == "cosineannealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name.lower() == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = None

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
                    target_scale=3,
                    singleton_weight=1
                )
                if loss.requires_grad:
                    loss_list.append(loss)

            if not loss_list:
                total_loss = torch.tensor(0.0, device=self.device)
            else:
                total_loss = torch.stack(loss_list).sum()

            
            
            # loss_max = self._compute_loss(
            #     max_combinations=max_combinations,          # ограничить комбинации, если много
            #     random_combinations=random_combinations,    # перемешать комбинации
            #     max_drugs_per_combo=10-current_max_drugs,   # максимальное количество препаратов в комбинации
            #     min_drugs_per_combo=10-current_max_drugs,   # минимальное количество препаратов в комбинации
            #     target_scale=3.0,                           # уменьшаем верхнюю границу целевых значений
            #     singleton_weight=3.0                        # усиливаем вклад одиночных препаратов
            # )

            # loss_min = self._compute_loss(
            #     max_combinations=max_combinations,          # ограничить комбинации, если много
            #     random_combinations=random_combinations,    # перемешать комбинации
            #     max_drugs_per_combo=current_max_drugs,      # максимальное количество препаратов в комбинации
            #     min_drugs_per_combo=current_max_drugs,      # минимальное количество препаратов в комбинации
            #     target_scale=3.0,                           # уменьшаем верхнюю границу целевых значений
            #     singleton_weight=3.0                        # усиливаем вклад одиночных препаратов
            # )


            # l2_loss = torch.sum(torch.stack([torch.sum(p ** 2) for p in self.model.logit_params]))
            # total_loss = loss_max + loss_min + weight_decay * l2_loss

            total_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)


            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                current_lr = optimizer.param_groups[0]['lr']
                # log_msg = f"Epoch {epoch}/{epochs} Loss: {total_loss.item():.8f} Raw Loss_min: {loss_min.item():.8f} Raw Loss_max: {loss_max.item():.8f} LR: {current_lr:.6f}"
                log_msg = f"Epoch {epoch}/{epochs} Loss: {total_loss.item():.8f} LR: {current_lr:.6f}"

                # log_msg += f" MaxDrugs: {current_max_drugs}"
                log_to_file(self.logfile_path, log_msg)
                
                prob_data = self.model.export_prob_data()
                os.makedirs(os.path.dirname(intermediate_file) or ".", exist_ok=True)
                self.save_probabilities(prob_data, intermediate_file)

                if scheduler:
                    scheduler.step()

            optimizer.step()

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
    graph_path = f"{DIR}data/graphs/graphs_10_6.json"
    orlov_path = f"{DIR}data/Orlov.json"
    logfile = f"{DIR}data/opt_calc_10_6.txt"
    # prob_data = None
    # prob_data = f"{DIR}data/intermediate_probabilities_opt.json"
    prob_data = f"{DIR}data/probabilities_opt_10_6_pytorch_del_3_comb_10.json"
    intermediate_file = f"{DIR}data/intermediate_probabilities_opt.json"
    trainer = BayesianTrainer(graph_path, orlov_path, prob_data=None, logfile_path=logfile,  device='cpu')
    final_probs, final_loss = trainer.train_model(
        epochs=500,
        lr=1e-1,
        log_interval=10,
        intermediate_file=intermediate_file,
        final_file=f"{DIR}data/probabilities_opt_10_6_pytorch_del_3_comb_10_range.json",
        optimizer_name="AdamW",
        # optimizer_name="Adam",
        scheduler_name="exponential",
        weight_decay=1e-5,
        grad_clip=1,
        max_combinations=2000,
        max_drug_in_comb=10
    )

    print("Final loss:", final_loss)
