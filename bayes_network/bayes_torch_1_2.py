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
    def __init__(self, graph_data: dict, prob_data: Optional[dict] = None, device: Optional[torch.device] = None, logfile_path = None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.logfile_path = logfile_path

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
            if self.id_to_idx.get(link["target"], None) and self.id_to_idx.get(link["source"], None):
                parent_map[link["target"]].append(link["source"])
                child_map[link["source"]].append(link["target"])

        # print("self.node_meta:", self.node_meta)

        self.parents_idx = {self.id_to_idx[nid]: [self.id_to_idx[p] for p in parent_map[nid]] for nid in self.node_ids}
        self.children_idx = {self.id_to_idx[nid]: [self.id_to_idx[c] for c in child_map[nid]] for nid in self.node_ids}

        self.n_nodes = len(self.node_ids)
        self.sorted_idx = self._topological_sort()

        self.cpt_parents_count = [len(self.parents_idx[i]) for i in range(self.n_nodes)]
        self.configs_per_node = [1 << k if k > 0 else 1 for k in self.cpt_parents_count]
        self.configs_matrix = [_configs_matrix(k, self.device) if k > 0 else torch.empty((1,0), device=self.device) for k in self.cpt_parents_count]

        if prob_data:
            log_to_file(self.logfile_path, f"Начальные веса загружены из файла: {self.device}")
        else:
            log_to_file(self.logfile_path, f"Начальные веса сгенерированы: {self.device}")

        self.logit_params = nn.ParameterList()
        for i in range(self.n_nodes):
            n_conf = self.configs_per_node[i]
            if prob_data and self.idx_to_id[i] in prob_data:
                cpt = prob_data[self.idx_to_id[i]]
                probs = [float(cpt.get(",".join(map(str, [(conf_int >> (self.cpt_parents_count[i]-1-b))&1 for b in range(self.cpt_parents_count[i])])), 0.5)) if n_conf>1 else float(cpt.get("",0.5)) for conf_int in range(n_conf)]
                probs_t = torch.tensor(probs, dtype=torch.float32, device=self.device)
                logits = torch.log(probs_t.clamp(1e-6,1-1e-6)/(1-probs_t.clamp(1e-6,1-1e-6)))
            else:
                init_p = torch.rand(n_conf, device=self.device)*0.8 + 0.1
                logits = torch.log(init_p/(1-init_p))
            self.logit_params.append(nn.Parameter(logits))

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
        evidence = evidence or {}
        log_marginals = torch.zeros(self.n_nodes, device=self.device, dtype=torch.float32)
        eps = 1e-6
        min_val = torch.log(torch.tensor(eps, device=self.device))
        max_val = torch.log(torch.tensor(1.0 - eps, device=self.device))

        for idx in self.sorted_idx:
            if idx in evidence:
                val = evidence[idx]
                log_marginals[idx] = torch.log(torch.tensor(val, device=self.device) + eps)
                continue

            k = self.cpt_parents_count[idx]
            logit = self.logit_params[idx]

            log_p_cpt = F.logsigmoid(logit)
            if k == 0:
                log_marginals[idx] = log_p_cpt
                continue

            parent_indices = self.parents_idx[idx]
            parent_log_probs = log_marginals[parent_indices].unsqueeze(0)
            parent_log_probs = parent_log_probs.clamp(min=min_val, max=max_val)

            configs = self.configs_matrix[idx]

            log1m_parent = torch.log1p(-torch.exp(parent_log_probs).clamp(max=1-eps))
            log_probs_given_config = configs * parent_log_probs + (1 - configs) * log1m_parent
            log_prod = torch.sum(log_probs_given_config, dim=1)

            log_marginal = torch.logsumexp(log_prod + log_p_cpt, dim=0)
            log_marginals[idx] = log_marginal

        return torch.exp(log_marginals)

    def export_prob_data(self) -> Dict[str, Dict[str, float]]:
        prob_data = {}
        for i in range(self.n_nodes):
            p = torch.sigmoid(self.logit_params[i]).detach().cpu().numpy().tolist()
            n_conf = self.configs_per_node[i]
            mapping = {}
            for conf_int in range(n_conf):
                key = "" if n_conf == 1 else ",".join(str((conf_int >> (self.cpt_parents_count[i]-1-b))&1) for b in range(self.cpt_parents_count[i]))
                mapping[key] = round(float(p[conf_int]), 6)
            prob_data[self.idx_to_id[i]] = mapping
        return prob_data




class BayesianTrainer:
    def __init__(self, graph_path: str, orlov_path: str, prob_data = None, logfile_path: str = "opt_calc.txt", device: Optional[torch.device] = None):
        self.logfile_path = logfile_path
        self.graph_json = load_json(logfile_path, graph_path)
        self.orlov_raw = load_json(logfile_path, orlov_path)
        if prob_data:
            self.prob_data = load_json(logfile_path, prob_data)
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

    def _process_combo(self, combo: List[int]) -> torch.Tensor:
        # Создаем evidence для текущей комбинации
        evidence = {idx: 1.0 if idx in combo else 0.0 for idx in self.drug_indices}
        marginals = self.model(evidence)

        # Формируем целевой вектор как сумму вероятностей побочек от препаратов в комбинации
        target_dict = {}
        for drug_idx in combo:
            for se_idx, se_prob in self.orlov[drug_idx]:
                target_dict[se_idx] = target_dict.get(se_idx, 0.0) + se_prob

        # Ограничиваем суммарные вероятности до 1.0
        for se_idx in target_dict:
            target_dict[se_idx] = min(target_dict[se_idx], 1.0)

        if not target_dict:
            return torch.tensor(0.0, device=self.device)

        # Сортируем индексы для стабильного порядка
        sorted_se_indices = sorted(target_dict.keys())
        targets = torch.tensor([target_dict[se_idx] for se_idx in sorted_se_indices], device=self.device)
        preds = marginals[sorted_se_indices]

        # Вычисляем MSE между предсказанными и целевыми вероятностями
        loss = torch.sum((preds - targets) ** 2)
        return loss

    def _compute_loss(self, max_combinations: Optional[int] = None, random_combinations: bool = True) -> torch.Tensor:
        if not self.orlov:
            return torch.tensor(0.0, device=self.device)

        # Создаем все возможные комбинации препаратов
        n_drugs = len(self.drug_indices)
        all_combinations = []
        for i in range(1, 2**n_drugs): # Все непустые подмножества
            combination = []
            for j in range(n_drugs):
                if i & (1 << j):
                    combination.append(self.drug_indices[j])
            all_combinations.append(combination)

        # Если задано максимальное количество комбинаций
        if max_combinations is not None and len(all_combinations) > max_combinations:
            if random_combinations:
                selected_combinations = random.sample(all_combinations, max_combinations)
            else:
                selected_combinations = all_combinations[:max_combinations]
        else:
            selected_combinations = all_combinations

        # Параллельная обработка комбинаций
        losses = []
        # Используем ThreadPoolExecutor для параллелизации
        # Можно настроить max_workers в зависимости от количества ядер процессора
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Отправляем задачи в пул
            future_to_combo = {executor.submit(self._process_combo, combo): combo for combo in selected_combinations}
            
            # Собираем результаты
            for future in as_completed(future_to_combo):
                loss = future.result()
                if loss.item() != 0.0:  # Пропускаем нулевые потери
                    losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=self.device)

        return torch.sum(torch.stack(losses))

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
            random_combinations: bool = True
        ):
        self.model.to(self.device)
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
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        else:
            scheduler = None

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = self._compute_loss(max_combinations=max_combinations, random_combinations=random_combinations)

            l2_loss = torch.sum(torch.stack([torch.sum(p ** 2) for p in self.model.logit_params]))
            total_loss = loss + weight_decay * l2_loss

            total_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            optimizer.step()
            if scheduler:
                scheduler.step()

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                log_to_file(self.logfile_path, f"Epoch {epoch}/{epochs} Loss: {total_loss.item():.8f} Raw Loss: {loss}")
                prob_data = self.model.export_prob_data()
                os.makedirs(os.path.dirname(intermediate_file) or ".", exist_ok=True)
                self.save_probabilities(prob_data, intermediate_file)

        final_prob_data = self.model.export_prob_data()
        os.makedirs(os.path.dirname(final_file) or ".", exist_ok=True)
        self.save_probabilities(final_prob_data, final_file)
        log_to_file(self.logfile_path, f"Training finished. Probabilities saved to {final_file}")
        return final_prob_data, total_loss.item()

    # Аналогично для SGD
    def train_model_sgd(
        self,
        epochs: int = 500,
        lr: float = 1e-3,
        momentum: float = 0.9,
        log_interval: int = 10,
        intermediate_file: str = "intermediate_probabilities_sgd.json",
        final_file: str = "probabilities_sgd.json",
        weight_decay: float = 1e-4,
        grad_clip: Optional[float] = 1.0,
        max_combinations: Optional[int] = None,
        random_combinations: bool = True
    ):
        self.model.to(self.device)
        log_to_file(self.logfile_path, "", init=True)
        log_to_file(self.logfile_path, f"Training (SGD) on device: {self.device}")
        if max_combinations is not None:
            log_to_file(self.logfile_path, f"Using max {max_combinations} combinations, random selection: {random_combinations}")

        optimizer = optim.SGD(self.model.parameters(), lr=lr, nesterov=True, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = self._compute_loss(max_combinations=max_combinations, random_combinations=random_combinations)

            l2_loss = torch.sum(torch.stack([torch.sum(p ** 2) for p in self.model.logit_params]))
            total_loss = loss + weight_decay * l2_loss

            total_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                log_to_file(self.logfile_path, f"Epoch {epoch}/{epochs} Loss: {total_loss.item():.8f} Raw Loss: {loss}")
                prob_data = self.model.export_prob_data()
                os.makedirs(os.path.dirname(intermediate_file) or ".", exist_ok=True)
                self.save_probabilities(prob_data, intermediate_file)

        final_prob_data = self.model.export_prob_data()
        os.makedirs(os.path.dirname(final_file) or ".", exist_ok=True)
        self.save_probabilities(final_prob_data, final_file)
        log_to_file(self.logfile_path, f"SGD training finished. Probabilities saved to {final_file}")
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
    prob_data = f"{DIR}data/intermediate_probabilities_opt.json"
    trainer = BayesianTrainer(graph_path, orlov_path, prob_data=prob_data, logfile_path=logfile,  device='cpu')
    final_probs, final_loss = trainer.train_model(
        epochs=1200,
        lr=4e-2,
        log_interval=10,
        intermediate_file="data/intermediate_probabilities_opt.json",
        final_file="data/probabilities_opt_pytorch_10_5.json",
        # optimizer_name="AdamW",
        optimizer_name="Adam",
        scheduler_name="CosineAnnealing",
        weight_decay=1e-5,
        grad_clip=1,
        max_combinations = None
    )
    # final_probs, final_loss = trainer.train_model_sgd(
    #     epochs=1500,
    #     lr=5e-3,
    #     momentum = 0.95,
    #     log_interval=25,
    #     intermediate_file="data/intermediate_probabilities_opt.json",
    #     final_file="data/probabilities_opt_pytorch_10_5_sgd.json",
    #     weight_decay=1e-4,
    #     grad_clip=1
    # )
    print("Final loss:", final_loss)
    trainer.save_probabilities(final_probs, "data/probabilities_opt_pytorch_10_5_sgd.json")
