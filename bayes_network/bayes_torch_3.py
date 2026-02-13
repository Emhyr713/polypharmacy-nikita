# bayes_torch.py
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def log_to_file(filename: str, message: str, add_timestamp: bool = True, init: bool = False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if add_timestamp else ""
    prefix = f"[{timestamp}] " if add_timestamp else ""
    mode = "w" if init else "a"
    with open(filename, mode, encoding="utf-8") as f:
        f.write(f"{prefix}{message}\n")


def load_json(filename: str) -> Any:
    log_to_file("opt_calc.txt", f"Loading JSON: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    log_to_file("opt_calc.txt", f"Loaded JSON: {filename}")
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


# В class BayesianTorchModel: сохранить prior_logits и добавить опцию factorized CPT (логистическая аппроксимация)
class BayesianTorchModel(nn.Module):
    def __init__(self, graph_data: dict, prob_data: Optional[dict] = None, device: Optional[torch.device] = None,
                 use_factorized_cpt: bool = False):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.use_factorized_cpt = use_factorized_cpt

        # (оставляем прежнюю индексацию узлов)
        self.node_ids = [n["id"] for n in graph_data["nodes"]]
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.idx_to_id = {i: nid for nid, i in self.id_to_idx.items()}
        self.node_meta = {
            self.id_to_idx[n["id"]]: {
                "name": n["name"],
                "label": n.get("label", "")
            }
            for n in graph_data["nodes"]
        }

        # parent/child maps
        parent_map = {nid: [] for nid in self.node_ids}
        child_map = {nid: [] for nid in self.node_ids}
        for link in graph_data["links"]:
            parent_map[link["target"]].append(link["source"])
            child_map[link["source"]].append(link["target"])

        self.parents_idx = {
            self.id_to_idx[nid]: [self.id_to_idx[p] for p in parent_map[nid]]
            for nid in self.node_ids
        }
        self.children_idx = {
            self.id_to_idx[nid]: [self.id_to_idx[c] for c in child_map[nid]]
            for nid in self.node_ids
        }

        self.n_nodes = len(self.node_ids)
        self.sorted_idx = self._topological_sort()
        self.cpt_parents_count = [len(self.parents_idx[i]) for i in range(self.n_nodes)]
        self.configs_per_node = [1 << k if k > 0 else 1 for k in self.cpt_parents_count]
        self.configs_matrix = [
            _configs_matrix(k, self.device) if k > 0 else torch.empty((1, 0), device=self.device)
            for k in self.cpt_parents_count
        ]

        # Параметры CPT: либо полные таблицы (как раньше), либо факторизованные веса (w0 + sum w_j * parent_j)
        self.logit_params = nn.ParameterList()
        self.factorized_params = nn.ParameterList()  # содержит параметры для факторизованной версии (если включена)
        prior_logits_list = []

        for i in range(self.n_nodes):
            n_conf = self.configs_per_node[i]

            if prob_data and self.idx_to_id[i] in prob_data and not self.use_factorized_cpt:
                cpt = prob_data[self.idx_to_id[i]]
                probs = []
                for conf_int in range(n_conf):
                    if n_conf == 1:
                        key = ""
                    else:
                        bits = [
                            (conf_int >> (self.cpt_parents_count[i] - 1 - b)) & 1
                            for b in range(self.cpt_parents_count[i])
                        ]
                        key = ",".join(map(str, bits))
                    prob_value = float(cpt.get(key, 0.5))
                    probs.append(prob_value)
                probs_t = torch.tensor(probs, dtype=torch.float32, device=self.device)
                logits = torch.log(probs_t.clamp(1e-6, 1 - 1e-6) / (1 - probs_t.clamp(1e-6, 1 - 1e-6)))
                param = nn.Parameter(logits)
                prior_logits_list.append(logits.detach().clone())

            else:
                # инициализация случайным образом, либо факторизованные параметры
                if self.use_factorized_cpt and self.cpt_parents_count[i] > 0:
                    # факторизованная: один bias и one weight per parent
                    w = torch.randn(self.cpt_parents_count[i] + 1, device=self.device) * 0.1
                    param = nn.Parameter(w)
                    # сохраняем placeholder в logit_params для совместимости (нужно при экспортe)
                    self.factorized_params.append(param)
                    prior_logits_list.append(param.detach().clone())  # в prior сохраним веса (не логиты)
                    # в logit_params добавляем нулевой параметр, чтобы индексация не ломалась
                    self.logit_params.append(nn.Parameter(torch.zeros(1, device=self.device)))
                    continue
                else:
                    init_p = torch.rand(n_conf, dtype=torch.float32, device=self.device) * 0.8 + 0.1
                    logits = torch.log(init_p / (1.0 - init_p))
                    param = nn.Parameter(logits)
                    prior_logits_list.append(logits.detach().clone())

            self.logit_params.append(param)

        # сохраняем приоритетные значения для регуляризации
        self.prior_logits = prior_logits_list
        self.to(self.device)

    def forward(self, evidence: Optional[Dict[int, float]] = None, parent_dropout: float = 0.0,
                training: bool = False, config_dropout: float = 0.0, temp: float = 1.0) -> torch.Tensor:
        device = self.device
        evidence = evidence or {}
        marginals = torch.zeros(self.n_nodes, dtype=torch.float32, device=device)
        # если training True, используем стохастику (dropout). Используйте self.train() для переключения состояния.

        for idx in self.sorted_idx:
            if idx in evidence:
                marginals[idx] = torch.tensor(float(evidence[idx]), dtype=torch.float32, device=device)
                continue

            k = self.cpt_parents_count[idx]

            # если факторизованная CPT
            if self.use_factorized_cpt and k > 0 and len(self.factorized_params) > 0:
                # найдём соответствующий параметр по порядку: допускаем, что factorized_params хранится в порядке узлов с k>0
                # Для простоты: сопоставим по idx: здесь нужно хранить mapping; для минимального изменения -- вычислим index
                # В этой короткой вставке предполагается, что order совпадает; для надёжности можно хранить dict node->param при инициализации.
                # Здесь демонстрация формулы:
                # bias + sum_j weight_j * parent_j
                # parent_j в {0..1} (marginals)
                # TODO: в production сохранить сопоставление idx->позиция в factorized_params
                raise NotImplementedError("Factorized CPT requires mapping idx->factorized_param; см. предложения ниже.")
            
            logits = self.logit_params[idx]  # shape (n_conf,)
            # конфигурационные вероятности
            p_cpt = torch.sigmoid(logits / temp)  # temperature позволяет избегать чрезмерной детерминации

            if k == 0:
                marginals[idx] = p_cpt.squeeze()
                continue

            parent_indices = self.parents_idx[idx]
            parent_probs = marginals[parent_indices]  # shape (k,)

            # parent-dropout: при обучении с вероятностью parent_dropout заменяем parent_prob на 0.5 (нейтральная)
            if training and parent_dropout > 0.0:
                mask = torch.rand(len(parent_probs), device=device) < parent_dropout
                if mask.any():
                    parent_probs = parent_probs.clone()
                    parent_probs[mask] = 0.5

            configs = self.configs_matrix[idx]  # (n_conf, k) float (0/1)
            parent_probs_exp = parent_probs.unsqueeze(0)  # (1, k)
            probs_given_config = configs * parent_probs_exp + (1.0 - configs) * (1.0 - parent_probs_exp)  # (n_conf, k)
            prod_probs = torch.prod(probs_given_config, dim=1)  # (n_conf,)

            # config-dropout: во время тренировки можем занулять некоторые конфигурации (устранение зависимостей от отдельных конфигов)
            if training and config_dropout > 0.0:
                conf_mask = (torch.rand_like(prod_probs) >= config_dropout).to(prod_probs.dtype)
                prod_probs = prod_probs * conf_mask
                # избегаем полного зануления: небольшая регуляризация
                prod_probs = prod_probs + 1e-12

            marg = torch.sum(p_cpt * prod_probs)
            marginals[idx] = marg

        return marginals
    
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
    
    def export_prob_data(self) -> Dict[str, Dict[str, float]]:
        prob_data = {}
        for i in range(self.n_nodes):
            p = torch.sigmoid(self.logit_params[i]).detach().cpu().numpy().tolist()
            n_conf = self.configs_per_node[i]
            mapping = {}
            for conf_int in range(n_conf):
                if n_conf == 1:
                    key = ""
                else:
                    bits = [(conf_int >> (self.cpt_parents_count[i] - 1 - b)) & 1 for b in range(self.cpt_parents_count[i])]
                    key = ",".join(map(str, bits))
                mapping[key] = round(float(p[conf_int]), 6)
            prob_data[self.idx_to_id[i]] = mapping
        return prob_data

class BayesianTrainer:
    def __init__(self, graph_path: str, orlov_path: str, logfile_path: str = "opt_calc.txt", device: Optional[torch.device] = None,
                 prior_lambda: float = 1e-3, use_factorized: bool = False):
        self.logfile_path = logfile_path
        self.graph_json = load_json(graph_path)
        self.orlov_raw = load_json(orlov_path)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # создание модели с опцией факторизации CPT
        self.model = BayesianTorchModel(self.graph_json, prob_data=None, device=self.device, use_factorized_cpt=use_factorized)
        self.prior_lambda = prior_lambda

        # mapping name->idx
        self.name_to_idx = {self.model.node_meta[i]["name"]: i for i in range(self.model.n_nodes)}
        # parse Orlov dataset into mapping: drug_idx -> list of (se_idx, target_prob)
        self.orlov: Dict[int, List[Tuple[int, float]]] = {}
        for drug_name, se_list in self.orlov_raw.items():
            if drug_name not in self.name_to_idx:
                continue
            d_idx = self.name_to_idx[drug_name]
            entries = []
            for se_name, p in se_list:
                if se_name not in self.name_to_idx:
                    continue
                se_idx = self.name_to_idx[se_name]
                entries.append((se_idx, float(p)))
            if entries:
                self.orlov[d_idx] = entries

        # precompute list of drug indices for batching / cv
        self.orlov_drugs = list(self.orlov.keys())

    def save_probabilities(self, prob_data, path: str):
        save_json(prob_data, path)

    def loss_for_current(self, marginals_func, train_drug_indices: List[int], training: bool = True,
                         parent_dropout: float = 0.0, config_dropout: float = 0.0, temp: float = 1.0) -> torch.Tensor:
        """
        marginals_func: callable(evidence_dict, training, parent_dropout, config_dropout, temp) -> marginals tensor
        train_drug_indices: iterable of drug indices to use in this batch / fold
        """
        loss = torch.tensor(0.0, device=self.device)
        for drug_idx in train_drug_indices:
            se_entries = self.orlov[drug_idx]
            evidence = {drug_idx: 1.0}
            # используем forward с dropout только при training==True
            marg_with_evidence = marginals_func(evidence, training=training, parent_dropout=parent_dropout,
                                                config_dropout=config_dropout, temp=temp)
            for se_idx, target in se_entries:
                pred = marg_with_evidence[se_idx]
                loss = loss + (pred - target) ** 2

        # априорная L2-регуляризация относительно prior_logits
        if self.model.prior_logits and self.prior_lambda > 0.0:
            reg = torch.tensor(0.0, device=self.device)
            # учитываем только реальные logit_params (если факторизованные параметры используются, их надо регул. отдельно)
            for i, p in enumerate(self.model.logit_params):
                prior = self.model.prior_logits[i].to(self.device)
                # если shapes совпадают
                if p.shape == prior.shape:
                    reg = reg + torch.sum((p - prior) ** 2)
            loss = loss + self.prior_lambda * reg

        return loss

    def train(self,
              epochs: int = 1000,
              lr: float = 1e-2,
              log_interval: int = 10,
              intermediate_file: str = "intermediate_probabilities_opt.json",
              best_res_file: str = "best_res_opt.json",
              weight_decay: float = 1e-4,
              parent_dropout: float = 0.1,
              config_dropout: float = 0.0,
              temp: float = 1.0,
              batch_size: int = 8,
              k_folds: int = 5,
              early_stop_patience: int = 20,
              max_grad_norm: float = 1.0):
        log_to_file(self.logfile_path, "", init=True)
        log_to_file(self.logfile_path, f"Starting training on device: {self.device}", add_timestamp=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Prepare CV folds
        random.seed(42)
        drug_indices = self.orlov_drugs.copy()
        random.shuffle(drug_indices)
        folds = [[] for _ in range(k_folds)]
        for i, d in enumerate(drug_indices):
            folds[i % k_folds].append(d)

        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(1, epochs + 1):
            # for each epoch, iterate folds as train/val to get robust measure
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            self.model.train()  # enable training-mode stochasticity
            for fold_idx in range(k_folds):
                val_fold = folds[fold_idx]
                train_fold = [d for j, f in enumerate(folds) if j != fold_idx for d in f]
                # shuffle training fold for mini-batching
                random.shuffle(train_fold)

                # mini-batches over drugs
                for start in range(0, len(train_fold), batch_size):
                    batch = train_fold[start:start + batch_size]
                    optimizer.zero_grad()
                    # создаём wrapper для forward
                    marg_func = lambda evidence, training, parent_dropout, config_dropout, temp: \
                        self.model.forward(evidence=evidence, parent_dropout=parent_dropout, training=training,
                                           config_dropout=config_dropout, temp=temp)
                    loss = self.loss_for_current(marg_func, batch, training=True,
                                                 parent_dropout=parent_dropout, config_dropout=config_dropout, temp=temp)
                    loss.backward()
                    # градиентный клиппинг
                    clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    epoch_train_loss += loss.item()

                # валидация на валидационном фолде (без dropout)
                self.model.eval()
                with torch.no_grad():
                    marg_func_eval = lambda evidence, training, parent_dropout, config_dropout, temp: \
                        self.model.forward(evidence=evidence, parent_dropout=0.0, training=False,
                                           config_dropout=0.0, temp=temp)
                    val_loss = self.loss_for_current(marg_func_eval, val_fold, training=False,
                                                     parent_dropout=0.0, config_dropout=0.0, temp=temp)
                    epoch_val_loss += val_loss.item()
                self.model.train()

            # усреднённые значения
            epoch_train_loss /= max(1, len(drug_indices))
            epoch_val_loss /= k_folds

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                log_to_file(self.logfile_path, f"Epoch {epoch}/{epochs} TrainLoss: {epoch_train_loss:.8f} ValLoss: {epoch_val_loss:.8f}")
                prob_data = self.model.export_prob_data()
                self.save_probabilities(prob_data, intermediate_file)

            # ранняя остановка по валидации
            if epoch_val_loss < best_val_loss - 1e-8:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                # сохраняем лучшие вероятности
                self.save_probabilities(self.model.export_prob_data(), best_res_file)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    log_to_file(self.logfile_path, f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.8f}")
                    break

        final_prob_data = self.model.export_prob_data()
        log_to_file(self.logfile_path, "Training finished. Final probabilities saved to probabilities_opt.json")

        final_loss_tensor = self.loss_for_current(lambda e, training, parent_dropout, config_dropout, temp:
                                                  self.model.forward(evidence=e, parent_dropout=0.0, training=False,
                                                                     config_dropout=0.0, temp=temp),
                                                  list(self.orlov.keys()), training=False)
        final_loss = final_loss_tensor.detach().cpu().numpy().item()
        return final_prob_data, final_loss


if __name__ == "__main__":
    # graph_path = "bayes_network/data/graphs/TEST.json"
    graph_path = "bayes_network/data/graphs/graphs_10_5.json"
    orlov_path = "bayes_network/data/Orlov.json"
    logfile = "bayes_network/data/opt_calc.txt"
    trainer = BayesianTrainer(graph_path, orlov_path, logfile_path=logfile)
    final_probs, final_loss = trainer.train(
        epochs=1000, lr=1e-3, log_interval=10,
        intermediate_file="bayes_network/data/intermediate_probabilities_opt.json",
        best_res_file = "bayes_network/data/probabilities_opt_pytorch.json",

        weight_decay = 1e-5,
        parent_dropout = 0.1,
        config_dropout = 0,
        temp = 1,
        batch_size = 8,
        k_folds = 5,
        max_grad_norm = 1
    )
    print("Final loss:", final_loss)
    # trainer.save_probabilities(final_probs, "bayes_network/data/probabilities_opt_pytorch.json")
