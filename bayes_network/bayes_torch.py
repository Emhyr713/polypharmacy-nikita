# bayes_torch.py
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


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


class BayesianTorchModel(nn.Module):
    def __init__(self, graph_data: dict, prob_data: Optional[dict] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # Node indexing
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

        # Parent and child maps
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

        # Number of nodes must be defined before topological sort
        self.n_nodes = len(self.node_ids)

        # Now topological sort is safe
        self.sorted_idx = self._topological_sort()

        # CPT configuration parameters
        self.cpt_parents_count = [len(self.parents_idx[i]) for i in range(self.n_nodes)]
        self.configs_per_node = [
            1 << k if k > 0 else 1
            for k in self.cpt_parents_count
        ]
        self.configs_matrix = [
            _configs_matrix(k, self.device) if k > 0 else torch.empty((1, 0), device=self.device)
            for k in self.cpt_parents_count
        ]
        # Initialize logit parameters from provided prob_data if available, otherwise random in logit space
        self.logit_params = nn.ParameterList()
        for i in range(self.n_nodes):
            n_conf = self.configs_per_node[i]

            if prob_data and self.idx_to_id[i] in prob_data:
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

            else:
                init_p = torch.rand(n_conf, dtype=torch.float32, device=self.device) * 0.8 + 0.1
                logits = torch.log(init_p / (1.0 - init_p))
                param = nn.Parameter(logits)

            self.logit_params.append(param)

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
        device = self.device
        evidence = evidence or {}
        marginals = torch.zeros(self.n_nodes, dtype=torch.float32, device=device)

        for idx in self.sorted_idx:
            if idx in evidence:
                marginals[idx] = torch.tensor(float(evidence[idx]), dtype=torch.float32, device=device)
                continue
            k = self.cpt_parents_count[idx]
            logits = self.logit_params[idx]
            p_cpt = torch.sigmoid(logits)  # shape (n_conf,)

            if k == 0:
                marginals[idx] = p_cpt.squeeze()
                continue

            parent_indices = self.parents_idx[idx]
            parent_probs = marginals[parent_indices]  # shape (k,)

            configs = self.configs_matrix[idx]  # shape (n_conf, k) float
            parent_probs_exp = parent_probs.unsqueeze(0)  # (1, k)
            probs_given_config = configs * parent_probs_exp + (1.0 - configs) * (1.0 - parent_probs_exp)  # (n_conf, k)
            prod_probs = torch.prod(probs_given_config, dim=1)  # (n_conf,)
            marg = torch.sum(p_cpt * prod_probs)
            marginals[idx] = marg

        return marginals  # tensor of length n_nodes

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
    def __init__(self, graph_path: str, orlov_path: str, logfile_path: str = "opt_calc.txt", device: Optional[torch.device] = None):
        self.logfile_path = logfile_path
        self.graph_json = load_json(graph_path)
        self.orlov_raw = load_json(orlov_path)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.model = BayesianTorchModel(self.graph_json, prob_data=None, device=self.device)
        # Build mapping from names to indices
        self.name_to_idx = {self.model.node_meta[i]["name"]: i for i in range(self.model.n_nodes)}
        self.label_map = {self.model.node_meta[i]["label"]: self.model.node_meta[i]["label"] for i in range(self.model.n_nodes)}
        # Parse Orlov dataset into internal mapping: drug_idx -> list of (se_idx, target_prob)
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

    def loss_for_current(self, marginals: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for drug_idx, se_entries in self.orlov.items():
            # enforce evidence drug=1
            evidence = {drug_idx: 1.0}
            marg_with_evidence = self.model(evidence=evidence)
            for se_idx, target in se_entries:
                pred = marg_with_evidence[se_idx]
                loss = loss + (pred - target) ** 2
        return loss
    
    def save_probabilities(self, prob_data, path: str):
        save_json(prob_data, path)

    def train(
            self,
            epochs: int = 500,
            lr: float = 1e-2,
            log_interval: int = 10,
            intermediate_file: str = "intermediate_probabilities_opt.json"
        ):
        log_to_file(self.logfile_path, "", init=True)
        log_to_file(self.logfile_path, f"Starting training on device: {self.device}", add_timestamp=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            marginals = self.model(evidence={})
            loss = self.loss_for_current(marginals)
            loss.backward()

            optimizer.step()
            # scheduler.step()

            with torch.no_grad():
                for p in self.model.logit_params:
                    pass

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                log_to_file(self.logfile_path, f"Epoch {epoch}/{epochs} Loss: {loss.item():.8f}")
                prob_data = self.model.export_prob_data()
                self.save_probabilities(prob_data, intermediate_file)

                
        final_prob_data = self.model.export_prob_data()
        self.save_probabilities(final_prob_data, "probabilities_opt.json")
        log_to_file(self.logfile_path, "Training finished. Final probabilities saved to probabilities_opt.json")

        debug = {}
        for idx in self.model.sorted_idx:
            node_id = self.model.idx_to_id[idx]
            node = self.model.node_meta[idx]
            debug[node_id] = {
                "name": node["name"],
                "label": node.get("label", ""),
                "parents_idx": self.model.parents_idx[idx],
                "parents_names": [self.model.node_meta[p]["name"] for p in self.model.parents_idx[idx]],
                "prob_table": final_prob_data[node_id]
            }
        save_json(debug, "debug_probabilities_opt.json")

        final_loss_tensor = self.loss_for_current(self.model(evidence={}))
        final_loss = final_loss_tensor.detach().cpu().numpy().item()
        return final_prob_data, final_loss
    

if __name__ == "__main__":
    # graph_path = "bayes_network/data/graphs/TEST.json"
    graph_path = "bayes_network/data/graphs/graphs_10_5.json"
    orlov_path = "bayes_network/data/Orlov.json"
    logfile = "bayes_network/data/opt_calc.txt"
    trainer = BayesianTrainer(graph_path, orlov_path, logfile_path=logfile)
    final_probs, final_loss = trainer.train(epochs=1000, lr=1e-2, log_interval=10, intermediate_file="bayes_network/data/intermediate_probabilities_opt.json")
    print("Final loss:", final_loss)
    trainer.save_probabilities(final_probs, "bayes_network/data/probabilities_opt_pytorch.json")
