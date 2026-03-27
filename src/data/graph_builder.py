"""
Graph construction pipeline for EV charging network representation.

Converts tabular network traffic / IDS records into PyTorch Geometric
Data objects, implementing:

    EV_n(gc_j, cs_i) = 1 - if(CC_j, OS_i, LC_i)         [Eq. 5]
    ED_n = 0.5 * [μ(de) + μ(pf_tot)] * (EM_i, r_ij)      [Eq. 6]
    gc = Σ_i go*NA_i*Ec_i / Σ_i (RNC + ds_i) + hd_e(.)   [Eq. 7]

For tabular IDS data, we construct a k-NN graph over feature space
to approximate the communication topology of the EV network.
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Converts arrays of feature vectors into batches of graph Data objects.

    Each sample becomes a "mini-graph" where:
        - Nodes represent sessions / EV entities / grid components.
        - Edges are constructed via k-NN similarity in feature space,
          representing communication, power-flow, and authentication links.

    Args:
        k_neighbors (int): Number of nearest neighbors per node in the k-NN graph.
        batch_size (int): Number of flow records grouped into one graph.
        edge_method (str): 'knn' (feature-space k-NN) or 'sequential' (temporal chain).
        self_loops (bool): Whether to include self-loop edges.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        batch_size: int = 10,
        edge_method: str = "knn",
        self_loops: bool = False,
    ):
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.edge_method = edge_method
        self.self_loops = self_loops

    def build_graphs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_label_mode: str = "majority",
    ) -> List[Data]:
        """
        Build a list of PyG Data objects from feature matrix X and labels y.

        Each Data object represents a group of `batch_size` consecutive records
        as a graph with nodes = records, edges = k-NN or sequential links.

        Args:
            X: Feature matrix [N, F].
            y: Label vector [N] (integer class indices).
            window_label_mode: How to assign graph-level label.
                'majority' = most frequent label in window.
                'any_attack' = 1 if any attack present.

        Returns:
            List of torch_geometric.data.Data objects.
        """
        N, F = X.shape
        data_list = []

        n_graphs = N // self.batch_size
        logger.info(f"Building {n_graphs} graphs from {N} samples (window={self.batch_size})")

        for i in range(n_graphs):
            start = i * self.batch_size
            end = start + self.batch_size
            x_window = X[start:end]  # [batch_size, F]
            y_window = y[start:end]  # [batch_size]

            # Compute graph-level label
            if window_label_mode == "majority":
                graph_label = int(np.bincount(y_window).argmax())
            else:
                graph_label = int(np.any(y_window > 0))

            # Build edge index
            edge_index = self._build_edges(x_window)

            # Node features → float tensor
            x_tensor = torch.tensor(x_window, dtype=torch.float)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            y_tensor = torch.tensor([graph_label], dtype=torch.long)

            # Per-node labels (used in semi-supervised training)
            node_labels = torch.tensor(y_window, dtype=torch.long)

            data = Data(
                x=x_tensor,
                edge_index=edge_index_tensor,
                y=y_tensor,
                node_labels=node_labels,
                num_nodes=self.batch_size,
            )
            data_list.append(data)

        logger.info(f"Built {len(data_list)} graph objects.")
        return data_list

    def _build_edges(self, x_window: np.ndarray) -> np.ndarray:
        """
        Construct edge_index from node feature matrix.

        Returns:
            edge_index: [2, E] array (source, destination).
        """
        n = len(x_window)

        if self.edge_method == "knn":
            edges = self._knn_edges(x_window)
        elif self.edge_method == "sequential":
            edges = self._sequential_edges(n)
        else:
            raise ValueError(f"Unknown edge_method: {self.edge_method}")

        if self.self_loops:
            self_loop = np.stack([np.arange(n), np.arange(n)], axis=0)
            edges = np.concatenate([edges, self_loop], axis=1)

        return edges

    def _knn_edges(self, x_window: np.ndarray) -> np.ndarray:
        """
        Build edges using k-nearest-neighbours in feature space.
        Approximates communication / interaction graph in EV network.
        """
        n = len(x_window)
        k = min(self.k_neighbors, n - 1)

        if k < 1:
            # Single node graph — no edges
            return np.zeros((2, 0), dtype=np.int64)

        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="ball_tree")
        nbrs.fit(x_window)
        indices = nbrs.kneighbors(x_window, return_distance=False)  # [n, k+1]

        # Build edge list (exclude self)
        sources, targets = [], []
        for src, neighbors in enumerate(indices):
            for tgt in neighbors:
                if tgt != src:
                    sources.append(src)
                    targets.append(tgt)

        return np.stack([sources, targets], axis=0).astype(np.int64)

    def _sequential_edges(self, n: int) -> np.ndarray:
        """
        Build a chain graph: each node connected to the next.
        Useful for temporal sequences of EV charging events.
        """
        if n <= 1:
            return np.zeros((2, 0), dtype=np.int64)
        src = np.arange(n - 1)
        tgt = np.arange(1, n)
        # Bidirectional chain
        return np.stack(
            [np.concatenate([src, tgt]), np.concatenate([tgt, src])], axis=0
        ).astype(np.int64)

    def build_ev_topology_graph(
        self,
        n_ev: int,
        n_cs: int,
        n_gc: int,
        ev_features: np.ndarray,
        cs_features: np.ndarray,
        gc_features: np.ndarray,
    ) -> Data:
        """
        Construct a heterogeneous EV charging topology graph:
            - EV_n: electric vehicle nodes
            - cs_i: charging station nodes
            - gc_j: grid component nodes

        Implements equations (5)-(7) from the paper.

        Args:
            n_ev: Number of EV nodes.
            n_cs: Number of charging station nodes.
            n_gc: Number of grid component nodes.
            ev_features: [n_ev, F_ev] feature matrix.
            cs_features: [n_cs, F_cs] feature matrix.
            gc_features: [n_gc, F_gc] feature matrix.

        Returns:
            PyG Data object with node features and edge_index.
        """
        # Pad features to same dimension (F_max)
        F_max = max(ev_features.shape[1], cs_features.shape[1], gc_features.shape[1])

        def pad(arr, target_cols):
            pad_width = target_cols - arr.shape[1]
            return np.pad(arr, ((0, 0), (0, pad_width))) if pad_width > 0 else arr

        ev_feat = pad(ev_features, F_max)
        cs_feat = pad(cs_features, F_max)
        gc_feat = pad(gc_features, F_max)

        x = np.concatenate([ev_feat, cs_feat, gc_feat], axis=0)  # [N_total, F_max]

        # Node type indicators: 0=EV, 1=CS, 2=GC
        node_type = np.array(
            [0] * n_ev + [1] * n_cs + [2] * n_gc, dtype=np.int64
        )

        # Build EV→CS edges (Eq. 5: EV_n connected to cs_i)
        ev_cs_src, ev_cs_tgt = [], []
        for ev_idx in range(n_ev):
            for cs_idx in range(n_cs):
                ev_cs_src.append(ev_idx)
                ev_cs_tgt.append(n_ev + cs_idx)
                ev_cs_src.append(n_ev + cs_idx)
                ev_cs_tgt.append(ev_idx)

        # Build CS→GC edges (Eq. 6: power flow)
        cs_gc_src, cs_gc_tgt = [], []
        for cs_idx in range(n_cs):
            for gc_idx in range(n_gc):
                cs_gc_src.append(n_ev + cs_idx)
                cs_gc_tgt.append(n_ev + n_cs + gc_idx)
                cs_gc_src.append(n_ev + n_cs + gc_idx)
                cs_gc_tgt.append(n_ev + cs_idx)

        src = np.array(ev_cs_src + cs_gc_src, dtype=np.int64)
        tgt = np.array(ev_cs_tgt + cs_gc_tgt, dtype=np.int64)
        edge_index = np.stack([src, tgt], axis=0)

        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_type=torch.tensor(node_type, dtype=torch.long),
            num_nodes=n_ev + n_cs + n_gc,
        )
