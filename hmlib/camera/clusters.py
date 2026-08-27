"""Clustering helpers used for grouping tracked player positions.

Provides a small wrapper around k-means (either via fast_pytorch_kmeans or
the C++ `compute_kmeans_clusters` binding) and a manager for multi-k setups.
"""

from typing import Dict, List, Optional, Tuple

import torch
from fast_pytorch_kmeans import KMeans

from hockeymon.core import compute_kmeans_clusters


class ClusterSnapshot:
    """Single k-means snapshot for a fixed number of clusters."""

    def __init__(
        self,
        num_clusters: int,
        device: str,
        centroids: Optional[torch.Tensor] = None,
        minibatch: Optional[int] = None,
        init_method: Optional[str] = None,
        cpp_clusters: bool = True,
        # cpp_clusters: bool = False,
    ):
        self._device: str = device
        self._num_clusters: int = num_clusters
        self._centroids: torch.Tensor | None = centroids
        self._cpp_clusters: bool = cpp_clusters
        self._last_centroids: Optional[torch.Tensor] = None
        self._kmeans_object: KMeans[int] | None = (
            KMeans(
                n_clusters=num_clusters,
                mode="euclidean",
                minibatch=minibatch,
                init_method=init_method,
            )
            if not cpp_clusters
            else None
        )
        self.reset()

    def reset(self) -> None:
        self._cluster_label_ids = None
        self._largest_cluster_label = None
        self._largest_cluster_id_set = None
        self._cluster_counts = None
        self._last_centroids = None

    def calculate_labels(self, center_points: torch.Tensor, cpp_clusters: bool) -> torch.Tensor:
        if self._kmeans_object is None or cpp_clusters:
            points: List[float] = center_points.flatten().cpu().tolist()
            cluster_results: Tuple[List[int], Dict[int, List[int]]] = compute_kmeans_clusters(
                points=points,
                num_clusters=self._num_clusters,
                dim=len(center_points[0]),
            )
            labels: torch.Tensor = torch.tensor(
                cluster_results[0], dtype=torch.int64, device=center_points.device
            )
        else:
            centroids = self._centroids
            if centroids is not None:
                centroids = centroids.clone()[: self._num_clusters]
            labels = self._kmeans_object.fit_predict(
                center_points.to(self._device), centroids=centroids
            )
        return labels

    def calculate(self, center_points: torch.Tensor, ids: torch.Tensor) -> None:
        assert len(center_points) == len(ids)
        self.reset()
        if len(center_points) < self._num_clusters:
            # Minimum clusters that we can have is the minimum number of objects
            max_clusters = len(center_points)
            if not max_clusters:
                return
            elif max_clusters < self._num_clusters:
                return

        labels: torch.Tensor = self.calculate_labels(center_points, self._cpp_clusters)
        if labels.device != ids.device:
            labels = labels.to(device=ids.device)

        self._cluster_label_ids: List[torch.Tensor] = list()
        for i in range(self._num_clusters):
            cids = ids[labels == i]
            self._cluster_label_ids.append(cids)

        self._cluster_counts = torch.tensor(
            [len(t) for t in self._cluster_label_ids], dtype=torch.int64
        )
        index_of_max_count = torch.argmax(self._cluster_counts)
        self._largest_cluster_label = index_of_max_count
        self._largest_cluster_id_set = set(
            self._cluster_label_ids[self._largest_cluster_label].tolist()
        )

        try:
            centroids: List[torch.Tensor] = []
            for i in range(self._num_clusters):
                mask = labels == i
                if torch.any(mask):
                    centroids.append(center_points[mask].mean(dim=0))
                else:
                    centroids.append(
                        torch.zeros(
                            (center_points.shape[1],),
                            dtype=center_points.dtype,
                            device=center_points.device,
                        )
                    )
            self._last_centroids = torch.stack(centroids).detach().cpu()
        except Exception:
            self._last_centroids = None

    def prune_not_in_largest_cluster(self, ids) -> list | torch.Tensor:
        if not self._largest_cluster_id_set:
            return []
        result_ids = []
        for id in ids:
            id_item = id.item()
            if id_item in self._largest_cluster_id_set:
                result_ids.append(id)
        if not result_ids:
            return []
        return torch.tensor(result_ids, dtype=torch.int64, device=ids.device)

    def last_centroids(self) -> Optional[torch.Tensor]:
        if self._last_centroids is None:
            return None
        return self._last_centroids.clone()


class ClusterMan:
    """Manage multiple ClusterSnapshot instances for different cluster counts."""

    def __init__(
        self,
        sizes: List[int],
        device="cpu",
        init_method: Optional[str] = None,
        centroids: Optional[torch.Tensor] = None,
    ):
        self._sizes = sizes
        self._device = device
        self.cluster_snapshots = dict()
        for i in sizes:
            self.cluster_snapshots[i] = ClusterSnapshot(
                num_clusters=i,
                device=device,
                init_method="random" if not init_method else init_method,
                centroids=centroids,
            )

    @property
    def cluster_counts(self):
        return self._sizes

    def reset_clusters(self):
        for cs in self.cluster_snapshots.values():
            cs.reset()

    def calculate_all_clusters(self, center_points: torch.Tensor, ids: torch.Tensor):
        self.reset_clusters()
        if isinstance(ids, list):
            if not ids:
                # No items are currently being tracked
                # This should always be the case when 'ids' is of type 'list'
                # Allowin it to fall-through in order to catch as an error if it is not empty.
                return
        if ids.device != self._device:
            ids = ids.to(self._device)
        for cluster_snapshot in self.cluster_snapshots.values():
            cluster_snapshot.calculate(center_points=center_points, ids=ids)

    def prune_not_in_largest_cluster(self, num_clusters, ids):
        return self.cluster_snapshots[num_clusters].prune_not_in_largest_cluster(ids)

    def get_last_centroids(self, num_clusters: Optional[int] = None) -> Optional[torch.Tensor]:
        if not self.cluster_snapshots:
            return None
        if num_clusters is None:
            num_clusters = max(self._sizes)
        snap = self.cluster_snapshots.get(num_clusters)
        if snap is None:
            return None
        return snap.last_centroids()
