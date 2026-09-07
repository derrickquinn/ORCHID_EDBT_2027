import torch


class TorchKmeans:
    def __init__(
        self,
        d,
        k,
        niter=10,
        verbose=False,
        seed=0,
        *,
        spherical=False,
        bf16_clustering=False,
        k_block: int | None = None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.verbose = bool(verbose)
        self.seed = int(seed)
        self.spherical = bool(spherical)
        self.bf16_clustering = bool(bf16_clustering)
        self.k_block = (
            self.k if not k_block or k_block <= 0 else min(int(k_block), self.k)
        )
        self.centroids = torch.empty((0, self.d), dtype=torch.float32)
        self.obj = None
        self._assign_block_size = 32768
        self._assign_scores_bf16 = None
        self._assign_scores_f32 = None
        self._assign_dists = None
        self._assign_best_vals_bf16 = None
        self._assign_best_vals_f32 = None
        self._assign_block_vals_bf16 = None
        self._assign_block_vals_f32 = None
        self._assign_best_idx = None
        self._assign_block_idx = None

    def _ensure_scores(
        self, device: torch.device, dtype: torch.dtype, *, k_use: int
    ) -> torch.Tensor:
        buf = (
            self._assign_scores_bf16
            if dtype == torch.bfloat16
            else self._assign_scores_f32
        )
        if (
            buf is None
            or buf.shape[0] != self._assign_block_size
            or buf.shape[1] != k_use
            or buf.dtype != dtype
            or buf.device != device
        ):
            buf = torch.empty(
                (self._assign_block_size, k_use), device=device, dtype=dtype
            )
            if dtype == torch.bfloat16:
                self._assign_scores_bf16 = buf
            else:
                self._assign_scores_f32 = buf
        return buf

    def _ensure_dists(self, device: torch.device, *, k_use: int) -> torch.Tensor:
        buf = self._assign_dists
        if (
            buf is None
            or buf.shape[0] != self._assign_block_size
            or buf.shape[1] != k_use
            or buf.dtype != torch.float32
            or buf.device != device
        ):
            buf = torch.empty(
                (self._assign_block_size, k_use), device=device, dtype=torch.float32
            )
            self._assign_dists = buf
        return buf

    def _ensure_best_vals(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        buf = (
            self._assign_best_vals_bf16
            if dtype == torch.bfloat16
            else self._assign_best_vals_f32
        )
        if (
            buf is None
            or buf.shape[0] != self._assign_block_size
            or buf.dtype != dtype
            or buf.device != device
        ):
            buf = torch.empty(
                (self._assign_block_size,), device=device, dtype=dtype
            )
            if dtype == torch.bfloat16:
                self._assign_best_vals_bf16 = buf
            else:
                self._assign_best_vals_f32 = buf
        return buf

    def _ensure_block_vals(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        buf = (
            self._assign_block_vals_bf16
            if dtype == torch.bfloat16
            else self._assign_block_vals_f32
        )
        if (
            buf is None
            or buf.shape[0] != self._assign_block_size
            or buf.dtype != dtype
            or buf.device != device
        ):
            buf = torch.empty(
                (self._assign_block_size,), device=device, dtype=dtype
            )
            if dtype == torch.bfloat16:
                self._assign_block_vals_bf16 = buf
            else:
                self._assign_block_vals_f32 = buf
        return buf

    def _ensure_best_idx(self, device: torch.device) -> torch.Tensor:
        buf = self._assign_best_idx
        if (
            buf is None
            or buf.shape[0] != self._assign_block_size
            or buf.dtype != torch.int64
            or buf.device != device
        ):
            buf = torch.empty(
                (self._assign_block_size,), device=device, dtype=torch.int64
            )
            self._assign_best_idx = buf
        return buf

    def _ensure_block_idx(self, device: torch.device) -> torch.Tensor:
        buf = self._assign_block_idx
        if (
            buf is None
            or buf.shape[0] != self._assign_block_size
            or buf.dtype != torch.int64
            or buf.device != device
        ):
            buf = torch.empty(
                (self._assign_block_size,), device=device, dtype=torch.int64
            )
            self._assign_block_idx = buf
        return buf

    def _assign_spherical_block(
        self,
        x_block: torch.Tensor,
        centroids_t_bf16: torch.Tensor,
        scores_bf16: torch.Tensor,
        k_block: int,
        device: torch.device,
    ) -> torch.Tensor:
        block_len = x_block.shape[0]
        best_vals = self._ensure_best_vals(device, torch.bfloat16)[:block_len]
        best_idx = self._ensure_best_idx(device)[:block_len]
        block_vals = self._ensure_block_vals(device, torch.bfloat16)[:block_len]
        block_idx = self._ensure_block_idx(device)[:block_len]
        best_vals.fill_(float("-inf"))
        best_idx.zero_()
        for cstart in range(0, self.k, k_block):
            cend = min(cstart + k_block, self.k)
            k_len = cend - cstart
            scores = scores_bf16[:block_len, :k_len]
            torch.mm(x_block, centroids_t_bf16[:, cstart:cend], out=scores)
            torch.max(scores, dim=1, out=(block_vals, block_idx))
            mask = block_vals > best_vals
            best_vals[mask] = block_vals[mask]
            best_idx[mask] = block_idx[mask] + cstart
        return best_idx

    def _assign_l2_block(
        self,
        x_block: torch.Tensor,
        centroids_t_bf16: torch.Tensor,
        c_norms: torch.Tensor,
        scores_bf16: torch.Tensor,
        scores_f32: torch.Tensor,
        dists: torch.Tensor,
        k_block: int,
        device: torch.device,
    ) -> torch.Tensor:
        block_len = x_block.shape[0]
        x_norms = (x_block.float() * x_block.float()).sum(dim=1, keepdim=True)
        best_vals = self._ensure_best_vals(device, torch.float32)[:block_len]
        best_idx = self._ensure_best_idx(device)[:block_len]
        block_vals = self._ensure_block_vals(device, torch.float32)[:block_len]
        block_idx = self._ensure_block_idx(device)[:block_len]
        best_vals.fill_(float("inf"))
        best_idx.zero_()
        for cstart in range(0, self.k, k_block):
            cend = min(cstart + k_block, self.k)
            k_len = cend - cstart
            scores = scores_bf16[:block_len, :k_len]
            dists_block = dists[:block_len, :k_len]
            scores_f32_block = scores_f32[:block_len, :k_len]
            torch.mm(x_block, centroids_t_bf16[:, cstart:cend], out=scores)
            torch.add(x_norms, c_norms[:, cstart:cend], out=dists_block)
            scores_f32_block.copy_(scores)
            dists_block.add_(scores_f32_block, alpha=-2.0)
            torch.min(dists_block, dim=1, out=(block_vals, block_idx))
            mask = block_vals < best_vals
            best_vals[mask] = block_vals[mask]
            best_idx[mask] = block_idx[mask] + cstart
        return best_idx

    def _labels_spherical_blocked(
        self,
        x_bf16: torch.Tensor,
        centroids_t_bf16: torch.Tensor,
        scores_buf: torch.Tensor,
        k_block: int,
        device: torch.device,
    ) -> torch.Tensor:
        n = x_bf16.shape[0]
        best_vals = torch.full((n,), float("-inf"), device=device)
        best_idx = torch.zeros((n,), device=device, dtype=torch.int64)
        for start in range(0, self.k, k_block):
            end = min(start + k_block, self.k)
            k_len = end - start
            scores = scores_buf[:, :k_len]
            torch.mm(x_bf16, centroids_t_bf16[:, start:end], out=scores)
            block_vals, block_idx = scores.max(dim=1)
            block_vals = block_vals.float()
            mask = block_vals > best_vals
            best_vals[mask] = block_vals[mask]
            best_idx[mask] = block_idx[mask] + start
        return best_idx

    def _labels_l2_blocked(
        self,
        x_bf16: torch.Tensor,
        centroids_t_bf16: torch.Tensor,
        x_norms: torch.Tensor,
        c_norms: torch.Tensor,
        scores_buf: torch.Tensor,
        dists_buf: torch.Tensor,
        k_block: int,
        device: torch.device,
    ) -> torch.Tensor:
        n = x_bf16.shape[0]
        best_vals = torch.full((n,), float("inf"), device=device)
        best_idx = torch.zeros((n,), device=device, dtype=torch.int64)
        for start in range(0, self.k, k_block):
            end = min(start + k_block, self.k)
            k_len = end - start
            scores = scores_buf[:, :k_len]
            dists = dists_buf[:, :k_len]
            torch.mm(x_bf16, centroids_t_bf16[:, start:end], out=scores)
            torch.add(x_norms, c_norms[:, start:end], out=dists)
            dists.add_(scores.float(), alpha=-2.0)
            block_vals, block_idx = dists.min(dim=1)
            mask = block_vals < best_vals
            best_vals[mask] = block_vals[mask]
            best_idx[mask] = block_idx[mask] + start
        return best_idx

    def assign(self, x: torch.Tensor, *, use_gpu: bool = False) -> torch.Tensor:
        device = torch.device("cuda") if use_gpu else torch.device("cpu")
        x = x.to(device=device, dtype=torch.bfloat16).contiguous()
        centroids = self.centroids.to(device=device, dtype=torch.float32).contiguous()
        centroids_t_bf16 = centroids.t().contiguous().to(torch.bfloat16)
        c_norms = None
        if not self.spherical:
            c_norms = (centroids * centroids).sum(dim=1).unsqueeze(0)

        n = x.shape[0]
        clusters = torch.empty((n,), device=device, dtype=torch.int64)
        block_size = self._assign_block_size
        k_block = self.k_block

        scores_bf16 = self._ensure_scores(device, torch.bfloat16, k_use=k_block)
        scores_f32 = None
        dists = None
        if not self.spherical:
            scores_f32 = self._ensure_scores(device, torch.float32, k_use=k_block)
            dists = self._ensure_dists(device, k_use=k_block)

        with torch.no_grad():
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                x_block = x[start:end]
                if self.spherical:
                    labels = self._assign_spherical_block(
                        x_block,
                        centroids_t_bf16,
                        scores_bf16,
                        k_block,
                        device,
                    )
                else:
                    labels = self._assign_l2_block(
                        x_block,
                        centroids_t_bf16,
                        c_norms,
                        scores_bf16,
                        scores_f32,
                        dists,
                        k_block,
                        device,
                    )
                clusters[start:end] = labels

        return clusters

    def train(self, x: torch.Tensor) -> None:
        x = x.to(dtype=torch.float32).contiguous()
        if self.spherical:
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        x_bf16 = x.to(torch.bfloat16)

        g = torch.Generator(device=x.device)
        g.manual_seed(self.seed)

        n = x.shape[0]
        perm = torch.randperm(n, generator=g, device=x.device)
        centroids = x[perm[: self.k]].clone()
        x_norms = None
        if not self.spherical:
            x_norms = (x * x).sum(dim=1, keepdim=True)

        k_block = self.k_block
        scores_buf = torch.empty((n, k_block), device=x.device, dtype=torch.bfloat16)
        dists_buf = None
        if not self.spherical:
            dists_buf = torch.empty((n, k_block), device=x.device, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(self.niter):
                if self.spherical:
                    centroids = centroids / centroids.norm(
                        dim=1, keepdim=True
                    ).clamp_min(1e-12)
                centroids_t_bf16 = centroids.t().contiguous().to(torch.bfloat16)
                if self.spherical:
                    labels = self._labels_spherical_blocked(
                        x_bf16,
                        centroids_t_bf16,
                        scores_buf,
                        k_block,
                        x.device,
                    )
                else:
                    c_norms = (centroids * centroids).sum(dim=1).unsqueeze(0)
                    labels = self._labels_l2_blocked(
                        x_bf16,
                        centroids_t_bf16,
                        x_norms,
                        c_norms,
                        scores_buf,
                        dists_buf,
                        k_block,
                        x.device,
                    )

                new_centroids = torch.zeros_like(centroids)
                new_centroids.scatter_add_(
                    0, labels[:, None].expand(-1, self.d), x
                )
                counts = torch.bincount(labels, minlength=self.k)
                empty = counts == 0
                counts = counts.clamp_min(1)
                new_centroids = new_centroids / counts[:, None]
                if empty.any():
                    repl = x[
                        torch.randint(
                            0,
                            n,
                            (int(empty.sum()),),
                            generator=g,
                            device=x.device,
                        )
                    ]
                    new_centroids[empty] = repl
                centroids = new_centroids

        if self.spherical:
            centroids = centroids / centroids.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scores = x_bf16 @ centroids.t().to(torch.bfloat16)
            self.obj = scores.max(dim=1).values.float().mean().item()
        else:
            c_norms = (centroids * centroids).sum(dim=1).unsqueeze(0)
            scores = x_bf16 @ centroids.t().to(torch.bfloat16)
            dists = x_norms + c_norms - 2.0 * scores.float()
            self.obj = dists.min(dim=1).values.mean().item()

        self.centroids = centroids
