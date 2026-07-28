from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.necks import FPN
from mmdet.registry import MODELS


@MODELS.register_module()
class DBSSFPN(FPN):
    """FPN with dynamic background-subspace suppression on P3."""

    def __init__(
            self,
            *args,
            embed_channels: int = 64,
            candidate_grid: tuple[int, int] = (8, 8),
            shortlist_size: int = 24,
            num_bases: int = 8,
            diversity_beta: float = 1.0,
            basis_similarity_threshold: float = 0.9,
            projection_mode: str = 'ridge',
            ridge_lambda: float = 1e-3,
            temperature: float = 0.1,
            gamma_max: float = 0.1,
            use_haar_reliability: bool = False,
            hidden_channels: int = 64,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        candidate_count = math.prod(candidate_grid)
        if embed_channels < 1 or hidden_channels < 1:
            raise ValueError('DBSS channel counts must be positive')
        if min(candidate_grid) < 1:
            raise ValueError('candidate_grid values must be positive')
        if not 1 <= num_bases <= shortlist_size <= candidate_count:
            raise ValueError(
                'Require 1 <= num_bases <= shortlist_size <= grid size')
        if diversity_beta < 0:
            raise ValueError('diversity_beta must be non-negative')
        if not -1 <= basis_similarity_threshold <= 1:
            raise ValueError(
                'basis_similarity_threshold must be in [-1, 1]')
        if projection_mode not in {'ridge', 'softmax'}:
            raise ValueError(
                "projection_mode must be either 'ridge' or 'softmax'")
        if ridge_lambda <= 0 or temperature <= 0 or gamma_max < 0:
            raise ValueError(
                'ridge_lambda and temperature must be positive and '
                'gamma_max non-negative')

        self.embed_channels = int(embed_channels)
        self.candidate_grid = tuple(candidate_grid)
        self.shortlist_size = int(shortlist_size)
        self.num_bases = int(num_bases)
        self.diversity_beta = float(diversity_beta)
        self.basis_similarity_threshold = float(
            basis_similarity_threshold)
        self.projection_mode = projection_mode
        self.ridge_lambda = float(ridge_lambda)
        self.temperature = float(temperature)
        self.gamma_max = float(gamma_max)
        self.use_haar_reliability = bool(use_haar_reliability)

        channels = self.out_channels
        self.embedding = nn.Conv2d(channels, embed_channels, 1)
        self.embedding_norm = nn.LayerNorm(embed_channels)
        direction_in = channels + embed_channels
        self.direction = nn.Sequential(
            nn.Conv2d(direction_in, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1))
        magnitude_in = embed_channels
        if self.use_haar_reliability:
            self.haar_projection = nn.Conv2d(3 * channels, embed_channels, 1)
            magnitude_in += embed_channels
        else:
            self.haar_projection = None
        self.magnitude = nn.Sequential(
            nn.Conv2d(magnitude_in, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1))
        self._init_dbss_output()

    def _init_dbss_output(self) -> None:
        nn.init.zeros_(self.direction[-1].weight)
        nn.init.zeros_(self.direction[-1].bias)

    def init_weights(self) -> None:
        super().init_weights()
        self._init_dbss_output()

    def _embed(self, p3: Tensor) -> Tensor:
        embedding = self.embedding(p3).permute(0, 2, 3, 1)
        return self.embedding_norm(embedding).permute(0, 3, 1, 2)

    @staticmethod
    def _normalize_valid_shapes(
            p3: Tensor,
            valid_shapes: Sequence[tuple[int, int]] | None
    ) -> list[tuple[int, int]]:
        batch_size, _, height, width = p3.shape
        if valid_shapes is None:
            return [(height, width)] * batch_size
        if len(valid_shapes) != batch_size:
            raise ValueError(
                f'Expected {batch_size} valid shapes, got {len(valid_shapes)}')
        output = []
        for valid_height, valid_width in valid_shapes:
            valid_height = min(height, max(1, int(valid_height)))
            valid_width = min(width, max(1, int(valid_width)))
            output.append((valid_height, valid_width))
        return output

    def _select_bases(self, scores: Tensor,
                      normalized_candidates: Tensor) -> Tensor:
        shortlist = torch.topk(
            scores, k=min(self.shortlist_size, scores.numel())).indices
        chosen = [shortlist[0]]
        available = torch.ones(
            shortlist.numel(), dtype=torch.bool, device=scores.device)
        available[0] = False
        for _ in range(1, min(self.num_bases, shortlist.numel())):
            candidates = normalized_candidates[shortlist]
            selected = normalized_candidates[torch.stack(chosen)]
            redundancy = candidates @ selected.transpose(0, 1)
            max_similarity = redundancy.max(dim=1).values
            selection_score = (
                scores[shortlist]
                - self.diversity_beta * max_similarity)
            diverse = available & (
                max_similarity <= self.basis_similarity_threshold)
            eligible = diverse if diverse.any() else available
            selection_score = selection_score.masked_fill(
                ~eligible, -torch.inf)
            next_position = selection_score.argmax()
            chosen.append(shortlist[next_position])
            available[next_position] = False

        if len(chosen) < self.num_bases:
            ranked = torch.argsort(scores, descending=True)
            chosen_values = {int(index) for index in chosen}
            for index in ranked:
                if int(index) not in chosen_values:
                    chosen.append(index)
                    chosen_values.add(int(index))
                if len(chosen) == self.num_bases:
                    break
        return torch.stack(chosen)

    def _project(self, tokens: Tensor, bases: Tensor) -> Tensor:
        original_dtype = tokens.dtype
        if self.projection_mode == 'ridge':
            with torch.autocast(
                    device_type=tokens.device.type, enabled=False):
                tokens32 = tokens.float()
                bases32 = bases.float()
                gram = bases32 @ bases32.transpose(0, 1)
                gram = gram + self.ridge_lambda * torch.eye(
                    bases.shape[0], device=bases.device, dtype=torch.float32)
                rhs = bases32 @ tokens32.transpose(0, 1)
                coefficients = torch.linalg.solve(gram, rhs)
                projected = coefficients.transpose(0, 1) @ bases32
            return projected.to(original_dtype)

        normalized_tokens = F.normalize(tokens, dim=-1)
        normalized_bases = F.normalize(bases, dim=-1)
        weights = (
            normalized_tokens @ normalized_bases.transpose(0, 1)
            / self.temperature).softmax(dim=-1)
        return weights @ bases

    @staticmethod
    def _basis_diagnostics(normalized_bases: Tensor) -> tuple[Tensor, Tensor]:
        pairwise = normalized_bases @ normalized_bases.transpose(0, 1)
        if normalized_bases.shape[0] == 1:
            max_cosine = pairwise.new_zeros(())
        else:
            mask = torch.eye(
                pairwise.shape[0], dtype=torch.bool, device=pairwise.device)
            max_cosine = pairwise.masked_fill(mask, -torch.inf).max()
        singular_values = torch.linalg.svdvals(normalized_bases.float())
        probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        return max_cosine, effective_rank.to(normalized_bases.dtype)

    def _background_residual(
            self, p3: Tensor, embedding: Tensor,
            valid_shapes: Sequence[tuple[int, int]]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _, _, height, width = p3.shape
        residuals = []
        selected_indices = []
        representativeness = []
        selected_raw_candidates = []
        max_cosines = []
        effective_ranks = []

        for image_index, (valid_height, valid_width) in enumerate(valid_shapes):
            valid_embedding = embedding[
                image_index:image_index + 1, :, :valid_height, :valid_width]
            valid_p3 = p3[
                image_index:image_index + 1, :, :valid_height, :valid_width]
            candidate64 = F.adaptive_avg_pool2d(
                valid_embedding, self.candidate_grid).flatten(2).squeeze(0).t()
            candidate256 = F.adaptive_avg_pool2d(
                valid_p3, self.candidate_grid).flatten(2).squeeze(0).t()
            tokens = valid_embedding.flatten(2).squeeze(0).t()
            normalized_tokens = F.normalize(tokens, dim=-1)
            normalized_candidates = F.normalize(candidate64, dim=-1)
            scores = (
                normalized_candidates
                @ normalized_tokens.transpose(0, 1)).mean(dim=1)
            indices = self._select_bases(scores, normalized_candidates)
            bases = candidate64[indices]
            normalized_bases = normalized_candidates[indices]
            background = self._project(tokens, bases)
            valid_residual = (
                tokens - background).t().reshape(
                    1, self.embed_channels, valid_height, valid_width)
            residuals.append(F.pad(
                valid_residual,
                (0, width - valid_width, 0, height - valid_height)))
            max_cosine, effective_rank = self._basis_diagnostics(
                normalized_bases)
            selected_indices.append(indices)
            representativeness.append(scores)
            selected_raw_candidates.append(candidate256[indices])
            max_cosines.append(max_cosine)
            effective_ranks.append(effective_rank)

        aux = dict(
            selected_indices=torch.stack(selected_indices),
            representativeness=torch.stack(representativeness),
            selected_candidates_p3=torch.stack(selected_raw_candidates),
            basis_max_cosine=torch.stack(max_cosines),
            basis_effective_rank=torch.stack(effective_ranks))
        return torch.cat(residuals), aux

    @staticmethod
    def _haar_magnitude(p3: Tensor) -> Tensor:
        height, width = p3.shape[-2:]
        padded = F.pad(p3, (0, width % 2, 0, height % 2), mode='replicate')
        a = padded[..., 0::2, 0::2]
        b = padded[..., 0::2, 1::2]
        c = padded[..., 1::2, 0::2]
        d = padded[..., 1::2, 1::2]
        horizontal = (a - b + c - d).abs() * 0.5
        vertical = (a + b - c - d).abs() * 0.5
        diagonal = (a - b - c + d).abs() * 0.5
        return torch.cat((horizontal, vertical, diagonal), dim=1)

    def enhance(
            self,
            p3: Tensor,
            valid_shapes: Sequence[tuple[int, int]] | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        valid_shapes = self._normalize_valid_shapes(p3, valid_shapes)
        embedding = self._embed(p3)
        residual, aux = self._background_residual(
            p3, embedding, valid_shapes)

        direction = self.direction(torch.cat((p3, residual), dim=1))
        magnitude_context = [residual]
        if self.haar_projection is not None:
            reliability = self.haar_projection(self._haar_magnitude(p3))
            reliability = F.interpolate(
                reliability, size=p3.shape[-2:], mode='bilinear',
                align_corners=False)
            magnitude_context.append(reliability)
            aux['haar_reliability_rms'] = (
                reliability.square().mean().sqrt())
        gamma = self.magnitude(torch.cat(magnitude_context, dim=1)).sigmoid()
        feature_scale = (
            p3.square().mean(dim=1, keepdim=True) + 1e-6).sqrt().detach()
        direction_norm = direction.norm(dim=1, keepdim=True)
        displacement = (
            self.gamma_max * feature_scale * gamma * direction
            / (1 + direction_norm))
        enhanced = p3 + displacement
        aux.update(
            pre_p3=p3,
            post_p3=enhanced,
            residual=residual,
            gamma=gamma,
            displacement=displacement,
            feature_scale=feature_scale,
            displacement_ratio=(
                displacement.square().mean().sqrt()
                / (p3.square().mean().sqrt() + 1e-6)))
        return enhanced, aux

    def forward_with_aux(
            self,
            inputs: tuple[Tensor, ...],
            valid_shapes: Sequence[tuple[int, int]] | None = None
    ) -> tuple[tuple[Tensor, ...], dict[str, Tensor]]:
        outputs = list(super().forward(inputs))
        outputs[0], aux = self.enhance(outputs[0], valid_shapes)
        return tuple(outputs), aux

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return self.forward_with_aux(inputs)[0]
