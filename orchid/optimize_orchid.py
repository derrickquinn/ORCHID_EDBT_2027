#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import test_orchid
from src.optimize_ivf import OptimizeIvfConfig, load_optimize_ivf_config


def _unique_preserve(values: list[float], *, ndigits: int = 12) -> list[float]:
    seen: set[float] = set()
    out: list[float] = []
    for value in values:
        key = round(value, ndigits)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _unique_pairs_preserve(
    values: list[tuple[float, float]], *, ndigits: int = 12
) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for u_val, v_val in values:
        key = (round(u_val, ndigits), round(v_val, ndigits))
        if key in seen:
            continue
        seen.add(key)
        out.append((u_val, v_val))
    return out


def _round_key(value: float, *, ndigits: int = 12) -> float:
    return round(value, ndigits)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise argparse.ArgumentTypeError("Boolean value is required")
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--nlist", type=int, default=512)
    parser.add_argument("--prob_dims", type=int, default=1)
    parser.add_argument("--pred_zarr", type=str, default="../predicate/laion_neg.zarr")
    parser.add_argument("--vec_zarr", type=str, default="../vector/laion.zarr")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--csv", type=str, default="logs/out.csv")
    parser.add_argument("--y_min", type=float, default=0.0001)
    parser.add_argument("--y_max", type=float, default=1.0)
    parser.add_argument("--x_min", type=float, default=0.01)
    parser.add_argument("--x_max", type=float, default=1.0)
    parser.add_argument("--num_queries", "-nq", type=int, default=512)
    parser.add_argument("--query_start", type=int, default=512)
    parser.add_argument("--sample_queries", type=int, default=512)
    parser.add_argument("--sample_docs", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--ablation_queries", type=int, default=1)
    parser.add_argument("--ablation_ood", action="store_true")
    parser.add_argument(
        "--use_gpu",
        "-gpu",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
    )
    parser.add_argument("--no_use_gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--predict_pl", "-ppl", action="store_true")

    parser.add_argument("--alpha0", type=float, default=None)
    parser.add_argument("--beta0", type=float, default=None)
    parser.add_argument("--step_log10", type=float, default=0.1)
    parser.add_argument("--min_step_log10", type=float, default=0.1)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--step_decay", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=1)
    return parser


def _cli_override_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    overrides: set[str] = set()
    for token in argv:
        if token == "--":
            break
        opt = token.split("=", 1)[0]
        action = parser._option_string_actions.get(opt)
        if action is not None and action.dest != "config":
            overrides.add(action.dest)
    return overrides


def _parse_args() -> OptimizeIvfConfig:
    parser = build_arg_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args_dict = vars(args)
    config_path = args_dict.pop("config", None)

    if config_path:
        base = load_optimize_ivf_config(config_path).model_dump()
        overrides = _cli_override_dests(parser, argv)
        for dest in overrides:
            if dest in args_dict:
                base[dest] = args_dict[dest]
        return OptimizeIvfConfig.model_validate(base)
    return OptimizeIvfConfig.model_validate(args_dict)


def _eval_pairs(
    state: test_orchid.TestIvfState,
    pairs: list[tuple[float, float]],
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> dict[tuple[float, float], test_orchid.IvfEvalResult]:
    if not pairs:
        return {}
    clamped_pairs = [
        (_clamp(u_val, u_min, u_max), _clamp(v_val, v_min, v_max))
        for u_val, v_val in pairs
    ]
    unique_pairs = _unique_pairs_preserve(clamped_pairs)

    u_vals: dict[float, float] = {}
    v_vals: dict[float, float] = {}
    u_keys: list[float] = []
    v_keys: list[float] = []
    pair_keys: set[tuple[float, float]] = set()
    for u_val, v_val in unique_pairs:
        u_key = _round_key(u_val)
        v_key = _round_key(v_val)
        pair_keys.add((u_key, v_key))
        if u_key not in u_vals:
            u_vals[u_key] = u_val
            u_keys.append(u_key)
        if v_key not in v_vals:
            v_vals[v_key] = v_val
            v_keys.append(v_key)

    def _eval_cartesian_block(
        u_keys_block: list[float], v_keys_block: list[float]
    ) -> dict[tuple[float, float], test_orchid.IvfEvalResult]:
        u_keys_sorted = sorted(u_keys_block)
        v_keys_sorted = sorted(v_keys_block)
        alphas = [10**u_vals[key] for key in u_keys_sorted]
        betas = [10**v_vals[key] for key in v_keys_sorted]
        results = test_orchid.evaluate_alphas(
            state,
            alphas,
            betas,
            write_csv=False,
            write_summary=False,
            csv_path=None,
        )
        block_map: dict[tuple[float, float], test_orchid.IvfEvalResult] = {}
        result_idx = 0
        for u_key in u_keys_sorted:
            for v_key in v_keys_sorted:
                block_map[(u_key, v_key)] = results[result_idx]
                result_idx += 1
        return block_map

    best_u_keys: list[float] | None = None
    best_v_keys: list[float] | None = None
    best_size = 0
    u_keys_sorted_all = sorted(u_keys)
    v_keys_sorted_all = sorted(v_keys)
    for u_mask in range(1, 1 << len(u_keys_sorted_all)):
        u_subset = [
            u_keys_sorted_all[i]
            for i in range(len(u_keys_sorted_all))
            if u_mask & (1 << i)
        ]
        for v_mask in range(1, 1 << len(v_keys_sorted_all)):
            v_subset = [
                v_keys_sorted_all[i]
                for i in range(len(v_keys_sorted_all))
                if v_mask & (1 << i)
            ]
            size = len(u_subset) * len(v_subset)
            if size < 2 or size < best_size:
                continue
            complete = True
            for u_key in u_subset:
                for v_key in v_subset:
                    if (u_key, v_key) not in pair_keys:
                        complete = False
                        break
                if not complete:
                    break
            if not complete:
                continue
            if size > best_size:
                best_size = size
                best_u_keys = u_subset
                best_v_keys = v_subset
            elif size == best_size:
                if best_u_keys is None or len(u_subset) > len(best_u_keys):
                    best_u_keys = u_subset
                    best_v_keys = v_subset
                elif best_v_keys is not None and len(u_subset) == len(best_u_keys):
                    if len(v_subset) > len(best_v_keys):
                        best_u_keys = u_subset
                        best_v_keys = v_subset

    result_map: dict[tuple[float, float], test_orchid.IvfEvalResult] = {}
    remaining_pairs = unique_pairs
    if best_size == len(pair_keys) and best_u_keys is not None and best_v_keys is not None:
        return _eval_cartesian_block(best_u_keys, best_v_keys)

    if best_size >= 2 and best_u_keys is not None and best_v_keys is not None:
        block_map = _eval_cartesian_block(best_u_keys, best_v_keys)
        result_map.update(block_map)
        block_keys = set(block_map.keys())
        remaining_pairs = [
            (u_val, v_val)
            for u_val, v_val in unique_pairs
            if (_round_key(u_val), _round_key(v_val)) not in block_keys
        ]
        if not remaining_pairs:
            return result_map

    grouped: dict[float, dict[float, float]] = {}
    u_vals_remaining: dict[float, float] = {}
    for u_val, v_val in remaining_pairs:
        u_key = _round_key(u_val)
        v_key = _round_key(v_val)
        u_vals_remaining.setdefault(u_key, u_val)
        grouped.setdefault(u_key, {})[v_key] = v_val

    for u_key in sorted(grouped.keys()):
        u_val = u_vals_remaining[u_key]
        v_map = grouped[u_key]
        v_keys_sorted = sorted(v_map.keys())
        alphas = [10**u_val]
        betas = [10**v_map[key] for key in v_keys_sorted]
        results = test_orchid.evaluate_alphas(
            state,
            alphas,
            betas,
            write_csv=False,
            write_summary=False,
            csv_path=None,
        )
        for v_key, result in zip(v_keys_sorted, results):
            result_map[(u_key, v_key)] = result
    return result_map


def _eval_pairs_memo(
    state: test_orchid.TestIvfState,
    pairs: list[tuple[float, float]],
    cache: dict[tuple[float, float], test_orchid.IvfEvalResult],
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> dict[tuple[float, float], test_orchid.IvfEvalResult]:
    if not pairs:
        return {}
    clamped_pairs: list[tuple[float, float]] = []
    for u_val, v_val in pairs:
        clamped_pairs.append((_clamp(u_val, u_min, u_max), _clamp(v_val, v_min, v_max)))
    unique_pairs = _unique_pairs_preserve(clamped_pairs)

    result_map: dict[tuple[float, float], test_orchid.IvfEvalResult] = {}
    missing: list[tuple[float, float]] = []
    for u_val, v_val in unique_pairs:
        key = (_round_key(u_val), _round_key(v_val))
        cached = cache.get(key)
        if cached is not None:
            result_map[key] = cached
        else:
            missing.append((u_val, v_val))

    if missing:
        new_results = _eval_pairs(state, missing, u_min, u_max, v_min, v_max)
        cache.update(new_results)
        result_map.update(new_results)

    return result_map


def _center_cost(
    result_map: dict[tuple[float, float], test_orchid.IvfEvalResult],
    u_val: float,
    v_val: float,
) -> test_orchid.IvfEvalResult:
    key = (_round_key(u_val), _round_key(v_val))
    return result_map[key]


def _log_neighborhood(
    u: float,
    v: float,
    step: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> list[tuple[float, float]]:
    """Return the center and complete 3x3 log-space neighborhood."""
    # alpha is now an amplitude: halve its log step to preserve fitted indexes.
    u_values = [max(u - step / 2, u_min), u, min(u + step / 2, u_max)]
    v_values = [max(v - step, v_min), v, min(v + step, v_max)]
    return _unique_pairs_preserve(
        [(u, v)]
        + [
            (u_candidate, v_candidate)
            for u_candidate in u_values
            for v_candidate in v_values
        ]
    )


def _ensure_positive_bounds(args: OptimizeIvfConfig) -> None:
    if args.x_min <= 0 or args.y_min <= 0:
        raise ValueError("x_min and y_min must be > 0 for log-space optimization")


def _init_opt_csv(path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "iter,alpha,beta,avg_cost,nprobe,step_log10,grad_log10_alpha,grad_log10_beta,elapsed_sec\n"
        )


def main() -> None:
    config = _parse_args()
    _ensure_positive_bounds(config)

    alpha0 = config.alpha0
    if alpha0 is None:
        alpha0 = math.sqrt(config.x_min * config.x_max)
    beta0 = config.beta0
    if beta0 is None:
        beta0 = math.sqrt(config.y_min * config.y_max)

    alpha0 = _clamp(alpha0, config.x_min, config.x_max)
    beta0 = _clamp(beta0, config.y_min, config.y_max)

    u_min = math.log10(config.x_min)
    u_max = math.log10(config.x_max)
    v_min = math.log10(config.y_min)
    v_max = math.log10(config.y_max)

    u = _clamp(math.log10(alpha0), u_min, u_max)
    v = _clamp(math.log10(beta0), v_min, v_max)

    test_orchid.reset_tracker()
    loaded = test_orchid.load_ivf_arrays(config, load_pred_mt=True, write_pred_mt=True)
    state = test_orchid.prepare_state_from_loaded(config, loaded)
    _init_opt_csv(config.csv)

    t0 = time.perf_counter()
    best: tuple[float, float, float] | None = None
    prev_pos: tuple[float, float] | None = None
    prev_prev_pos: tuple[float, float] | None = None
    no_improve = 0
    step_log10 = config.step_log10
    pair_cache: dict[tuple[float, float], test_orchid.IvfEvalResult] = {}

    for iter_idx in range(config.max_iters):
        u_minus = max(u - step_log10 / 2, u_min)
        u_plus = min(u + step_log10 / 2, u_max)
        v_minus = max(v - step_log10, v_min)
        v_plus = min(v + step_log10, v_max)

        candidates = _log_neighborhood(
            u, v, step_log10, u_min, u_max, v_min, v_max
        )
        result_map = _eval_pairs_memo(
            state, candidates, pair_cache, u_min, u_max, v_min, v_max
        )
        alpha = 10**u
        beta = 10**v
        center = _center_cost(result_map, u, v)

        f_u_minus = _center_cost(result_map, u_minus, v).avg_cost
        f_u_plus = _center_cost(result_map, u_plus, v).avg_cost
        delta_u = u_plus - u_minus
        grad_u = (f_u_plus - f_u_minus) / delta_u if delta_u > 0 else 0.0

        f_v_minus = _center_cost(result_map, u, v_minus).avg_cost
        f_v_plus = _center_cost(result_map, u, v_plus).avg_cost
        delta_v = v_plus - v_minus
        grad_v = (f_v_plus - f_v_minus) / delta_v if delta_v > 0 else 0.0

        best_uv = min(
            candidates,
            key=lambda uv: _center_cost(result_map, uv[0], uv[1]).avg_cost,
        )
        best_cost = _center_cost(result_map, best_uv[0], best_uv[1]).avg_cost

        if best is None or best_cost < best[2]:
            best = (10**best_uv[0], 10**best_uv[1], best_cost)

        improved_local = best_cost < center.avg_cost * (1 - config.tol)
        if improved_local:
            u_new, v_new = best_uv
            no_improve = 0
        else:
            u_new, v_new = u, v
            no_improve += 1

        with open(config.csv, "a") as f:
            elapsed = time.perf_counter() - t0
            f.write(
                f"{iter_idx},{alpha:.6g},{beta:.6g},{center.avg_cost:.6g},"
                f"{center.nprobe},{step_log10:.6g},{grad_u:.6g},{grad_v:.6g},"
                f"{elapsed:.6f}\n"
            )

        oscillating = False
        if prev_prev_pos is not None:
            oscillating = (
                abs(u_new - prev_prev_pos[0]) <= 1e-9
                and abs(v_new - prev_prev_pos[1]) <= 1e-9
            )
        if oscillating:
            step_log10 *= config.step_decay
            no_improve = 0
            if step_log10 < config.min_step_log10:
                break
            prev_prev_pos = prev_pos
            prev_pos = (u, v)
            continue

        moved = (abs(u_new - u) > 1e-12) or (abs(v_new - v) > 1e-12)
        if not moved:
            no_improve = config.patience

        if no_improve >= config.patience:
            step_log10 *= config.step_decay
            no_improve = 0
            if step_log10 < config.min_step_log10:
                break

        u, v = u_new, v_new
        prev_prev_pos = prev_pos
        prev_pos = (u, v)

    if best is not None:
        alpha_best, beta_best, cost_best = best
        elapsed = time.perf_counter() - t0
        print(
            f"Best: alpha={alpha_best:.6g}, beta={beta_best:.6g}, cost={cost_best:.6g}"
        )
        print(f"elapsed_sec={elapsed:.3f}")

    analytics_dir = "analytics"
    os.makedirs(analytics_dir, exist_ok=True)
    csv_name = os.path.basename(config.csv)
    summary_path = os.path.join(analytics_dir, f"optimize_{csv_name}.log")
    with open(summary_path, "w") as summary_file:
        test_orchid.tracker.print_summary(file=summary_file)


if __name__ == "__main__":
    main()
