import json
from pathlib import Path

import numpy as np
import torch


STATE_COMPAT_PREFIX = 'state_compat_'
SPIKE_METADATA_KEYS = [
    'spike_local_template_id',
    'spike_parent_template_id',
    'spike_template_window_id',
    'spike_match_score',
    'spike_residual_energy',
    'spike_residual_energy_normed',
]


def enabled(ops):
    return bool(ops['settings'].get('state_compat_enabled', False))


def local_templates_enabled(ops):
    return enabled(ops) and (
        bool(ops['settings'].get('state_compat_use_local_templates', False))
        or bool(ops['settings'].get('state_compat_enabled', False))
    )


def protect_same_parent_templates(ops):
    return local_templates_enabled(ops) and bool(
        ops['settings'].get('state_compat_protect_same_parent_templates', True)
    )


def graph_features_enabled(ops):
    return enabled(ops) and bool(
        ops['settings'].get('state_compat_use_scaled_graph_features', False)
    )


def edge_padding_samples(ops):
    return int(round(
        ops['settings'].get('state_compat_template_window_overlap_s', 0.0)
        * ops['fs']
    ))


def recording_end_sample(ops, spike_samples=None):
    n_batches = int(ops.get('Nbatches', 0))
    batch_size = int(ops['settings'].get('batch_size', ops['batch_size']))
    downsampling = int(ops['settings'].get('batch_downsampling', 1))
    end_sample = n_batches * batch_size * downsampling
    if end_sample == 0 and spike_samples is not None and spike_samples.size > 0:
        end_sample = int(spike_samples.max()) + 1
    return end_sample


def spike_samples_from_st0(st0, ops):
    return np.rint(st0[:, 0] * ops['fs']).astype('int64')


def spike_snr_from_tF(tF):
    if isinstance(tF, torch.Tensor):
        return torch.norm(tF, dim=(-2, -1)).cpu().numpy()
    return np.linalg.norm(tF, axis=(-2, -1))


def make_graph_features(tF, st, ops, mode):
    alpha = ops['settings']['state_compat_feature_scale_alpha']
    shape = tuple(tF.shape)

    if isinstance(tF, torch.Tensor):
        F = tF.reshape(tF.shape[0], -1)
        sigma = torch.std(F, dim=0, unbiased=False)
        median_sigma = torch.median(sigma)
        epsilon = max(1e-6, float(alpha * median_sigma.item()))
        feature_scale = torch.clamp(sigma, min=epsilon)
        F_scaled = F / feature_scale.unsqueeze(0)
        feature_scale_out = feature_scale.cpu().numpy()
        Xd_graph_scaled = F_scaled.reshape(shape)
    else:
        F = tF.reshape(tF.shape[0], -1)
        sigma = F.std(axis=0)
        median_sigma = np.median(sigma)
        epsilon = max(1e-6, float(alpha * median_sigma))
        feature_scale_out = np.maximum(sigma, epsilon)
        F_scaled = F / feature_scale_out[np.newaxis, :]
        Xd_graph_scaled = F_scaled.reshape(shape)

    feature_shape_metadata = {
        'mode': mode,
        'input_shape': shape,
        'flat_shape': (shape[0], int(np.prod(shape[1:]))),
        'alpha': alpha,
        'epsilon': epsilon,
        'n_spikes': st.shape[0],
    }
    return Xd_graph_scaled, feature_scale_out, feature_shape_metadata


def store_graph_feature_metadata(ops, mode, feature_scale, meta, scaled_features):
    if 'state_compat' not in ops:
        ops['state_compat'] = {}
    ops['state_compat']['enabled'] = True
    key = 'spikes' if mode == 'spikes' else 'template'
    ops['state_compat'][f'feature_scale_{key}'] = feature_scale
    ops['state_compat'][f'feature_shape_metadata_{key}'] = meta
    if (
        ops['settings'].get('state_compat_save_scaled_features', True)
        or ops['settings'].get('state_compat_compute_unit_anisotropy', True)
    ):
        if isinstance(scaled_features, torch.Tensor):
            scaled_features = scaled_features.cpu().numpy()
        ops['state_compat'][f'scaled_graph_features_{key}'] = scaled_features


def make_snr_adaptive_template_windows(st0, clu0, tF0, Wall_parent, ops):
    settings = ops['settings']
    fs = ops['fs']
    target_snr = settings['state_compat_template_target_snr']
    min_spikes = int(settings['state_compat_min_spikes_per_template_window'])
    max_spikes = int(settings['state_compat_max_spikes_per_template_window'])
    min_duration = int(round(settings['state_compat_min_window_duration_s'] * fs))
    max_duration = int(round(settings['state_compat_max_window_duration_s'] * fs))
    max_windows = int(settings['state_compat_max_windows_per_parent'])

    spike_samples = spike_samples_from_st0(st0, ops)
    # st0[:, 2] is KS4's detection amplitude from template_match; use it as
    # the existing per-spike amplitude-over-noise proxy for window sizing.
    spike_snr = np.asarray(st0[:, 2], dtype='float32')
    recording_end = recording_end_sample(ops, spike_samples)
    n_parents = Wall_parent.shape[0]
    spike_to_template_window = np.full(st0.shape[0], -1, dtype='int32')
    rows = []

    def add_window(parent_id, window_id, spike_idx, start_sample, end_sample, snr):
        local_template_id = len(rows)
        rows.append((
            local_template_id, parent_id, window_id, start_sample, end_sample,
            spike_idx.size, snr
        ))
        spike_to_template_window[spike_idx] = local_template_id

    for parent_id in range(n_parents):
        spike_idx = np.flatnonzero(clu0 == parent_id)
        if spike_idx.size == 0:
            continue

        spike_idx = spike_idx[np.argsort(spike_samples[spike_idx])]
        parent_samples = spike_samples[spike_idx]
        median_snr = float(np.median(spike_snr[spike_idx]))
        if (
            (not np.isfinite(median_snr))
            or median_snr < target_snr
            or spike_idx.size < min_spikes
        ):
            continue

        n_required = int(np.ceil((target_snr / median_snr) ** 2))
        n_required = max(n_required, min_spikes)
        n_required = min(n_required, max_spikes)
        n_required = max(n_required, int(np.ceil(spike_idx.size / max_windows)))

        if n_required >= spike_idx.size:
            continue

        chunks = []
        i = 0
        while i < spike_idx.size:
            j = min(i + n_required, spike_idx.size)
            while (
                j < spike_idx.size
                and parent_samples[j-1] - parent_samples[i] < min_duration
            ):
                j += 1
            if max_duration > 0 and parent_samples[j-1] - parent_samples[i] > max_duration:
                k = np.searchsorted(
                    parent_samples, parent_samples[i] + max_duration, side='right'
                )
                if k - i >= min_spikes:
                    j = k
            chunks.append(spike_idx[i:j])
            i = j

        if len(chunks) > max_windows:
            chunks = [
                x for x in np.array_split(spike_idx, max_windows) if x.size > 0
            ]

        if len(chunks) == 1:
            continue

        starts = [0]
        ends = []
        for left, right in zip(chunks[:-1], chunks[1:]):
            boundary = int(
                (spike_samples[left[-1]] + spike_samples[right[0]]) // 2
            )
            ends.append(boundary)
            starts.append(boundary)
        ends.append(recording_end)

        for window_id, idx in enumerate(chunks):
            start = max(0, int(starts[window_id]))
            end = min(recording_end, int(ends[window_id]))
            add_window(parent_id, window_id, idx, start, end, median_snr)

    dtype = [
        ('local_template_id', 'int32'),
        ('parent_id', 'int32'),
        ('window_id', 'int32'),
        ('start_sample', 'int64'),
        ('end_sample', 'int64'),
        ('n_spikes', 'int32'),
        ('median_snr', 'float32'),
    ]
    template_window_table = np.array(rows, dtype=dtype)
    return (
        template_window_table,
        spike_to_template_window,
        template_window_table['parent_id'].copy(),
        template_window_table['window_id'].copy(),
        template_window_table['start_sample'].copy(),
        template_window_table['end_sample'].copy(),
    )


def _estimate_template_from_spikes(st0, tF0, spike_idx, ops):
    iC = ops['iC'].detach().cpu().long()
    tF = tF0.detach().cpu() if isinstance(tF0, torch.Tensor) else torch.from_numpy(tF0)
    spike_templates = torch.from_numpy(st0[spike_idx, 5].astype('int64'))
    chans = iC[:, spike_templates].T.reshape(-1)
    feats = tF[spike_idx].reshape(-1, tF.shape[-1])
    template = torch.zeros((ops['Nchan'], tF.shape[-1]), dtype=tF.dtype)
    template.index_add_(0, chans, feats)
    template /= spike_idx.size
    return template


def build_local_templates(Wall_parent, st0, clu0, tF0, template_window_table, ops):
    spike_samples = spike_samples_from_st0(st0, ops)
    Wall_parent_cpu = Wall_parent.detach().cpu()
    n_parents = Wall_parent_cpu.shape[0]
    n_local = template_window_table.size
    Wall_local = torch.zeros(
        (n_local + n_parents, Wall_parent_cpu.shape[1], Wall_parent_cpu.shape[2]),
        dtype=Wall_parent_cpu.dtype
    )

    for row in template_window_table:
        local_id = int(row['local_template_id'])
        parent_id = int(row['parent_id'])

        start = int(row['start_sample'])
        end = int(row['end_sample'])
        spike_idx = np.flatnonzero(
            (clu0 == parent_id)
            & (spike_samples >= start)
            & (spike_samples < end)
        )
        if spike_idx.size == 0:
            Wall_local[local_id] = Wall_parent_cpu[parent_id]
        else:
            Wall_local[local_id] = _estimate_template_from_spikes(
                st0, tF0, spike_idx, ops
            )

    recording_end = recording_end_sample(ops, spike_samples)
    spike_snr = np.asarray(st0[:, 2], dtype='float32')
    rows = [tuple(row) for row in template_window_table]
    for parent_id in range(n_parents):
        local_id = n_local + parent_id
        spike_idx = np.flatnonzero(clu0 == parent_id)
        median_snr = (
            float(np.median(spike_snr[spike_idx]))
            if spike_idx.size > 0 else np.nan
        )
        rows.append((
            local_id, parent_id, -1, 0, recording_end,
            spike_idx.size, median_snr
        ))
        Wall_local[local_id] = Wall_parent_cpu[parent_id]

    local_template_table = np.array(rows, dtype=template_window_table.dtype)
    return (
        Wall_local,
        local_template_table,
        local_template_table['parent_id'].copy(),
        local_template_table['window_id'].copy(),
        local_template_table['start_sample'].copy(),
        local_template_table['end_sample'].copy(),
    )


def store_template_metadata(
    ops, template_window_table, parent_id, window_id, valid_start, valid_end
):
    if 'state_compat' not in ops:
        ops['state_compat'] = {}
    ops['state_compat']['enabled'] = True
    ops['state_compat']['local_template_table'] = template_window_table
    ops['state_compat']['local_template_parent_id'] = parent_id.astype('int32')
    ops['state_compat']['local_template_window_id'] = window_id.astype('int32')
    ops['state_compat']['local_template_valid_start_sample'] = valid_start.astype('int64')
    ops['state_compat']['local_template_valid_end_sample'] = valid_end.astype('int64')

    ops['state_compat_local_template_table'] = template_window_table
    ops['state_compat_local_template_parent_id'] = parent_id.astype('int32')
    ops['state_compat_local_template_window_id'] = window_id.astype('int32')
    ops['state_compat_local_template_valid_start'] = valid_start.astype('int64')
    ops['state_compat_local_template_valid_end'] = valid_end.astype('int64')


def template_time_mask(valid_start, valid_end, batch_start, n_time, ops,
                       edge_padding, device):
    sample_times = (
        torch.arange(n_time, device=device, dtype=torch.long)
        - int(ops['nt'])
        + int(batch_start)
        - int(ops['nt']) // 2
        + int(ops['nt0min'])
    )
    start = torch.as_tensor(valid_start, device=device, dtype=torch.long).unsqueeze(1)
    end = torch.as_tensor(valid_end, device=device, dtype=torch.long).unsqueeze(1)
    return (
        (sample_times.unsqueeze(0) >= (start - int(edge_padding)))
        & (sample_times.unsqueeze(0) < (end + int(edge_padding)))
    )


def compute_residual_energy(Xres, U, wPCA, iCC, iU, Ucc, stt, amps, tiwave):
    if stt.shape[0] == 0:
        z = torch.zeros((0,), device=Xres.device)
        return z, z
    res = Xres[iCC[:, iU[stt[:, 1:2]]], stt[:, :1] + tiwave]
    residual_energy = torch.linalg.norm(res, dim=(0, 2))
    template_pc = Ucc[:, stt[:, 1]]
    template_wave = torch.einsum('csp,pt->cst', template_pc, wPCA)
    template_wave = template_wave * amps.reshape(1, -1, 1)
    template_energy = torch.linalg.norm(template_wave, dim=(0, 2))
    return residual_energy, residual_energy / (template_energy + 1e-12)


def store_spike_metadata(
    ops, local_template_id, parent_template_id, template_window_id, match_score,
    residual_energy=None, residual_energy_normed=None
):
    ops['spike_local_template_id'] = local_template_id.astype('int32')
    ops['spike_parent_template_id'] = parent_template_id.astype('int32')
    ops['spike_template_window_id'] = template_window_id.astype('int32')
    ops['spike_match_score'] = match_score.astype('float32')
    if residual_energy is not None:
        ops['spike_residual_energy'] = residual_energy.astype('float32')
        ops['spike_residual_energy_normed'] = residual_energy_normed.astype('float32')


def sort_spike_metadata(ops, sorted_idx):
    for key in SPIKE_METADATA_KEYS:
        if key in ops and len(ops[key]) == len(sorted_idx):
            ops[key] = ops[key][sorted_idx]
    if 'state_compat' in ops:
        key = 'scaled_graph_features_template'
        if key in ops['state_compat'] and len(ops['state_compat'][key]) == len(sorted_idx):
            # merging_function re-sorts spikes after final merges; anisotropy
            # consumes these saved graph features in the final spike order.
            ops['state_compat'][key] = ops['state_compat'][key][sorted_idx]


def filter_spike_metadata(ops, kept_spikes):
    out = {}
    for key in SPIKE_METADATA_KEYS:
        if key in ops:
            out[key] = ops[key][kept_spikes]
    return out


def save_spike_metadata(results_dir, ops, kept_spikes):
    results_dir = Path(results_dir)
    for key, value in filter_spike_metadata(ops, kept_spikes).items():
        np.save(results_dir / f'{key}.npy', value)


def compute_unit_anisotropy(scaled_graph_features, final_clu, kept_spikes, ops):
    if isinstance(scaled_graph_features, torch.Tensor):
        F = scaled_graph_features.detach().cpu().numpy()
    else:
        F = scaled_graph_features
    F = F.reshape(F.shape[0], -1)

    clu = final_clu[kept_spikes]
    X = F[kept_spikes]
    n_units = int(clu.max()) + 1
    n_features = X.shape[1]
    cluster_anisotropy = np.zeros(n_units, 'float32')
    cluster_state_axis = np.zeros((n_units, n_features), 'float32')
    cluster_feature_mean = np.zeros((n_units, n_features), 'float32')
    spike_state_coord = np.zeros(kept_spikes.sum(), 'float32')

    for u in range(n_units):
        idx = np.flatnonzero(clu == u)
        if idx.size == 0:
            continue
        Xu = X[idx]
        mu = Xu.mean(axis=0)
        Xc = Xu - mu
        var_total = np.sum(Xc**2) / idx.size
        cluster_feature_mean[u] = mu
        if var_total == 0:
            continue
        _, _, vh = np.linalg.svd(Xc, full_matrices=False)
        pc1 = vh[0]
        coord = Xc @ pc1
        var_pc1 = np.sum(coord**2) / idx.size
        cluster_state_axis[u] = pc1
        cluster_anisotropy[u] = var_pc1 / var_total
        spike_state_coord[idx] = coord

    return cluster_anisotropy, cluster_state_axis, cluster_feature_mean, spike_state_coord


def save_unit_anisotropy(
    results_dir, cluster_anisotropy, cluster_state_axis, cluster_feature_mean,
    spike_state_coord
):
    results_dir = Path(results_dir)
    with open(results_dir / 'cluster_anisotropy.tsv', 'w') as f:
        f.write('cluster_id\tanisotropy\n')
        for i, a in enumerate(cluster_anisotropy):
            f.write(f'{i}\t{a:.8g}\n')
    np.save(results_dir / 'cluster_state_axis.npy', cluster_state_axis)
    np.save(results_dir / 'cluster_feature_mean.npy', cluster_feature_mean)
    np.save(results_dir / 'spike_state_coord.npy', spike_state_coord)


def save_local_template_outputs(results_dir, ops):
    results_dir = Path(results_dir)
    sc = ops.get('state_compat', {})
    table = sc.get('local_template_table', None)
    if table is None:
        return

    np.save(results_dir / 'local_template_parent_id.npy',
            sc['local_template_parent_id'])
    np.save(results_dir / 'local_template_window_id.npy',
            sc['local_template_window_id'])
    np.save(results_dir / 'local_template_valid_start_sample.npy',
            sc['local_template_valid_start_sample'])
    np.save(results_dir / 'local_template_valid_end_sample.npy',
            sc['local_template_valid_end_sample'])

    with open(results_dir / 'local_template_table.tsv', 'w') as f:
        names = [
            'local_template_id', 'parent_id', 'window_id',
            'valid_start_sample', 'valid_end_sample', 'n_spikes', 'median_snr'
        ]
        f.write('\t'.join(names) + '\n')
        for row in table:
            values = [
                row['local_template_id'], row['parent_id'], row['window_id'],
                row['start_sample'], row['end_sample'], row['n_spikes'],
                row['median_snr']
            ]
            f.write('\t'.join(str(v) for v in values) + '\n')


def save_graph_feature_outputs(results_dir, ops, kept_spikes):
    results_dir = Path(results_dir)
    sc = ops.get('state_compat', {})
    if 'feature_scale_spikes' in sc:
        np.save(results_dir / 'feature_scale_spikes.npy', sc['feature_scale_spikes'])
    if 'feature_scale_template' in sc:
        np.save(results_dir / 'feature_scale_template.npy', sc['feature_scale_template'])
    if (
        ops['settings'].get('state_compat_save_scaled_features', True)
        and 'scaled_graph_features_template' in sc
    ):
        F = sc['scaled_graph_features_template'][kept_spikes]
        np.save(results_dir / 'graph_features_scaled.npy', F.reshape(F.shape[0], -1))


def save_state_compat_outputs(results_dir, ops, final_clu, kept_spikes):
    if not enabled(ops):
        return None
    results_dir = Path(results_dir)
    if 'state_compat' not in ops:
        ops['state_compat'] = {}
    ops['state_compat']['enabled'] = True

    with open(results_dir / 'state_compat_enabled.json', 'w') as f:
        json.dump({'state_compat_enabled': True}, f)

    save_spike_metadata(results_dir, ops, kept_spikes)
    save_local_template_outputs(results_dir, ops)
    save_graph_feature_outputs(results_dir, ops, kept_spikes)

    if ops['settings'].get('state_compat_compute_unit_anisotropy', True):
        scaled_features = ops['state_compat']['scaled_graph_features_template']
        cluster_anisotropy, cluster_state_axis, cluster_feature_mean, \
            spike_state_coord = compute_unit_anisotropy(
                scaled_features, final_clu, kept_spikes, ops
            )
        ops['state_compat']['cluster_anisotropy'] = cluster_anisotropy
        ops['state_compat']['cluster_state_axis'] = cluster_state_axis
        ops['state_compat']['cluster_feature_mean'] = cluster_feature_mean
        save_unit_anisotropy(
            results_dir, cluster_anisotropy, cluster_state_axis,
            cluster_feature_mean, spike_state_coord
        )
        return cluster_anisotropy

    return None


def global_local_match_counts(ops, kept_spikes):
    if not enabled(ops) or 'spike_template_window_id' not in ops:
        return None
    window_id = ops['spike_template_window_id'][kept_spikes]
    total = int(window_id.size)
    global_count = int(np.sum(window_id == -1))
    local_count = int(np.sum(window_id != -1))
    return global_count, local_count, total


def strip_disabled_state_compat(ops):
    if ops['settings'].get('state_compat_enabled', False):
        return ops
    out = ops.copy()
    out['settings'] = {
        k: v for k, v in ops['settings'].items()
        if not k.startswith(STATE_COMPAT_PREFIX)
    }
    for key in list(out.keys()):
        if (
            key == 'state_compat'
            or key.startswith(STATE_COMPAT_PREFIX)
            or key in SPIKE_METADATA_KEYS
        ):
            del out[key]
    return out
