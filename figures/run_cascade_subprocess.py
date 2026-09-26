# -*- coding: utf-8 -*-
"""
figures/run_cascade_subprocess.py

CASCADE subprocess helper -- runs spike inference inside the cascade conda environment.

Functions
---------
_patched_il_init
    Keras InputLayer patch for batch_shape compatibility.
_check_ruamel
    Fail with an actionable message when ruamel.yaml is missing.
_probs_to_spikes
    Convert CASCADE probability trace to spike times.
_predict_block
    Run cascade.predict on equal-length traces, optionally in chunks.
mode_inference
    Run CASCADE forward inference and save outputs to NPZ.
mode_loo_predict
    Run leave-one-out CASCADE prediction across all held-out cells.
mode_loo_train
    Train one leave-one-out CASCADE model from a list of ground-truth datasets.
main
    Parse CLI arguments and dispatch to inference, loo-train, or loo-predict mode.


DMM, March 2026
"""

import argparse
import os
import sys
import time
import numpy as np
from scipy.signal import find_peaks

_device_arg = 'gpu'
if '--device' in sys.argv:
    _dev_idx = sys.argv.index('--device')
    if _dev_idx + 1 < len(sys.argv):
        _device_arg = sys.argv[_dev_idx + 1]

if _device_arg == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for _gpu in gpus:
                tf.config.experimental.set_memory_growth(_gpu, True)
            print('[cascade-subprocess] GPU(s) visible: {}.'.format([g.name for g in gpus]))
        else:
            print('[cascade-subprocess] WARNING: no GPU visible to TensorFlow -- '
                  'running on CPU. If GPU was intended, check CUDA/driver install.')
    except Exception as _gpu_exc:
        print('[cascade-subprocess] Could not configure GPU: {}.'.format(_gpu_exc))

try:
    import keras.engine.input_layer as _kil
    _orig_il_init = _kil.InputLayer.__init__

    def _patched_il_init(self, *args, **kwargs):
        """ Remap batch_shape to batch_input_shape for older Keras compatibility. """
        if 'batch_shape' in kwargs and 'batch_input_shape' not in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        _orig_il_init(self, *args, **kwargs)

    _kil.InputLayer.__init__ = _patched_il_init
except Exception as _e:
    pass

try:
    import keras
    from keras.mixed_precision.policy import Policy as _Policy
    _custom = keras.utils.get_custom_objects()
    if 'DTypePolicy' not in _custom:
        _custom['DTypePolicy'] = _Policy
except Exception as _e:
    pass


# CASCADE_SPIKE_DETECTION controls how probability traces are converted to spike times.
# 'peaks'     : find local maxima above height with min inter-peak distance of 50 ms.
# 'threshold' : return every frame whose probability exceeds height.
CASCADE_SPIKE_DETECTION = 'peaks'


def _probs_to_spikes(probs, fs, height=0.5):
    """ Convert CASCADE probability trace to spike times in seconds.

    Parameters
    ----------
    probs : np.ndarray
        Per-frame spike probability, shape (n_frames,).
    fs : float
        Sampling rate in Hz.
    height : float, optional
        Detection threshold (probability units).

    Returns
    -------
    spike_times : np.ndarray
        Spike times in seconds.
    """
    if CASCADE_SPIKE_DETECTION == 'threshold':
        return np.where(probs > height)[0] / fs

    min_dist = max(1, int(0.05 * fs))
    peaks, _ = find_peaks(probs, height=height, distance=min_dist)
    return peaks / fs


def _check_ruamel():

    try:
        import ruamel.yaml  # noqa: F401
        return
    except ImportError:
        pass

    print('[cascade-subprocess] ruamel.yaml is missing from this environment.\n'
          '  CASCADE needs it, and its own auto-install is broken on pip 10+,\n'
          '  so the traceback you would otherwise see points at pip, not here.\n'
          '  Fix with:  conda run -n {} pip install "ruamel.yaml<0.18"'.format(
              os.environ.get('CONDA_DEFAULT_ENV', 'cascade')),
          file=sys.stderr)
    sys.exit(2)


def _predict_block(cascade, model_name, dff, model_folder, max_per_call=None):
    """ Run cascade.predict on equal-length traces, optionally in chunks.

    Parameters
    ----------
    cascade : module
        The cascade2p.cascade module.
    model_name : str
        CASCADE model name.
    dff : np.ndarray
        Traces of equal length, shape (n_cells, n_frames).
    model_folder : str
        Folder holding the model.
    max_per_call : int or None, optional
        Split into calls of at most this many cells.

    Returns
    -------
    probs : np.ndarray
        Spike probabilities, shape (n_cells, n_frames).
    elapsed : float
        Time spent in cascade.predict, in seconds.
    """
    n_cells = dff.shape[0]
    if max_per_call and n_cells > max_per_call:
        n_chunks = int(np.ceil(n_cells / max_per_call))
        chunks = np.array_split(dff, n_chunks, axis=0)
        print('[cascade-subprocess] n_cells={} exceeds max_cells_per_call={}; '
              'splitting into {} calls of ~{} cells.'.format(
                  n_cells, max_per_call, n_chunks, chunks[0].shape[0]))
        probs_parts = []
        elapsed = 0.0
        for ci, chunk in enumerate(chunks):
            t0 = time.time()
            part = cascade.predict(model_name, chunk, model_folder=model_folder, verbosity=1)
            elapsed += time.time() - t0
            print('[cascade-subprocess]   chunk {}/{} ({} cells) done.'.format(
                ci + 1, n_chunks, chunk.shape[0]))
            probs_parts.append(part)
        return np.concatenate(probs_parts, axis=0), elapsed
    t0 = time.time()
    probs = cascade.predict(model_name, dff, model_folder=model_folder, verbosity=1)
    return probs, time.time() - t0


def mode_inference(args):
    """ Run CASCADE forward inference on dF/F traces and save results.

    The input NPZ holds either 'dff', a 2-D array of equal-length traces, or
    'dff_list', an object array of 1-D traces of any lengths. With 'dff_list'
    the traces are predicted in groups of equal length, so every trace is used
    in full, and 'cascade_probs' is saved as an object array in input order.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with fields: input, output, model, and
        optionally max_cells_per_call and model_folder.
    """
    _check_ruamel()
    import cascade2p.cascade as cascade

    with np.load(args.input, allow_pickle=True) as data:
        fs = float(data['fs'])
        ragged = 'dff_list' in data.files
        if ragged:
            traces = [np.asarray(t, dtype=np.float32) for t in data['dff_list']]
        else:
            dff = data['dff'].astype(np.float32)
            traces = list(dff)
    n_cells = len(traces)

    model_name = getattr(args, 'model', None) or 'Global_EXC_30Hz_smoothing50ms_causalkernel'
    print('[cascade-subprocess] inference  n_cells={}  fs={:.1f}  model={}'.format(
        n_cells, fs, model_name))

    model_folder = (getattr(args, 'model_folder', None) or
                    os.path.join(os.path.dirname(os.path.dirname(cascade.__file__)),
                                 "Pretrained_models"))
    max_per_call = getattr(args, 'max_cells_per_call', None)

    if ragged:
        groups = {}
        for i, t in enumerate(traces):
            groups.setdefault(len(t), []).append(i)
        print('[cascade-subprocess] {} traces in {} equal-length groups.'.format(
            n_cells, len(groups)))
        probs = [None] * n_cells
        elapsed = 0.0
        for length, idx in sorted(groups.items()):
            block, t_block = _predict_block(cascade, model_name,
                                            np.stack([traces[i] for i in idx]),
                                            model_folder, max_per_call)
            elapsed += t_block
            for row, i in enumerate(idx):
                probs[i] = block[row].astype(np.float32)
    else:
        probs, elapsed = _predict_block(cascade, model_name, dff, model_folder, max_per_call)
        probs = probs.astype(np.float32)
    print('[cascade-subprocess] Finished in {:.1f}s.'.format(elapsed))

    spikes = []
    for i in range(n_cells):
        spikes.append(_probs_to_spikes(np.nan_to_num(probs[i], nan=0.0), fs))

    if ragged:
        probs_out = np.empty(n_cells, dtype=object)
        for i, p in enumerate(probs):
            probs_out[i] = p
    else:
        probs_out = probs
    spikes_out = np.empty(n_cells, dtype=object)
    for i, sp in enumerate(spikes):
        spikes_out[i] = sp
    np.savez(
        args.output,
        cascade_probs=probs_out,
        cascade_spikes=spikes_out,
        cascade_time=np.float64(elapsed),
        fs=np.float32(fs),
    )
    print('[cascade-subprocess] Saved to {}.'.format(args.output))


def mode_loo_predict(args):
    """ Run leave-one-out CASCADE prediction for all held-out cells.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with fields: raster_cells, loo_models_dir, output.
    """
    _check_ruamel()
    import cascade2p.cascade as cascade

    if not os.path.exists(args.raster_cells):
        raise FileNotFoundError('raster_cells NPZ not found: {}'.format(args.raster_cells))

    data = np.load(args.raster_cells, allow_pickle=False)
    n_cells = int(data['n_cells'])
    loo_dir = args.loo_models_dir
    print('[cascade-subprocess] loo-predict  n_cells={}  loo_dir={}'.format(n_cells, loo_dir))

    preds = {'n_cells': np.int32(n_cells)}

    for i in range(n_cells):
        ds_raw = data['dataset_{}'.format(i)].item()
        ds = ds_raw.decode() if hasattr(ds_raw, 'decode') else str(ds_raw)
        fs = float(data['fs_{}'.format(i)])
        dff = data['dff_{}'.format(i)].astype(np.float32)

        model_path = os.path.join(loo_dir, ds)
        if not os.path.isfile(os.path.join(model_path, 'config.yaml')):
            print('  Cell {} ({}): no LOO model found, skipping.'.format(i, ds))
            preds['pred_spikes_{}'.format(i)] = np.array([], dtype=np.float64)
            preds['dataset_{}'.format(i)] = data['dataset_{}'.format(i)]
            preds['cell_idx_{}'.format(i)] = data['cell_idx_{}'.format(i)]
            continue

        print('  Cell {} ({})  fs={:.1f} Hz  n_frames={}...'.format(i, ds, fs, len(dff)))
        dff_2d = dff[np.newaxis, :]

        probs_2d = cascade.predict(ds, dff_2d, model_folder=loo_dir, verbosity=0)
        probs = np.nan_to_num(probs_2d[0], nan=0.0)
        spk = _probs_to_spikes(probs, fs)
        print('    {} spikes detected.'.format(len(spk)))

        preds['pred_spikes_{}'.format(i)] = spk.astype(np.float64)
        preds['dataset_{}'.format(i)] = data['dataset_{}'.format(i)]
        preds['cell_idx_{}'.format(i)] = data['cell_idx_{}'.format(i)]

    np.savez(args.output, **preds)
    print('[cascade-subprocess] Saved to {}.'.format(args.output))


# Marker written after cascade.train_model returns. CASCADE itself never marks a
# model finished (it leaves training_finished at 'Running'), and predict() uses
# whatever .h5 files exist, so a crashed run would otherwise be used silently.
LOO_DONE_MARKER = 'LOO_TRAINING_COMPLETE'


def mode_loo_train(args):
    """ Train one leave-one-out CASCADE model from a list of ground-truth datasets.

    Skips training if the model folder already holds LOO_DONE_MARKER. A model
    folder without it (an interrupted run) is deleted and trained again.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with fields: model_name, loo_models_dir,
        ground_truth_dir, training_datasets (comma separated), sampling_rate,
        smoothing, causal_kernel, noise_levels (comma separated), nr_of_epochs,
        ensemble_size, batch_size, dense_expansion.
    """
    import shutil
    _check_ruamel()
    import cascade2p.cascade as cascade

    model_path = os.path.join(args.loo_models_dir, args.model_name)
    if os.path.isfile(os.path.join(model_path, LOO_DONE_MARKER)):
        print('[cascade-subprocess] loo-train  {} already trained.'.format(args.model_name))
        return
    if os.path.isdir(model_path):
        print('[cascade-subprocess] loo-train  removing unfinished {}.'.format(model_path))
        shutil.rmtree(model_path)
    os.makedirs(args.loo_models_dir, exist_ok=True)

    training = [d for d in args.training_datasets.split(',') if d]
    cfg = {
        'model_name':        args.model_name,
        'sampling_rate':     float(args.sampling_rate),
        'training_datasets': training,
        'noise_levels':      [int(n) for n in args.noise_levels.split(',')],
        'smoothing':         float(args.smoothing),
        'causal_kernel':     int(args.causal_kernel),
        'nr_of_epochs':      int(args.nr_of_epochs),
        'ensemble_size':     int(args.ensemble_size),
        'batch_size':        int(args.batch_size),
        'dense_expansion':   int(args.dense_expansion),
        'verbose':           1,
    }
    print('[cascade-subprocess] loo-train  {}  fs={:.2f}  {} training datasets:\n  {}'.format(
        args.model_name, cfg['sampling_rate'], len(training), '\n  '.join(training)))
    cascade.create_model_folder(cfg, model_folder=args.loo_models_dir)
    t0 = time.time()
    cascade.train_model(args.model_name, model_folder=args.loo_models_dir,
                        ground_truth_folder=args.ground_truth_dir)
    with open(os.path.join(model_path, LOO_DONE_MARKER), 'w') as f:
        f.write('trained in {:.0f} s\n'.format(time.time() - t0))
    print('[cascade-subprocess] loo-train  finished in {:.1f} min.'.format((time.time() - t0) / 60))


def main():

    parser = argparse.ArgumentParser(
        description='CASCADE subprocess helper (run in cascade conda env)'
    )
    parser.add_argument('--mode', required=True,
                        choices=['inference', 'loo-train', 'loo-predict'],
                        help='Operation mode')

    parser.add_argument('--input',  help='Input NPZ path (inference mode)')
    parser.add_argument('--output', help='Output NPZ path (inference and loo-predict modes)')
    parser.add_argument('--model',  default=None,
                        help='CASCADE model name (inference mode, optional)')
    parser.add_argument('--model-folder', dest='model_folder', default=None,
                        help='Folder holding the model (inference mode; default: CASCADE\'s '
                             'Pretrained_models)')
    parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'],
                        help='Hardware device for inference: cpu or gpu (default: gpu)')
    parser.add_argument('--max-cells-per-call', dest='max_cells_per_call',
                        type=int, default=None,
                        help='Inference mode, optional: split into multiple cascade.predict() '
                             'calls of at most this many cells, to avoid a single call\'s input '
                             'tensor exceeding GPU memory. Times are summed across calls. '
                             'Default: no splitting (unchanged single-call behavior).')

    parser.add_argument('--raster-cells', dest='raster_cells',
                        help='Path to raster_cells.npz (loo-predict mode)')
    parser.add_argument('--loo-models-dir', dest='loo_models_dir',
                        help='Directory containing CASCADE LOO model folders '
                             '(loo-train and loo-predict modes)')

    loo = parser.add_argument_group('loo-train mode')
    loo.add_argument('--model-name', dest='model_name')
    loo.add_argument('--ground-truth-dir', dest='ground_truth_dir')
    loo.add_argument('--training-datasets', dest='training_datasets',
                     help='Comma-separated ground-truth dataset folder names')
    loo.add_argument('--sampling-rate', dest='sampling_rate', type=float)
    loo.add_argument('--smoothing', type=float, default=0.2)
    loo.add_argument('--causal-kernel', dest='causal_kernel', type=int, default=0)
    loo.add_argument('--noise-levels', dest='noise_levels', default='2,3,4,5,6,7,8')
    loo.add_argument('--nr-of-epochs', dest='nr_of_epochs', type=int, default=10)
    loo.add_argument('--ensemble-size', dest='ensemble_size', type=int, default=5)
    loo.add_argument('--batch-size', dest='batch_size', type=int, default=8192)
    loo.add_argument('--dense-expansion', dest='dense_expansion', type=int, default=30)

    args = parser.parse_args()

    if args.mode == 'inference':
        if not args.input or not args.output:
            parser.error('--input and --output are required for inference mode')
        mode_inference(args)
    elif args.mode == 'loo-train':
        missing = [a for a in ('model_name', 'loo_models_dir', 'ground_truth_dir',
                               'training_datasets', 'sampling_rate')
                   if getattr(args, a) in (None, '')]
        if missing:
            parser.error('loo-train mode requires: ' +
                         ', '.join('--' + m.replace('_', '-') for m in missing))
        mode_loo_train(args)
    elif args.mode == 'loo-predict':
        if not args.raster_cells or not args.loo_models_dir or not args.output:
            parser.error('--raster-cells, --loo-models-dir and --output are required for '
                         'loo-predict mode')
        mode_loo_predict(args)


if __name__ == '__main__':
    main()
