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
mode_inference
    Run CASCADE forward inference and save outputs to NPZ.
mode_loo_predict
    Run leave-one-out CASCADE prediction across all held-out cells.
main
    Parse CLI arguments and dispatch to inference or loo-predict mode.


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
    """ Fail with an actionable message when ruamel.yaml is missing.

    cascade2p/config.py does 'from pip._internal import main as pip' at module
    level, then calls pip.main(...) in its ImportError handler for ruamel.
    pip is bound to a function there, not the module, so the handler raises
    AttributeError and buries the real cause: ruamel.yaml is not installed.
    """

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


def mode_inference(args):
    """ Run CASCADE forward inference on dF/F traces and save results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with fields: input, output, model.
    """
    _check_ruamel()
    import cascade2p.cascade as cascade

    data = np.load(args.input, allow_pickle=True)
    dff = data['dff'].astype(np.float32)
    fs = float(data['fs'])
    n_cells = dff.shape[0]

    model_name = getattr(args, 'model', None) or 'Global_EXC_30Hz_smoothing50ms_causalkernel'
    print('[cascade-subprocess] inference  n_cells={}  fs={:.1f}  model={}'.format(
        n_cells, fs, model_name))

    t0 = time.time()
    import os
    model_folder = os.path.join(os.path.dirname(os.path.dirname(cascade.__file__)), "Pretrained_models")
    probs = cascade.predict(model_name, dff, model_folder=model_folder, verbosity=1)
    elapsed = time.time() - t0
    print('[cascade-subprocess] Finished in {:.1f}s.'.format(elapsed))

    spikes = []
    for i in range(n_cells):
        spikes.append(_probs_to_spikes(np.nan_to_num(probs[i], nan=0.0), fs))

    np.savez(
        args.output,
        cascade_probs=probs.astype(np.float32),
        cascade_spikes=np.array(spikes, dtype=object),
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


def main():

    parser = argparse.ArgumentParser(
        description='CASCADE subprocess helper (run in cascade conda env)'
    )
    parser.add_argument('--mode', required=True,
                        choices=['inference', 'loo-predict'],
                        help='Operation mode')

    parser.add_argument('--input',  help='Input NPZ path (inference mode)')
    parser.add_argument('--output', required=True, help='Output NPZ path')
    parser.add_argument('--model',  default=None,
                        help='CASCADE model name (inference mode, optional)')
    parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'],
                        help='Hardware device for inference: cpu or gpu (default: gpu)')

    parser.add_argument('--raster-cells', dest='raster_cells',
                        help='Path to raster_cells.npz (loo-predict mode)')
    parser.add_argument('--loo-models-dir', dest='loo_models_dir',
                        help='Directory containing CASCADE LOO model folders (loo-predict mode)')

    args = parser.parse_args()

    if args.mode == 'inference':
        if not args.input:
            parser.error('--input is required for inference mode')
        mode_inference(args)
    elif args.mode == 'loo-predict':
        if not args.raster_cells or not args.loo_models_dir:
            parser.error('--raster-cells and --loo-models-dir are required for loo-predict mode')
        mode_loo_predict(args)


if __name__ == '__main__':
    main()
