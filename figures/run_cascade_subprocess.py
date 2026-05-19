#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
            print(f"[cascade-subprocess] GPU(s) visible: {[g.name for g in gpus]}")
        else:
            print("[cascade-subprocess] WARNING: no GPU visible to TensorFlow — "
                  "running on CPU. If GPU was intended, check CUDA/driver install.")
    except Exception as _gpu_exc:
        print(f"[cascade-subprocess] Could not configure GPU: {_gpu_exc}")

try:
    import keras.engine.input_layer as _kil
    _orig_il_init = _kil.InputLayer.__init__

    def _patched_il_init(self, *args, **kwargs):
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


#   'peaks'     : find local maxima above `height` with a minimum inter-peak
#                 distance of 50 ms (original CASCADE behaviour).
#   'threshold' : return every frame whose probability exceeds `height`,
#                 equivalent to the frame-by-frame threshold used by OASIS.
CASCADE_SPIKE_DETECTION = 'peaks'


def _probs_to_spikes(probs, fs, height=0.5):

    if CASCADE_SPIKE_DETECTION == 'threshold':
        return np.where(probs > height)[0] / fs

    min_dist = max(1, int(0.05 * fs))
    peaks, _ = find_peaks(probs, height=height, distance=min_dist)
    return peaks / fs


def mode_inference(args):

    import cascade2p.cascade as cascade

    data = np.load(args.input, allow_pickle=True)
    dff = data['dff'].astype(np.float32)
    fs = float(data['fs'])
    n_cells = dff.shape[0]

    model_name = getattr(args, 'model', None) or 'Global_EXC_30Hz_smoothing50ms_causalkernel'
    print(f"[cascade-subprocess] inference  n_cells={n_cells}  fs={fs:.1f}  model={model_name}")

    t0 = time.time()
    import os
    model_folder = os.path.join(os.path.dirname(os.path.dirname(cascade.__file__)), "Pretrained_models")
    probs = cascade.predict(model_name, dff, model_folder=model_folder, verbosity=1)
    elapsed = time.time() - t0
    print(f"[cascade-subprocess] finished in {elapsed:.1f}s")

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
    print(f"[cascade-subprocess] saved -> {args.output}")


def mode_loo_predict(args):

    import cascade2p.cascade as cascade

    if not os.path.exists(args.raster_cells):
        raise FileNotFoundError(f"raster_cells NPZ not found: {args.raster_cells}")

    data = np.load(args.raster_cells, allow_pickle=False)
    n_cells = int(data['n_cells'])
    loo_dir = args.loo_models_dir
    print(f"[cascade-subprocess] loo-predict  n_cells={n_cells}  loo_dir={loo_dir}")

    preds = {'n_cells': np.int32(n_cells)}

    for i in range(n_cells):
        ds_raw = data[f'dataset_{i}'].item()
        ds = ds_raw.decode() if hasattr(ds_raw, 'decode') else str(ds_raw)
        fs = float(data[f'fs_{i}'])
        dff = data[f'dff_{i}'].astype(np.float32)

        model_path = os.path.join(loo_dir, ds)
        if not os.path.isfile(os.path.join(model_path, 'config.yaml')):
            print(f"  Cell {i} ({ds}): no LOO model found, skipping.")
            preds[f'pred_spikes_{i}'] = np.array([], dtype=np.float64)
            preds[f'dataset_{i}'] = data[f'dataset_{i}']
            preds[f'cell_idx_{i}'] = data[f'cell_idx_{i}']
            continue

        print(f"  Cell {i} ({ds})  fs={fs:.1f} Hz  n_frames={len(dff)} ...")
        dff_2d = dff[np.newaxis, :]

        probs_2d = cascade.predict(ds, dff_2d, model_folder=loo_dir, verbosity=0)
        probs = np.nan_to_num(probs_2d[0], nan=0.0)
        spk = _probs_to_spikes(probs, fs)
        print(f"    -> {len(spk)} spikes detected")

        preds[f'pred_spikes_{i}'] = spk.astype(np.float64)
        preds[f'dataset_{i}'] = data[f'dataset_{i}']
        preds[f'cell_idx_{i}'] = data[f'cell_idx_{i}']

    np.savez(args.output, **preds)
    print(f"[cascade-subprocess] saved -> {args.output}")


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
