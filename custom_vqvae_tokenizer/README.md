# Custom VQVAE Tokenizer Training (No official code modification)

This directory contains a standalone workflow for training SIGMA-PPG tokenizer on your own dataset.
It does not modify any official source files.

## Option A) Directly train from `train.csv + npy` (No H5 needed)

Run:

```powershell
.\custom_vqvae_tokenizer\run_tokenizer_training_from_csv.ps1 `
  -CsvPath E:\your_data\train.csv `
  -DataRoot E:\your_data `
  -OutputDir .\checkpoints\codebook_custom_csv `
  -LogDir .\log\codebook_custom_csv `
  -NprocPerNode 1 `
  -BatchSize 256 `
  -Epochs 100 `
  -InputSize 12000 `
  -ChannelIndex 0 `
  -ChannelAxis auto `
  -SplitMode patient `
  -UseConsistencyLoss `
  -Normalize
```

If you want full python command instead of the wrapper script:

```powershell
torchrun --nnodes=1 --nproc_per_node=1 .\custom_vqvae_tokenizer\train_tokenizer_from_csv.py `
  --csv_path E:\your_data\train.csv `
  --data_root E:\your_data `
  --output_dir .\checkpoints\codebook_custom_csv `
  --log_dir .\log\codebook_custom_csv `
  --batch_size 256 `
  --epochs 100 `
  --input_size 12000 `
  --channel_index 0 `
  --channel_axis auto `
  --split_mode patient `
  --codebook_n_emd 4096 `
  --codebook_emd_dim 64 `
  --quantize_kmeans_init `
  --use_consistency_loss `
  --normalize
```

Notes:
- `split_mode=patient` uses `id_clean` to split train/val by patient, reducing leakage.
- Use `split_mode=random` if your CSV does not contain patient id.
- This path still uses official model and engine (`codebook_training.py` logic), only data loading is replaced.

## Option B) Convert train.csv to H5 first

Expected CSV columns (configurable by args):
- `id_clean`
- `ppg_path`
- `i0`
- `i1`
- `n_ppg` (optional)

Expected npy shape:
- Prefer `(T, C)` such as `(T, 8)`.
- Script will extract one channel via `--channel_index`.

Run:

```powershell
python .\custom_vqvae_tokenizer\scripts\build_h5_from_train_csv.py `
  --csv_path E:\your_data\train.csv `
  --output_dir .\custom_vqvae_tokenizer\data_h5 `
  --ppg_path_col ppg_path `
  --i0_col i0 `
  --i1_col i1 `
  --n_ppg_col n_ppg `
  --window_size 12000 `
  --patch_size 100 `
  --channel_index 0 `
  --channel_axis auto `
  --segments_per_file 5000 `
  --normalize
```

Notes:
- `window_size=12000` matches 120s x 100Hz.
- Each segment is padded/truncated to `window_size`.
- Output H5 includes `signals`, `feat_amp`, `feat_skew`, `feat_avg` required by `codebook_training.py`.

## 2) Train tokenizer from H5

Single GPU example:

```powershell
.\custom_vqvae_tokenizer\run_tokenizer_training.ps1 `
  -DataPath .\custom_vqvae_tokenizer\data_h5 `
  -OutputDir .\checkpoints\codebook_custom `
  -LogDir .\log\codebook_custom `
  -NprocPerNode 1 `
  -BatchSize 256 `
  -Epochs 100 `
  -CodebookSize 4096 `
  -CodeDim 64 `
  -UseConsistencyLoss
```

Multi-GPU example (`NprocPerNode=4`):

```powershell
.\custom_vqvae_tokenizer\run_tokenizer_training.ps1 -NprocPerNode 4
```

## 3) Check outputs

- Checkpoints: `checkpoints/codebook_custom/`
- TensorBoard logs: `log/codebook_custom/`

Use TensorBoard:

```powershell
tensorboard --logdir .\log\codebook_custom
```
