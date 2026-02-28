param(
    [string]$DataPath = ".\\custom_vqvae_tokenizer\\data_h5",
    [string]$OutputDir = ".\\checkpoints\\codebook_custom",
    [string]$LogDir = ".\\log\\codebook_custom",
    [int]$NprocPerNode = 1,
    [int]$BatchSize = 256,
    [int]$Epochs = 100,
    [int]$CodebookSize = 4096,
    [int]$CodeDim = 64,
    [switch]$UseConsistencyLoss
)

$env:OMP_NUM_THREADS = "1"

$argsList = @(
    "--nnodes=1",
    "--nproc_per_node=$NprocPerNode",
    "codebook_training.py",
    "--data_path", $DataPath,
    "--output_dir", $OutputDir,
    "--log_dir", $LogDir,
    "--codebook_n_emd", $CodebookSize,
    "--codebook_emd_dim", $CodeDim,
    "--quantize_kmeans_init",
    "--batch_size", $BatchSize,
    "--clip_grad", "1.0",
    "--opt", "adamw",
    "--opt_betas", "0.9", "0.99",
    "--warmup_epochs", "10",
    "--epochs", $Epochs,
    "--save_ckpt_freq", "20",
    "--input_size", "12000"
)

if ($UseConsistencyLoss) {
    $argsList += "--use_consistency_loss"
}

torchrun @argsList
