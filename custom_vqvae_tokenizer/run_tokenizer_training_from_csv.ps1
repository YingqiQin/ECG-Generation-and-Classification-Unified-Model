param(
    [string]$CsvPath = "E:\\your_data\\train.csv",
    [string]$DataRoot = "",
    [string]$OutputDir = ".\\checkpoints\\codebook_custom_csv",
    [string]$LogDir = ".\\log\\codebook_custom_csv",
    [int]$NprocPerNode = 1,
    [int]$BatchSize = 256,
    [int]$Epochs = 100,
    [int]$CodebookSize = 4096,
    [int]$CodeDim = 64,
    [int]$InputSize = 12000,
    [int]$ChannelIndex = 0,
    [ValidateSet("auto", "0", "1")] [string]$ChannelAxis = "auto",
    [ValidateSet("patient", "random")] [string]$SplitMode = "patient",
    [switch]$UseConsistencyLoss,
    [switch]$Normalize
)

$env:OMP_NUM_THREADS = "1"

$argsList = @(
    "--nnodes=1",
    "--nproc_per_node=$NprocPerNode",
    ".\custom_vqvae_tokenizer\train_tokenizer_from_csv.py",
    "--csv_path", $CsvPath,
    "--output_dir", $OutputDir,
    "--log_dir", $LogDir,
    "--batch_size", $BatchSize,
    "--epochs", $Epochs,
    "--input_size", $InputSize,
    "--channel_index", $ChannelIndex,
    "--channel_axis", $ChannelAxis,
    "--split_mode", $SplitMode,
    "--codebook_n_emd", $CodebookSize,
    "--codebook_emd_dim", $CodeDim,
    "--quantize_kmeans_init",
    "--clip_grad", "1.0",
    "--opt", "adamw",
    "--opt_betas", "0.9", "0.99",
    "--warmup_epochs", "10",
    "--save_ckpt_freq", "20",
    "--num_workers", "8"
)

if ($DataRoot -ne "") {
    $argsList += @("--data_root", $DataRoot)
}

if ($UseConsistencyLoss) {
    $argsList += "--use_consistency_loss"
}

if ($Normalize) {
    $argsList += "--normalize"
}

torchrun @argsList
