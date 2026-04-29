# lazy_bp_product_eval.py 用法说明

## 0. 最懒入口：直接从模型和 test_loader 跑全部默认场景

```python
from lazy_bp_product_eval import run_lazy_product_eval, default_product_scenarios

summary, scenario_dfs = run_lazy_product_eval(
    model=model,
    test_loader=test_loader,
    device="cuda",
    scenarios=default_product_scenarios(seed=42),
    out_dir="eval_product_all",
)
```

输出：
- `eval_product_all/raw_predictions.csv`
- `eval_product_all/predictions_<scenario>.csv`
- `eval_product_all/metrics_<scenario>.json`
- `eval_product_all/scenario_metrics_summary.csv`

## 1. 如果已经有 raw prediction CSV，后处理即可

CSV 至少需要：
- `id_clean`
- `t_bp_ms`
- `sleep`
- `y_true_sbp`, `y_true_dbp`
- `y_pred_sbp_raw`, `y_pred_dbp_raw`

```python
from lazy_bp_product_eval import run_product_eval_from_saved_raw, default_product_scenarios

summary, scenario_dfs = run_product_eval_from_saved_raw(
    raw_csv="raw_predictions.csv",
    emb_path=None,
    scenarios=default_product_scenarios(seed=42),
    out_dir="eval_from_raw_csv",
)
```

命令行版本：

```bash
python lazy_bp_product_eval.py \
  --raw_csv raw_predictions.csv \
  --out_dir eval_from_raw_csv \
  --scenarios all
```

## 2. 场景 A：4+3，跨白天/黑夜，小时级 BP

```python
from lazy_bp_product_eval import ProductScenario

scenario = ProductScenario(
    name="4p3_daynight_hourly",
    calib_total=7,
    support_n=4,
    update_n=3,
    calib_strategy="min_gap",
    min_gap_minutes=30,
    sleep_quota={0: 4, 1: 3},
    eval_sleep="all",
    eval_mode="hourly",
    min_events_per_hour=2,
)
```

适合：产品明确要求校准点横跨白天与夜间。

## 3. 场景 B：4+3，不强制昼夜比例，小时级 BP

```python
scenario = ProductScenario(
    name="4p3_all_hourly",
    calib_total=7,
    support_n=4,
    update_n=3,
    calib_strategy="quantile",
    min_gap_minutes=30,
    calib_sleep="all",
    eval_sleep="all",
    eval_mode="hourly",
    min_events_per_hour=2,
)
```

适合：产品只要求 7 个 calibration points，但不指定昼夜配额。

## 4. 场景 C：2+2，计算 24h macro 均值/方差

```python
scenario = ProductScenario(
    name="2p2_macro24",
    calib_total=4,
    support_n=2,
    update_n=2,
    calib_strategy="quantile",
    min_gap_minutes=120,
    calib_sleep="all",
    eval_sleep="all",
    eval_mode="macro24",
    macro_window_hours=24,
    min_events_per_macro=6,
)
```

重点看输出中的：
- `bank_macro_sbp_mean_ME/STD/MAE`
- `bank_macro_dbp_mean_ME/STD/MAE`
- `bank_macro_sbp_std_ME/STD/MAE`
- `bank_macro_dbp_std_ME/STD/MAE`

## 5. 场景 D：白天 1+1，小时级 BP

```python
scenario = ProductScenario(
    name="day_1p1_hourly",
    calib_total=2,
    support_n=1,
    update_n=1,
    calib_strategy="min_gap",
    min_gap_minutes=60,
    calib_sleep="day",
    eval_sleep="day",
    eval_mode="hourly",
    min_events_per_hour=2,
)
```

适合：只在白天使用、只评估白天小时级 BP。

## 6. 场景 E：只有一次校准，计算 24h mean

```python
scenario = ProductScenario(
    name="one_calib_macro24",
    calib_total=1,
    support_n=1,
    update_n=0,
    calib_strategy="head",
    calib_sleep="all",
    eval_sleep="all",
    eval_mode="macro24",
    macro_window_hours=24,
    min_events_per_macro=6,
)
```

注意：单点 calibration 下 affine 没有真实 slope 信息，主要看 `bias` 和 `bank`。

## 7. 推荐校准方法选择

| 校准点数量 | 推荐方法 |
|---|---|
| 7 点 / 4+3 | bias + affine + bank 都跑 |
| 4 点 / 2+2 | bank + ridge affine |
| 2 点 / 1+1 | bias + bank，affine 仅作参考 |
| 1 点 | bias + bank，不建议宣传 affine |
| day-only | calib_sleep="day", eval_sleep="day" |
| day/night quota | sleep_quota={0: x, 1: y} |

## 8. 如果模型能多返回 pooled embedding

把模型输出从：

```python
return pred, weight
```

改成：

```python
return pred, weight, pooled
```

其中 `pooled` 是 `[B, D]` event embedding。代码会自动用于 residual-bank 相似度。
