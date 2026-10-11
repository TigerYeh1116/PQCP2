# GPU 大批次與 Neural Accelerators 實測

日期：2026-09-30。正式環境為 Python 3.9.6／PyTorch 2.8.0。

## 已查證的支援方向

Apple 的 M6 規格分別列出 GPU Neural Accelerators 與 Neural Engine：
[Mac mini M6 規格](https://support.apple.com/en-us/128108)。
本次關注 GPU 內的矩陣運算加速單元。

Apple 官方技術演講說明，PyTorch、MPS/MPSGraph 已可透過框架使用 GPU Neural Accelerators；
自訂 Metal TensorOps 則提供更直接的控制：
[Accelerate your machine learning workloads with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/)。
這是框架能力說明，不足以證明本專案的每個 FP32 矩陣形狀、目前 PyTorch 版本都命中加速單元。
需使用 Metal system trace／硬體分析工具確認；MPS 裝置標籤與計時本身不是證據。

現有 DFT 連續相關計算已使用 PyTorch matmul，但長度只有 44～94，矩陣較窄。
另外每 25 步的 observe 包含同步讀回、Python 排序及精確驗證。
本次實測已包含上述觀測開銷。同步的連續運算區間仍包含 CPU 提交時間，
不是硬體 GPU busy time。本機 `xcrun --find xctrace` 找不到工具，因此尚未取得
Metal 硬體追蹤，不能聲稱已確認 Neural Accelerators 使用率。

## 測試工具

1. `experiments/benchmark_gpu_batch.py`：固定以前次 4096／2048 為基準，比較 1／2／4／8 倍。
   保留每 25 步觀測，分開記錄初始化、同步完成的連續運算、觀測耗時，並記錄 MPS driver 記憶體抽樣。
   這是單階段測試，不代表 GPU 利用率或完整搜尋的新解速度。
2. `experiments/benchmark_neural_matmul.py`：比較現有 FP32 DFT、合併 cosine/sine 並補零對齊的
   FP32 GEMM，以及同樣配置的實驗 FP16 前向 DFT GEMM；另比較不補零合併及只改 FP16 的版本。
   後者仍以 FP32 計算功率與逆變換，但會改變連續數值誤差，不能直接替換正式預設。
   每個配置記錄相關值／梯度誤差及前向反向計時。此工具不執行正式搜尋，也不寫入結果檔。

3. `experiments/benchmark_hardware_resources.py`：完整 GPU＋C＋Z3 流程對照，
   `--reference previous` 固定為 4096／2048、6 C threads，候選 batch 可明確指定。

## 採用的配置

符合既有 MPS／12 核 CPU／32 GiB 偵測條件時，正式預設更新為：

- L ≤ 58：batch 8192（前次 4096）。
- 58 < L ≤ 94：batch 16384（前次 2048）。
- L > 94：維持 2048，不把量測結果外推到更長序列。
- C 維持 6 threads；矩陣運算保留原本 FP32 寫法。

搜尋方法、精確觀測、verifier、去重及正式 L.txt 未因本次工作更動。
手動 `--torch-batch-size` 仍有最高優先權。這是增加 GPU 工作量的配置，尚未證明新解率最佳。

## 大批次分階段結果

每個配置執行 200 個連續步驟，每 25 步完整觀測，計入初始化與觀測。
沒有並行 C／Z3 工作。本表為單次短測，首輪亦可能包含暖機成本。

| L | batch | 總秒數 | lane-steps/s |
| --- | ---: | ---: | ---: |
| 44 | 4096 | 2.070 | 395,665 |
| 44 | 8192 | 2.134 | 767,639 |
| 44 | 16384 | 4.069 | 805,286 |
| 44 | 32768 | 7.989 | 820,330 |
| 68 | 2048 | 1.956 | 209,433 |
| 68 | 4096 | 2.615 | 313,230 |
| 68 | 8192 | 3.697 | 443,200 |
| 68 | 16384 | 6.296 | 520,463 |

L44 在 8192 之後收益已明顯遞減。L68 在 16384 仍有收益，因此採用不同大小。
另補驗 L94／16384：200 步及完整觀測通過，8.978 秒、364,988 lane-steps/s。
該次 driver 記憶體抽樣約 2.18 GiB，沒有 OOM；這包含 driver 配置／快取，並非整個程序的峰值記憶體。

## 矩陣寫法與 FP16 結果

L44／L68、batch 4096／8192／16384，測量 DFT 前向與反向各 500 次；
之後以 batch 8192／16384 補測替代寫法。所有浮點結果與梯度均為有限值。

補測的耗時相對於原本 FP32（越小越好）：

| 寫法 | L44 B8192 | L44 B16384 | L68 B8192 | L68 B16384 |
| --- | ---: | ---: | ---: | ---: |
| 原本 FP32 | 1.000 | 1.000 | 1.000 | 1.000 |
| 補零合併 FP32 | 1.204 | 1.330 | 1.258 | 1.329 |
| 補零合併 FP16 | 1.303 | 1.501 | 1.308 | 1.575 |
| 不補零合併 FP32 | 1.082 | 1.138 | 1.112 | 1.148 |
| 不補零合併 FP16 | 1.224 | 1.317 | 1.207 | 1.345 |
| 原矩陣形狀 FP16 | 1.147 | 1.072 | 1.097 | 1.074 |

本機 PyTorch 2.8.0、這些形狀與測量方式下，替代寫法皆較慢。
FP16 的 PACF 最大絕對誤差約 0.0183，梯度相對 L2 誤差最高約 0.0823%；
不作正式預設。這不能推出 Neural Accelerators 一般無效，也未測試自訂 TensorOps 或新版本框架。
GPU 上兩種 FP32 合併寫法的相關值／梯度皆通過容差檢查。
CPU 測試涵蓋 8 個長度與兩種對齊方式；58／90 使用合成 target 只驗證矩陣恆等式，
不表示這些長度有可行搜尋 profile。

## 完整搜尋比較

seed=123，每輪 60 秒搜尋預算；CPU 6 threads、Z3 開啟，其餘搜尋參數一致。
每個配置獨立複製相同 inventory，只供去重，不作種子。總時間包含初始化與收尾。

| L | batch | 總 wall 秒 | GPU steps | GPU lane-steps/s | 新增非等價解 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 44 | 4096 | 63.44 | 503 | 32,475 | 2 |
| 44 | 8192 | 69.90 | 404 | 47,348 | 2 |
| 68 | 2048 | 63.33 | 689 | 22,282 | 1 |
| 68 | 16384 | 70.27 | 604 | 140,833 | 0 |

GPU lane-steps/s 以 `formation.steps × batch / 總 wall` 計算。
L44／L68 分別約為前次配置的 1.46／6.32 倍；單輪深度（steps）均減少。
新解數未改善；一個 seed 的短測不能判定平均新解時間，亦不能將零命中宣稱為無解。
四個隔離結果檔重新用獨立 C 驗證器驗證，811／811／3／2 筆全部通過。
正式 L44／L68 的 SHA-256 與測試開始快照一致。

回歸測試：1098 passed、12 skipped、1 warning，54.44 秒。
仍明確略過交接已知缺少 benchmark 模組的兩個測試檔；warning 為既有 NumPy 提示。

## 報告與重跑

本次核准已恢復，以上 GPU 測試均實際執行完成。原始報告在 Git ignore 的目錄：
`results/gpu_batch_probe`、`results/gpu_batch_boundary`、`results/neural_matmul_probe`、
`results/neural_matmul_followup`、`results/gpu_batch_full_pipeline`。
重跑請指定尚不存在的 output 目錄，例如：

```sh
.venv/bin/python experiments/benchmark_gpu_batch.py --output results/gpu_batch_repeat
.venv/bin/python experiments/benchmark_neural_matmul.py --output results/neural_matmul_repeat --steps 500
.venv/bin/python experiments/benchmark_hardware_resources.py --output results/gpu_pipeline_repeat --reference previous --candidate-batch-short 8192 --candidate-batch-long 16384 --seeds 123 456 789
```

測試會提高 GPU 負載，宜停止其他搜尋後依序執行。
