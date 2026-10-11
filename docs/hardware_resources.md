# M6／32 GiB 硬體配置

這次只修改資源配置。搜尋公式、loss、最佳化步數、FKM 初始化、GPU polish、
C 退火、45 秒 GPU formation／15 秒 C cycle、completion、剪枝、驗證與去重方法維持原樣。
增加批次和執行緒會改變候選數量與執行排程，因此相同 seed 不代表會得到相同軌跡。

## 一鍵執行

照常在 VS Code 執行 `main.py`，無須多輸入選項。
程式會先解析實際 GPU 裝置，再偵測 CPU 核心數與總記憶體。

| 配置 | L ≤ 58 GPU batch | 58 < L ≤ 94 GPU batch | hybrid C 執行緒上限 |
| --- | ---: | ---: | ---: |
| 原設定 | 2048 | 1024 | 4 |
| MPS、Apple Silicon、至少 12 CPU 核心與 32 GiB | 8192 | 16384 | 6 |

第二輪大批次量測後已更新上表。L > 94 的實驗長度維持前次 2048 上限，
不將大批次設定外推至未測試的更長序列。
本次大批次與 Neural Accelerators 實驗紀錄見 [GPU 測試](gpu_neural_accelerators.md)。

本機實測為 Apple M6、12 核 CPU（2 Super、4 Performance、6 Efficiency）、12 核 GPU、32 GiB。
保留其餘 CPU 容量供 GPU 指令提交、Python 驗證及背景 completion 使用。
這是保守的資源預設，並非已證明的新解率最佳配置。
記憶體有餘裕不代表應任意擴大所有 queue：本次不改 Z3 queue、MITM 或模型枚舉限制。

CPU／CUDA、舊款較小機器、硬體偵測失敗時沿用舊批次與 C 上限；不會偷偷切換搜尋 backend。
純 C 搜尋不受新 hybrid 上限影響。
library 直接建立 `TorchHybridConfig` 時仍保留原預設，可透過 `c_threads` 明確指定。

## 手動覆寫與回退

`--torch-batch-size` 始終優先於硬體自動配置。
新增 `--hybrid-c-threads`，實際 C 執行緒數仍不超過 `--workers`。
例如重現 L44 的舊配置：

```sh
.venv/bin/python main.py --L 44 --torch-batch-size 2048 --hybrid-c-threads 4
```

L68 舊 batch 為 1024。啟動時會印出實際 batch、C 執行緒與硬體容量。
不需要更改任何 MPS fallback 或記憶體限制環境變數。

## 隔離比較

可使用下列工具重跑相同方法、不同配置的完整比較：

```sh
.venv/bin/python experiments/benchmark_hardware_resources.py --output results/hardware_comparison --seeds 123 456 789 --seconds 60
```

output 必須是尚不存在的目錄。工具先快照正式結果，僅供驗證後去重；不作初始化。
每個配置／長度／seed 使用獨立目錄，交替配置順序並記錄 inventory SHA-256。
60 秒是搜尋預算，總 wall time 包含初始化、驗證及等待正在進行的 completion 收尾。
沒有找到新解的 runs 也必須保留。不能把 GPU kernel 吞吐量直接當成新解速度。

## 2026-09-30 第一輪本機量測（歷史配置 4096／2048）

單獨 kernel 測試（5 步暖機、50 步計時）：批次加倍的 lane-steps/s，
L44 從約 103 萬增至 174 萬，L68 從約 56 萬增至 95 萬。
這不含觀測、驗證與 completion。獨立 C 的 10 秒短測由 4／6／8 執行緒分別得到
約 3374／4493／5120 萬 moves/s；8 執行緒尚未驗證與 GPU 並行時的效果，預設採 6。
原始 kernel/C 資料保存在忽略追蹤的 `results/hardware_tuning/probe.json`。

完整 production 流程，seed=123、每次搜尋預算 60 秒：

| 長度 | 配置 | 總 wall 秒 | GPU steps × batch | C moves | 新增且通過驗證的非等價解 |
| --- | --- | ---: | ---: | ---: | ---: |
| 44 | 舊 | 67.87 | 1283 × 2048 | 1,126,489,047 | 6 |
| 44 | 新 | 75.02 | 1193 × 4096 | 1,471,379,117 | 4 |
| 68 | 舊 | 80.69 | 193 × 1024 | 1,115,248,055 | 0 |
| 68 | 新 | 68.32 | 146 × 2048 | 1,003,210,304 | 0 |

GPU steps 取自 `formation.steps`；batch 是每步平行處理的候選數。
以總 wall time 為分母，新配置 GPU lane-steps 吞吐量約增加 68%／79%，
C moves 吞吐量約增加 18%／6%。各階段時間重疊，不可相加。
這只是兩個長度各一個 seed 的資源健全性檢查，不能推出平均新解時間改善：
L44 本次新增數反而較少，L68 兩者皆未找到；多 seed 長測仍是後續效能判定依據。
L44 隔離結果再次經獨立 C 驗證器完整驗證，兩種配置全部通過。

正式 inventory 未修改。每輪完整 console、metrics、待解 queue 與解答保存在
`results/hardware_tuning/paired/`，彙整為 `report.json`；這些生成檔案維持 Git ignore。

回歸測試：1080 passed、12 skipped、1 warning，48.71 秒。
沿用交接已知的兩個缺失 benchmark 模組排除項，未刪除或隱藏其對應測試；
warning 為既有 NumPy 未安裝提示。新增測試包含資源回退、明確 batch 覆寫、
C 執行緒參數確實傳入 worker、workers 上限及非法參數檢查。
