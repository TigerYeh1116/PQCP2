# 新電腦環境設定與驗證（2026-09-30）

本次環境建立於 macOS arm64、Python 3.9.6；專案來源為 `pytorch` 分支的 `a1f7474`。

## VS Code 執行

1. 用 VS Code 開啟 PQCP2 專案資料夾。
2. 執行命令面板的 **Python: Select Interpreter**，選取專案 `.venv/bin/python`。
   `.vscode/settings.json` 已指定此預設值；若 VS Code 曾記住其他 interpreter，仍需手動選取。
3. 開啟 `main.py`，使用 **Run Python File in Terminal**，依提示輸入長度、seed 與選項。
4. 未指定時間時會持續搜尋，找到結果也會繼續；按 Ctrl+C 停止並等待收尾。

預設仍為 PyTorch、裝置 auto、Z3 開啟、lattice loss。這台 Mac 正常權限下會選到 MPS。
請勿同時啟動多個程序寫入同一個長度的結果檔。

## 重建環境

在專案根目錄執行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

版本清單固定四個主要依賴，並非所有間接依賴的完整 lock。
本次使用既有 Apple clang 21.0.0，已從來源重建 C 搜尋核心、affine 與 rectangle 加速模組、獨立驗證器。
`.venv` 與 `build` 均維持 Git ignore。

## 本機實測

- `pip check`：無相依衝突。
- 正常權限下 MPS 可用，GPU tensor 平方和結果為 140，符合預期。
  沙箱內 MPS 偵測為不可用，不代表此電腦不支援 GPU。
- 完整 pytest collection：1078 tests collected，但兩個既有缺失模組造成 collection errors：
  `experiments/benchmark_mps_acceleration.py`、`experiments/benchmark_torch_mathematical_loss.py`。
- 明確略過對應兩個測試檔後：1066 passed、12 skipped、1 warning，49.15 秒。
- 正常 GPU 權限下補跑 `test_torch_observation.py`、`test_torch_alternating.py`、
  `test_direct_compression.py`：34 passed、2 skipped（CUDA），8.02 秒。
- 獨立 C 驗證正式結果：4.txt 2 筆、44.txt 807 筆、46.txt 66 筆、68.txt 1 筆；
  共 876 筆全部合法。本次未對全部檔案做等價類稽核。
- 隔離 L44 MPS 短測：batch 32、15 秒搜尋預算，GPU 產生 32 個種子，C 同時搜尋；
  completion 實際觸發 Z3，未決 queue 保存，程序正常退出。初始化與收尾可能使總時間超過預算。
- 隔離 L4 無參數互動試跑：輸入長度及預設選項，15 秒後送出 SIGINT（Ctrl+C），正常退出。
  保存 2 筆結果，再經獨立 C 驗證通過；此輪 GPU 種子尚未交接，不能將結果歸功於 GPU。
- 正式結果檔與搜尋演算法未修改。

重跑主要測試：

```sh
.venv/bin/python -m pytest -q --ignore=tests/test_mps_acceleration_benchmark.py --ignore=tests/test_torch_mathematical_benchmark.py
```

環境沿用交接版本且未安裝 NumPy，因此 Torch import 會顯示 NumPy initialization warning。
目前 main/solver 未使用 `.numpy()` 或 `from_numpy`，上述測試及 MPS 搜尋通過；
若未來新增 NumPy 互轉功能，需另行安裝相容 NumPy。

本次暫存紀錄：`/tmp/pqcp2-tests.log`、`/tmp/pqcp2-collection.log`、
`/tmp/pqcp2-env-smoke/`（系統可能清除暫存內容）。
