# 精確 completion 的效能優化

## 結論與適用範圍

正常 main.py 的預設 L44／L68 分段 completion 已使用：

1. C 原生整數 Gray/PACF scan：取代 Python 每 assignment 的迴圈及
   `bin(...).count('1')`，不修改條件、候選順序或 verifier。
2. MITM 小批次：一份索引最多取 4 個不同 children 後一起原子保存，
   減少重複建表。不是長期保留多份大型索引，原 entry/model cap 不變。

**改善的是壓縮 completion，不是已證明的新 PQCP 端到端產率。**
PyTorch loss、C 退火策略、Z3 encoding、Project 2 定義、正式 L.txt
寫入及等價去重未修改。新增 benchmark／短測使用隔離輸出目錄，不回退
或覆寫正式搜尋結果。工作期間 44.txt 另有新增內容，已原樣保留。

## 實際量測

環境：專案 .venv、Python 3.9.6、macOS arm64、torch 2.8.0。
torch 回報 MPS built=True、available=False，CUDA available=False；
因此不能宣稱量測到使用者 Mac 上的 GPU 利用率或 MPS 整體加速。

修改前的 cProfile：一個 262,144 assignments 的固定 L68 子問題約
311 萬次 Python 呼叫；popcount wrapper 的累積時間占 profile 約 53%。
這包含 profiler 開銷，不以該時間直接計算加速倍率。

### 同樣二元子問題，兩種 scan

四個固定 L68 子問題，各三輪，Python/native 執行次序交替。
每次相同 target/content、10 秒上限、1,048,576 assignments cap。
首次 native 載入時間（若需編譯則包含編譯）另外記錄，不計入其中一側。

- Python：約 0.518–1.030 秒／子問題。
- Native：約 0.0040–0.0082 秒／子問題。
- 12 次 paired 比較的倍率中位數：約 **124.4 倍**。
- 每次 status、rank、assignment 數、examined、游標及 SAT 輸出均一致。

另將先前 38 個固定 L68 子問題全部用正式 native 路徑重算，和保存的
Python 基準逐項比較：全部相同，共 **12,582,912** 個精確方向檢查。
38 個固定 children 都 UNSAT，不能由此推論其 coarse parents 或 L68 無解。

### 包含 MITM 的分段工作

同一個已保存的 L68 coarse 工作，依固定順序完成同樣四個 children。
三種設定各三輪，次序交替；相同 10 秒 slice／4 個 models 上限。
矩形交換停用，避免將隨機搜尋差異混入這項局部比較；不是完整 GPU/C
主搜尋。這四個 children 都完整 UNSAT，父問題因 model cap 仍未解。

|設定|中位耗時|
|---|---:|
|Python scan＋每次 1 個 middle|3.396 秒|
|C scan＋每次 1 個 middle|1.319 秒|
|C scan＋每次最多 4 個 middle|0.366 秒|

合併改善約 **9.28 倍**，耗時減少約 **89.2%**。所有設定產生的前四個
children、順序、結果與最終游標相同。這不是「找到新 PQCP 快 9.28 倍」。
C scan 加速後，逐個模式的大部分時間都花在重建 MITM 索引，這是加入
batching 的量測根據，沒有藉由改 score 或放寬條件製造表面速度。

## 安全性與恢復

- 原生 kernel 支援 L≤128，使用 unsigned 128-bit bitset 及整數 popcount；
  專案長度都在範圍內。macOS/Linux 的 clang/GCC 可編譯，不需 pip 套件。
- ctypes.CDLL 呼叫掃描時釋放 GIL，讓其他 Python 執行緒能繼續；
  尚未用可用 GPU 驗證實際 GPU dispatch 改善幅度。
- 每次 native 呼叫最多 16,384 assignments；回 Python 後檢查時間。
  parity/content/cross 條件及獨立 verifier 仍由原精確路徑檢查。
- Python/native 使用完全相同的 Gray-order cursor，能中途切換實作恢復。
- native 回傳不合法 cursor/status 時直接報 implementation error，絕不
  靜默當成 UNSAT。缺編譯器／平台不支援才允許 auto 回退 Python。
- 索引產生部分 batch 後逾時，已找到的 children 和 cursor 一起保存；
  不丟棄部分成果、不把未掃完的中間層當作 UNSAT。
- 一個 batch 裡仍做完整 canonical 去重，pending representative 留在佇列。
  父問題仍要全部 middle／children 完整排除才可以 UNSAT。
- shared library 以來源 hash 命名、暫存編譯後 atomic replace；不覆寫
  原搜尋 executable。沒有編譯器時 warning 一次後自動用 Python。

回退本次兩項優化、仍保留上一版 MITM/GF(2)：

```sh
.venv/bin/python main.py --L 68 --z3-affine-backend python --z3-mitm-batch-size 1
```

## 程式、驗證與證據

新增 `csrc/affine_scan.c`、`solver/affine_native.py`；修改
`solver/affine_lift.py`、`solver/middle_mitm.py`、
`solver/staged_completion.py` 與 main.py 的參數／backend 輸出。
新增 `tests/test_affine_native.py`、`tests/test_middle_mitm_batch.py`，涵蓋：

- L4、8、12、44、64、68、94、128 的 scalar PACF 交叉驗證。
- 超過 64-bit 的旋轉／popcount、首個 SAT 次序、空區間與邊界。
- Python/native 互換恢復、cross 拒絕後繼續、fallback 與錯誤傳播。
- batch 與逐個順序相同、部分 batch 逾時保存、atomic write 失敗保留
  舊 checkpoint、model cap、batch-size 變更後恢復。

相關測試 78 passed；排除兩個既有缺失 benchmark 模組後，回歸為
**946 passed、8 skipped、1 warning（84.84 秒）**。原始 `pytest -q`
仍因缺少 `experiments.benchmark_mps_acceleration` 與
`experiments.benchmark_torch_mathematical_loss` 有兩個 collection errors。
warning 是既有 torch 缺 NumPy；新 C kernel 不需要 NumPy。
C `-Wall -Wextra -Werror -fsyntax-only` 通過。

證據保存在 `results/affine_kernel_optimization/`：

- `before.json`：修改前 wall times 與 cProfile。
- `paired.json`：四個 child、三輪交替 backend 比較。
- `staged_paired.json`：三設定的分段工作比較及完整 states。
- `all38_exact_comparison.json`：38 個固定問題與舊 Python 結果核對。
- `background_smoke/`：真實 main.py 背景佇列的短測及停止／恢復記錄，
  和正式搜尋結果隔離，不據此報新解平均耗時。

重跑 paired scan benchmark（使用新的 output 路徑）：

```sh
.venv/bin/python experiments/benchmark_affine_backends.py \
  --manifest results/middle_mitm_L68_followup_design/deep_inputs.json \
  --limit 4 --repeats 3 \
  --output results/affine_kernel_replay.json
```

下一個需要另外量測的問題是 GPU formation、候選跨層供應量，以及固定
中間層可展開為完整 PQCP 的比例。消除計算開銷不等於已解決這些問題。
