# 長時間搜尋的記憶體管理（2026-10-06）

本次目標是在保留搜尋方法、批次大小與精確驗證的前提下，改善運行時間
越長記憶體越高的問題。沒有更改 PyTorch 版本，也沒有刪除原有解庫或斷點。

## 原因與修正

### MPS 主機端暫存物件

在原始 L86、batch=16384 的 8,000-step 測試中，MPS 活躍張量約
52–56 MiB、驅動配置約 2.63 GiB，暖機後均相當穩定，但程序 RSS 仍增加。
其中 step3000 至 step8000 增加 24.00 MiB。這不是單看 GPU tensor 計數
便能發現的問題。

獨立 forward/backward 測試以 tracemalloc 確認 Python 保留量穩定，
原生 RSS 卻持續上升。加入 autorelease 邊界後，這個增長大幅下降。
[Apple 的記憶體管理說明](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html)
也要求長時間的非 UI 程式、產生大量暫存物件的迴圈及背景執行緒管理自己的
autorelease pool。

實作使用公開 Foundation `NSAutoreleasePool`，建立及釋放都在同一執行緒。
每個 `TorchSearch.step`、`observe` 及完整 GPU refresh cycle 均有明確
的暫存物件生命週期。它不清空可重用的 GPU allocator cache，也不另加
GPU synchronize。C helper 首次在既有 build/ 目錄編譯，之後以來源雜湊快取。
CPU/CUDA 不載入 Foundation。
當 refresh 已有 pool 時，內部 steps 與 observe 共用它，避免對仍在提交的
命令反覆建立、釋放 pool。暫存物件最多保留一個有時間上限的 refresh；
單獨呼叫 step 時仍有自己的釋放邊界。此狀態是 thread-local，例外退出
亦會清理，不會把另一個執行緒誤認為已受到保護。

### 避免 MPS 稀疏步幅與全零步幅梯度

PyTorch 2.8 的
[getStridedMPSNDArray 原始碼](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/mps/OperationUtils.mm)
含部分 retained NSArray copy，後續
[上游修正 #176648](https://github.com/pytorch/pytorch/pull/176648)
補上釋放。這次沒有升級整個環境，而是在本專案範圍避免部分相關輸入：

1. origin 平方先保留最後維度，再 squeeze，數值與導數不變。
2. 使用 per-lane loss 的全一向量 VJP，取代先 sum 再 backward。

若 f(θ) 是全部 lane 的 loss 向量，則

\[
\nabla_\theta\sum_i f_i(\theta)=J_f(\theta)^T\mathbf1.
\]

因此 `loss.backward(torch.ones_like(loss))` 與原方法有相同目標和梯度，
並避開 SumBackward 的全零步幅展開。沒有更改 loss 權重、隨機數或接受條件。
最終實作把全一向量建立一次，供全部 steps 和 restart 重用，避免每步
額外啟動 fill kernel；它是可重建的常數，不需加入 checkpoint 格式。

### Completion 歷史與去重索引

- completion 記憶體只保留最近 128 筆結果。執行次數、Z3 calls、驗證成功
  次數使用獨立累計數，history eviction 不影響總數或 callback。
- session 唯一壓縮工作計數、staged 等價代表索引使用精確 SQLite 臨時索引，
  page cache 設為 512 KiB、停用 mmap。使用明確的磁碟檔案，避免 SQLite
  空檔名的臨時資料庫仍保留原生記憶體。完整鍵保存於磁碟，沒有採用 Bloom filter、
  hash-only 判斷、LRU 遺忘或刪除搜尋工作。
- 恢復斷點時逐筆讀取並驗證 checksum，索引只保存代表 key、進度比較值和
  proof 狀態。依舊檢查等價 SAT/UNSAT 衝突，並保留原座標和所有原 JSON。
  同進度時維持原先 key 字典序的代表選擇。
- 關閉後 SQLite 檔案與專屬臨時目錄自動清理；staged store 需要重開時可由 JSON
  重新建索引。正式 checkpoint 格式沒有修改。
- 觀測索引 I/O 錯誤只停用 metrics；數學 checkpoint 錯誤依舊 fail closed。

## 已驗證的正確性

- CPU：origin 的值與解析梯度逐元素相同。
- CPU：完整 lattice loss 的 vector VJP 與 sum backward 逐元素相同。
- MPS：修復前後各做 192 steps、3 次 restart，完整 checkpoint 逐欄完全
  相同，包含 θ、Adam 動量、archive、best、parity profiles 和累計數。
- 10,000 次 completion 回報後，history 長度維持 128，全部累計數精確。
- 20,000 個完整鍵超過 page cache 後，重新查詢全部重複鍵仍正確。
- 最終完整鍵壓力測試：100,000 筆時，舊版 Python 保留量 102.62 MB、
  程序 RSS 164.40 MB；新索引 Python 保留量約 51 KB、RSS 18.38 MB。
  新版 20,000 至 100,000 筆的 RSS 只增加約 48 KB，全部重複鍵仍正確。
- 既有 corruption、等價去重、UNKNOWN backoff、關閉後 recovery 測試通過。

## 量測方式與重現

量測工具為 `experiments/benchmark_memory.py`。所有搜尋結果及失敗診斷
集中在 `results/memory_20261006/`，未寫入專案根目錄的正式 L.txt。
`baseline_source.tar.gz` 保存本次開始前的完整程式來源，可供回退及重現。

工具提供 formation 固定 step 數、完整 hybrid 固定時間兩種量測，記錄當前
RSS、peak RSS、MPS 活躍量與 driver 配置量。hybrid 的量測執行緒只讀取
allocator counters，不跨執行緒同步正在提交的 MPS stream。

首次嘗試跨執行緒 synchronize 時，舊版 MPS 產生 command-buffer assertion；
已修正量測方式，確認沒有遺留測試的 C 子程序，再重新做正式對照。單獨
改 origin shape 的實驗也沒有消除主要增長，故沒有把它當作充分的修復。
第一版空檔名 SQLite 的 Python 配置量很小，但原生 RSS 仍增加；已改為
明確檔案並重新量測 RSS，沒有把 tracemalloc 當成全部記憶體的證據。

正式比較的設定為 L86、seed123、batch16384、C threads6、lattice loss、
alternating polish、全部直接壓縮修復和背景 Z3。搜尋步數與 C proposal
計数用於檢查效率，初次初始化、快取暖機與回歸測試干擾分開解讀。

舊版完整 hybrid 運行 606.86 秒，完成 16,400 GPU steps 和約 376.51 億次
C swap evaluations。修正版本的二十分鐘測試完成 31,800 GPU steps。
同一 180–590 秒區間的 RSS 線性增長斜率，由約 10.49 MiB/min 降到
2.46 MiB/min；修正版本後十分鐘約 1.99 MiB/min。這是明顯改善，沒有
宣稱跨 refresh、allocator 高水位和所有上游快取已完全零增長。

固定工作量比對的後 4,000 steps，選在暖機與回歸測試結束後：

| 方法 | 搜尋秒數 | RSS 增量 |
|---|---:|---:|
| 舊版 | 91.02 | 19.44 MiB |
| 最終修正 | 92.87 | 4.11 MiB |

此區間耗時約增加 2.04%，RSS 增量降低約 78.9%。完整 hybrid 的首次
長短時間比較中，GPU steps/s 差約 −2.46%、C swap/s 差約 −5.47%；
這包含時間長度、背景活動和 seed handoff 的差異，所以另做相同十分鐘
的最終比較，而不宣稱 CPU 速度完全不變。

L46、batch8192 的 240 秒實際 completion 負載正常完成，72 次 completion
turns；同時運行真實 Z3，觀察到 RSS 在活躍解題時上升、後續下降。
暫存索引保持有界，待解工作全部保存在獨立 checkpoint 目錄；沒有將
Z3 活躍 clause 記憶體錯認成已經排除的洩漏。

可收集回歸為 1153 passed、14 skipped；實際 MPS 測試 42 passed、1 skipped。
沿用兩個既有缺少模組的 benchmark 測試排除：

```sh
.venv/bin/python -m pytest -q --ignore=tests/test_mps_acceleration_benchmark.py --ignore=tests/test_torch_mathematical_benchmark.py
```

主要結果集中於 paired_comparison.json、comparison_initial.json、
disk_file_metadata/report.json、fixed_hybrid_1200s/report.json 和
fixed_L46_hybrid_240s/report.json。最後的相同時間比较另存
final_hybrid_600s/。沒有把沒有找到 PQCP 當作記憶體修復的失敗或成功。

最終採用共用 refresh pool 的同時間對照（amortized_hybrid_600s）：

| 量測 | 舊版 | 最終版本 |
|---|---:|---:|
| RSS 增長斜率，180–590 秒 | 10.49 MiB/min | 2.13 MiB/min |
| 結束 RSS | 696.34 MiB | 605.70 MiB |
| GPU formation steps/s | 27.02 | 25.27 |
| C swap evaluations/s | 6204 萬 | 6338 萬 |

此對照的 RSS 增長率下降 79.7%，C 吞吐量提高 2.15%，GPU steps/s
下降 6.51%；固定工作量的差約 2%。不同时间段的硬體負載和 seed
handoff 会影響完整並行吞吐量，因此不宣稱完全零成本，也不宣稱新解
發現率提高。所有批次大小、C threads、loss 及搜尋方法維持一致。
資料與計算在 comparison_accepted.json。

最終 full regression 為 1153 passed、14 skipped；增加共用 pool 的
thread-local 驗證後，相關模組 29 項全部通過。實際 MPS 組 42 passed、
1 skipped；三次 restart 的最終完整 checkpoint 仍與舊版逐欄相同。

## 使用與限制

按 VS Code 執行 main.py 即自動使用修正，不需新指令或另外啟用選單。
保留 GPU 暖機快取，所以記憶體不會回到剛啟動時的最低值。持續累積於
磁碟的解和斷點是可恢復工作的一部分，本次不刪除這些資料。

大型 batch 的最大 workspace 仍由 batch、長度及 observation 算法決定。
本次優先修正時間累積，保留現有硬體工作量。長測不能證明所有未來環境
都沒有 PyTorch/Metal 問題；若更換 PyTorch 或 macOS，應再用相同量測驗證。
