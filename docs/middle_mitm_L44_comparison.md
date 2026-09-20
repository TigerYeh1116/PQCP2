# L44 中間層 MITM 搜尋比較

## 結論

本次實測支持 L44 採用新的實驗路徑，但未切換 `main.py` 的預設演算法。

- 5 seeds × 每方法 60 秒：新方法 127 組、舊方法 10 組；實測產出成本
  2.362 秒／組對 30.000 秒／組，產出率 12.7 倍，耗時降低 92.13%。
- 另做 seed 123 × 每方法 180 秒：現有去重規則下新方法 73 組、舊方法
  13 組。額外排除共同交錯取負等價後，新方法保守計為 **72 組**，即
  2.500 秒／組對 13.846 秒／組，產出率 **5.54 倍**，耗時降低 81.94%。
- 180 秒舊方法確實完成 2 次 MPS seed 交接（共 90 組 seed），不是 CPU
  回退模式。60 秒多數 run 未完成第一次交接，不能單憑短跑推論暖機後效率。
- 這是有限樣本、固定既有解集合下的端到端實驗；不是數小時穩態速度保證，
  也不能外推 L68，更不能把所有提升單獨歸因於中間層 MITM。

## 比較方法

舊方法：`main.py` 預設 PyTorch/MPS 連續矩陣 + C 退火 + 背景 Z3，
PACF mathematical loss=lattice，2048 lanes，C 同時 4 workers。保留原本
FKM、溫度、restart、polish、Z3 與剪枝設定，不讀既有解作初始化。

新方法：fresh fixed-content coarse FKM/PACF MITM → middle MITM →
既有 binary Z3 lift。沿用前一次成功實驗：每 coarse job 12 秒上限、
binary slice 1000 ms、最多 16 個 middle models、index 上限 250000。
先枚舉 L44、k=2、eta=-1、parity content=(8,10,8,12) 的 16928 個 coarse
配對，再以 seed 決定順序，並對 compressed jobs 做既有精確等價去重。
這不是對 production pipeline 僅替換單一函式的消融實驗。

每次從新程序／空 seed bank／空 Z3 checkpoint 啟動；使用已編譯的現有 C
執行檔。初始化、FKM 枚舉、驗證、保存都計時。方法依 seed 交錯先後順序，
所有 run 依序執行，不同方法不互搶 CPU/GPU。

所有 run 使用相同 `44.txt` 快照：549 組，SHA256
`c86b3c4b7e59a80d66f8b8ac83dbfbfc5da347a73335840216b2114eeeec7eb9`。
各 run 在獨立資料夾去重與寫入；完成所有測量後才合併正式結果。因此跨 run
重複找到同一新類別可以各計一次「相對相同起點的獨立試驗產出」，但正式
`44.txt` 絕不重複追加。表中的產出數不是跨 run 去重後的總檔案增量。

## 五 seeds、每 run 60 秒

| seed | 舊方法新解 | 新方法新解 | 舊方法首次新解(s) | 新方法首次新解(s) |
|---|---:|---:|---:|---:|
| 123 | 1 | 25 | 25.016 | 7.883 |
| 456 | 3 | 17 | 14.500 | 5.014 |
| 789 | 5 | 25 | 11.852 | 8.241 |
| 1024 | 0 | 29 | 60 秒內未達成 | 4.900 |
| 2026 | 1 | 31 | 19.750 | 5.050 |
| 合計 | 10 | 127 | 4/5 成功 | 5/5 成功 |

新方法產量勝出 5/5。新方法首次新解平均 6.218 秒。
舊方法成功的 4 次平均 17.779 秒，但這不是 5 次的完整平均：另一個 run
是右設限資料，不能將 60 秒冒充其成功時間，也不能把它從吞吐分母刪掉。

主要指標 `seconds_per_new = 全部配置的搜尋時間 / 預算內各 run 新解數`。
它是本次實測產出成本，不是只對成功 run 計算的首次命中平均。

### restart 的正確解讀

- 舊方法主 C 搜尋在 deadline 前最後已報告的 restart 合計 121004，
  停機最終合計 126073。因此預算內真正數量在這兩者之間，折合
  **12100–12607 次主 C restart／新解**。不是精確的逐解 restart 時戳，
  也沒有把額外 Z3 rectangle restart 混入主 C 計數。
- 新方法沒有 SA restart：134 個 coarse lift jobs / 127 組新解 =
  **1.055 個 coarse jobs／新解**，包含最後未完成的工作。
- 兩種工作單位不同，不可宣稱「restart 效率提高 12000 倍」。

## 180 秒補測（seed 123）

| 指標 | 舊方法 | 新方法 |
|---|---:|---:|
| 預算內新解，現有 key | 13 | 73 |
| 再排除共同交錯取負 | 13 | 72 |
| 保守秒／新解 | 13.846 | 2.500 |
| 首次新解(s) | 12.680 | 7.903 |
| 工作次數，收尾後 | 50813 主 C restart | 76 coarse jobs |

依現有 key，各分鐘新解數是舊方法 **2 / 5 / 6**，新方法 **25 / 30 / 18**。
舊方法後期確實有改善，不能說完全沒有 GPU 效果；但本次仍未追平。
只有一個 seed 的延長測試，不把它和五 seeds 合併宣稱六個獨立樣本。

## 驗證與資料保存

371 個保存事件（包括不採用其時間數據的前期試跑、收尾期事件）全數重新：

1. 使用既有 `correlation.py`、`verifier.py`、objective 核對。
2. 另用字面定義的完整 ±1 雙重求和交叉驗證，確認 S0=88、恰兩個非零
   sidelobes 且 magnitude=4、objective=0。
3. 對 MITM 的每個 SAT 核對實際 coarse pattern、target 與 parity content。
4. 使用現有完整序列 key 去重，另審核共同交錯取負：
   `a'_i=(-1)^i a_i, b'_i=(-1)^i b_i`（雙極表示）；偶數 L 時
   `S'(u)=(-1)^u S(u)`，故仍保存 PQCP 條件。

新候選跨全部實驗共有 223 個現有 key；加入上述額外對稱性後保留 222 個。
已透過既有獨立 verifier／atomic writer 追加 **222 組**到正式 `44.txt`，
549 → 771 組。包括已驗證的試跑候選，但試跑時間完全不計入表格。
原檔所有 bytes 保留為前綴；未清理原檔既有資料。
額外審核發現原有 549 筆在較大對稱群下是 547 類，這兩筆既有等價冗餘
不屬於本次測試的清理授權，因此沒有刪除，也沒有改正式去重函式。

## 計時與收尾限制

第一批試跑發現 Python 3.9/macOS `perf_counter` 起點跨程序不同。
該批全部標為 `INVALID_TIMING.json`，不得使用其中的平均或負時間。
後改用 OS `clock_gettime(CLOCK_MONOTONIC)`，新增跨程序 epoch 回歸測試，
並重跑全部五 seeds；正式報告僅用 v2 與 180s 資料。

超時收尾期候選仍保存，但不計入 deadline 內效率。有效實驗總配置時間
960 秒，含程序收尾實測約 1036.278 秒；舊方法的 Z3 收尾耗時另外記錄。
MPS cycle 觀察器曾有 +1 的顯示錯誤，已加測試修正；資料中的原值保留在
`gpu_cycles_before_observer_fix`，校正值由未修改的 `worker.log` 最終行重讀。
這只影響顯示 cycle 數，不影響搜尋、時間或新解數。

## 重現與檔案

```sh
.venv/bin/python experiments/benchmark_middle_mitm_L44_search.py \
  --seconds 60 --output-dir results/choose_a_new_directory
```

此實驗要求真實 MPS，無 MPS 時會報錯，不默默用 CPU 代替。每次輸出目錄
必須不存在。未來執行時的 `44.txt` 已包含本次新解，不能期待重現相同
「新解數」；如需重現此比較，必須使用原始 549 組快照作相同 novelty 起點。

- `results/middle_mitm_L44_comparison_v2_60s/`: 五 seeds 正式結果、參數、快照、
  `independent_audit.json`、`publication_plan.json`、`publication.json`。
- `results/middle_mitm_L44_comparison_180s/`: 延長測試。
- `results/middle_mitm_L44_comparison_60s/`: **時間無效**的試跑，僅保留原始紀錄。
- 新測試：`tests/test_middle_mitm_search_benchmark.py`，不執行長 benchmark。

本次未修改任何 production solver、數學定義、MPS/C/SA/Z3 搜尋參數或預設。

## 測試結果

- 中間層 MITM、coarse research、保存與新 benchmark 回歸：40 passed。
- 原樣執行 `python -m pytest -q`：兩個既有 collection errors，因專案缺少
  `experiments.benchmark_mps_acceleration` 與
  `experiments.benchmark_torch_mathematical_loss`；未假造模組或刪除測試。
- 明確排除這兩個已知無法收集的測試檔後：878 passed、8 skipped，81.36 秒。
  原有 PyTorch 缺少 NumPy warning 保留，沒有安裝或變更依賴。
