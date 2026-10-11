# L86 新搜尋：執行模擬

2026-10-04 新增非零位置／符號分離修復：啟用直接三值壓縮後，預設每輪處理
4 個候選，先用模 2 匹配非零位置，再用模 4 消元及 C 位元窮舉重排全部符號。
按執行鍵並在既有直接壓縮選單回答 y 即包含此分支。
推導、排除範圍及實測見 [非零位置與符號分離研究](L86_support_sign_search_20261004.md)。

2026-10-04 更新：GPU 三值 batch 後新增整數 C 延續搜尋，預設每輪選 4 個
候選、各 131072 次有效提案，再交既有精確修復。`--torch-direct-compression-native-top 0`
可關閉。數學限制、最新長測與實驗證據見 [10 月 4 日分析](L86_long_run_20261004.md)。

GPU 跨內容提案、跨內容精確修復、C 雙側兩交換匹配，均已接入正式 direct
compression 支路。既有 formation、C 搜尋、lift、完整驗證與解庫去重保留。
目前沒有找到 L86 解，沒有宣稱新方法提高了發現率。

## 執行

在專案目錄：

```sh
.venv/bin/python main.py --L 86 --seed 123 --torch-device mps --z3 --torch-direct-compression
```

持續運行直到 Ctrl+C。若需要固定一小時：

```sh
.venv/bin/python main.py --L 86 --seed 123 --torch-device mps --z3 --torch-direct-compression --hours 1
```

VS Code 直接執行 main.py 時，選 L86、啟用 Z3，並對「直接三值壓縮搜尋」回答 y，
便會包含全部新功能，不需另外選新開關。開始會列出實際參數。

| 參數 | 預設 | 意義 |
|---|---:|---|
| `--torch-direct-compression-transfer-interval` | 4 | 每 4 步改用跨內容提案；0 關閉 |
| `--torch-direct-compression-repair-top` | 8 | 每輪固定內容半徑 4 修復數；0 關閉 |
| `--torch-direct-compression-transfer-top` | 8 | 每輪跨內容精確修復數；0 關閉 |
| `--torch-direct-compression-mitm-top` | 1 | 每輪雙側兩交換匹配 reference 數；0 關閉 |
| `--torch-direct-compression-mitm-results` | 32 | 每個 MITM reference 最多保存的不同序列對，1..4096 |
| `--torch-direct-compression-support-top` | 4 | 每輪 support/sign 修復數；0 關閉 |
| `--torch-direct-compression-support-max` | 8 | 每個候選最多求解的 support 數 |
| `--torch-direct-compression-sign-assignments` | 8388608 | 每組 support 自由配置上限；未耗盡為 UNKNOWN |

原 batch=128、steps=128、candidates=8 預設保持不變。要比較較長 GPU 三值搜尋，
可另加 `--torch-direct-compression-steps 2048`，但不代表此數值已證明最佳。
提案間隔與低能量候選排序均為可調啟發式，不能作不存在證明。

若要比較上一版（保留原固定內容修復），使用：

```sh
.venv/bin/python main.py --L 86 --seed 123 --torch-device mps --z3 --torch-direct-compression \
  --torch-direct-compression-transfer-interval 0 \
  --torch-direct-compression-transfer-top 0 --torch-direct-compression-mitm-top 0
```

## 正確性與輸出限制

跨內容步同時對一側做 00→+-、另一側做 +-→00，位置不同且各側新符號個數
仍須能實現指定 binary 奇偶權重。不合法提案完全拒絕。GPU 使用含二次項的
精確相關增量，包含零位移；匯出每個 best 時再以 Python 整數完整重算。

C MITM 列出 A 的零、一、兩次交換路徑，依完整 21 維相關向量排序，串流
B 的對應路徑，用二分查詢找全部相等向量。沒有只存一個同向量代表；不同
序列對會分開匯出，交給原本的等價去重及 lift 流程。

- `UNSAT_NEIGHBORHOOD`：完整排除 N₂(A)×N₂(B)，不是完整半徑 8。
- `SAT_COMPLETE`：這個鄰域的不同壓縮配對已全部枚舉。
- `SAT_TRUNCATED`：已找到超過輸出上限的不同配對；只保存上限數量，不能宣稱枚舉完成。
- 所有 SAT 都只代表壓縮層成立；只有 full verifier 通過才能寫正式 L.txt。

有限輸出與原有 queue admission 都可能限制實際送入 lift 的工作量，故整體
仍為有界搜尋，不宣稱完整枚舉。C 每次工作至多列出兩側有限交換路徑；
停止訊號在 GPU block 與各 CPU reference 之間檢查，當前 C 呼叫完成後才返回。
第一次使用需本機 cc 編譯（本機已成功編譯）；錯誤會明確拋出，不轉為 UNSAT。

## 保存與查看

每次配置產生唯一 session ID，資料位於：

```
results/direct_compression/<session>/L86_seed123_cycle000000.json
results/direct_compression/<session>/L86_seed123_repair000000.json
results/direct_compression/<session>/L86_seed123_<job-id>.json
```

同 seed 不覆蓋舊執行。cycle 檔有 session、對應 metrics 路徑、GPU 轉換提案／
接受數、各 CPU 修復方法工作數、狀態、耗時及新增壓縮工作數。repair 檔的
`advanced_attempts` 保存新方法的 reference、target、結果；原修復仍在 `attempts`。
`results/search_metrics/` 保存整輪計數與新選項，包含 `content_transfer_jobs`、
`mitm_jobs`、`mitm_attempts` 等（前綴 `direct_compression.`）。
各修復秒數是 direct_compression elapsed 的子集，不可再相加估算總時間。

## 已完成驗證

- L6/L14：C MITM 全部輸出與獨立枚舉 N₂(A)×N₂(B) 完全相符，包含同向量不同
  序列、輸出截斷及去重。不是只測一個找到的解。
- CPU/MPS：跨內容增量與完整 PACF 相符；搜尋軌跡確實更換內容並始終保留
  指定總和、總平方和及 parity 可展開性。
- pipeline：新增工作交接、取消、關閉選項與輸出上限狀態。
- L86 MPS 三輪、seed123、128 lanes、2048 steps：每輪 2.89、2.71、2.53 秒；
  跨內容接受 148、158、169 次；C MITM 每輪約 70、87、109 ms。
  最低能量 8、10、8，尚無 exact job。資料：`results/L86_all_features_smoke/`。
  此為功能短測，當時有回歸測試並行，不作隔離效能或成功率比較。
- 正式 `main.py` 入口短測：指定 12 秒，含 block 完成及收尾約 13.65 秒，
  exit=0；4 輪 MITM、457 次 GPU 跨內容接受，所有選項及計數正常保存。
  暫存工作目錄的完整資料已複製至 `results/L86_main_entry_smoke/`，未改正式解庫。
- 完整可收集回歸：1116 passed、14 skipped；另補的 advanced handoff 測試及
  該模組 4 項測試全通過。實際 MPS 測試組 14 passed、1 skipped（CUDA）。
  全套沿用兩個既有缺失模組測試的排除：

  ```sh
  .venv/bin/python -m pytest -q --ignore=tests/test_mps_acceleration_benchmark.py --ignore=tests/test_torch_mathematical_benchmark.py
  ```

  PyTorch 未安裝 NumPy 的既有 warning 仍在，未影響以上驗證。

數學推導見 [設計報告](L86_next_search_design.md)。
