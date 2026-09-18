# PQCP 結果等價去重

`solver/solution_equivalence.py` 的 `pqcp_equivalence_key(a, b)` 計算以下
變換所生成之群的精確 canonical key：

- A、B 各自獨立的循環旋轉、反轉、全部取負（0/1 全部互補）。
- A/B 互換。
- A、B **共同**進行 `i -> q*i mod L`，其中 `gcd(q, L) = 1`。

前三種逐列操作保持各自完整 PACF。共同 decimation 只重新排列 shift，
因此保留 Project 2 的非零 shift 數量與 magnitude。不可對兩列使用不同 q。
canonical key 枚舉所有合法 q，並將逐列 dihedral/complement representative
排序。相同 PACF 或相同壓縮 signature **不會**直接視為序列等價。
此處不宣稱涵蓋所有可能的等價定義。

正式寫入入口為 `solver/search_runner.py` 的
`append_verified_solution_if_new`。先由既有 verifier 驗證，再與 L.txt 中的
canonical keys 比較；等價時不寫入、不增加本輪新增結果數。保留原始候選的
序列及既有 L.txt 格式，沒有修改搜尋決策、objective 或 verifier。

清理既有結果：

```sh
.venv/bin/python experiments/deduplicate_solutions.py
.venv/bin/python experiments/deduplicate_solutions.py --apply
```

第一行只預覽。第二行須先停止所有寫入這些檔案的搜尋程序，會先完整驗證
所有選定 L.txt，備份到 `results/backups/equivalence_<UTC timestamp>/`，
再 atomic replacement；每類保留最早出現的一筆，備份保留原始完整內容。
支援 `--L 44 --L 46` 限定檔案。不清理 logs、seed banks 或其他目錄。

共用 writer 的鎖協調同一 Python 程序中的 CPU/completion 執行緒，
不提供不同搜尋程序間的檔案鎖。更新程式後，既有搜尋程序需重新啟動。
