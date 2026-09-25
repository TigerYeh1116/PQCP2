# L=2 mod 4：壓縮／方向交替 GPU polish（實驗選項）

本改動不更換 PQCP 定義、objective、連續 loss、C 退火或 verifier。
它接在連續矩陣量化之後，替換該批次的 GPU discrete polish；不是新的
連續可微 loss，也不是 complete enumeration。預設仍是 `baseline`。

直接按 `main.py` 執行鍵時，適用長度會出現「壓縮／方向交替 polish」
互動問題，輸入 `y` 啟用，直接 Enter 保留基準。這項不要求開啟 Z3。

```sh
python main.py --L 86 --seed 123 --torch-polish-mode alternating
```

原本的 device 選擇仍有效：Mac 可加 `--torch-device mps`；CUDA 主機則加
`--torch-device cuda`。不指定時沿用 auto 偵測。
切回基準只需 `--torch-polish-mode baseline`。
`--torch-polish-phase-steps 32` 控制每階段步數。
不支援的 L（例如 44、68）會拒絕此選項，不暗中換回基準模式。

## 精確不變量

令 L=2d、d 為奇數。每個 antipodal pair 有一個偶數與一個奇數位置。
在同一條序列中選兩個 pair，滿足每對內兩個符號相反，而兩個偶數端點
的符號也相反。交換兩個偶數端點，並同時交換它們的奇數 partner：

- 每一個 c_i=x_i+x_(i+d) 不變（兩個受影響的 pair 原本及之後都為 0）。
- A/B 各自的普通 weight、奇數 weight、偶數 weight 均不變。
- 第一個 swap 後，用**中間序列**計算第二個 swap 的 delta；不能把兩個
  對原序列算出的 delta 直接相加。

方向階段只使用上述四點操作。支撐位置階段使用原本同 parity、異符號
二點交換，允許改變壓縮模式。兩階段交替，並使用原本的 full+compression
energy 和 Metropolis 接受公式。此選項不使用 baseline 的 stagnation kick；
離開固定壓縮子空間由週期性的支撐位置階段負責。這是 heuristic，不是 pruning。
能到達哪些 lift 及是否較快找到新解，不能由不變量直接推論。

同時保留每 lane 的最佳 objective candidate 與最佳 factor-2 殘差 candidate。
後者以壓縮平方殘差排序、原 energy 打破同分。在 CPU 完整重算之後，兩者
都在最終 seed-bank cutoff **之前**送到現有 exact compressed-match pool。
只有壓縮精確配對成功才成為 lift job；仍可能 UNSAT/UNKNOWN。原 radius
completion 的限制沒有修改。解的寫入與等價去重仍由既有 verifier/writer 處理。

## 量測

預設啟用 observation-only `SearchMetrics`；可用 `--no-search-metrics` 關閉。
輸出 `results/search_metrics/L86_seed123_<session>.json`，在粗粒度事件時每
60 秒最多 atomic replace 一次，結束時再保存。不是 resume checkpoint。

記錄連續 formation、GPU polish（含現有 CPU export 同步）、matching、
completion worker 等待與工作時間、queue 深度、完成狀態、可取得的 Z3
construction/solve/GF2 時間、rank/free-bits histogram。

- unique compressed matches/min：本次 session 中 pool 回傳的 canonical
  工作 key，包含因 queue 已滿尚未排入的 match；不把 retry 再算一次。
  不包含只從 checkpoint 恢復而未重新出現的工作。
- GF2 histogram 以 completed turn 計數，retry 會再次計入。
- `queue.capacity_deferrals` 是遇到 queue 上限的事件數，不是被略過的 job 數。
- `worker_waiting` 是等 queue 的 wall time，不是 GPU utilization。
- 不提供 timing 的 backend 不補零冒充量測；可看 `.reported_turns`。
- GPU／CPU 同時執行，時間會重疊，不可將各階段時間相加當總時間。
- 不改 RNG／接受決策，但 I/O 和 timing overhead 仍可能改變 wall-clock
  deadline 及非同步排程。不是 bit-exact wall-clock trajectory 保證。
- 量測寫檔失敗只警告並停用量測，不丟棄已驗證的搜尋結果。

## 短測

```sh
python experiments/benchmark_alternating_polish.py --L 46 86 --device cpu
```

使用相同隨機 fixed-content 起點，不讀 L.txt 或已找到的解。預設兩個 seed、
每批 16 lanes、64 steps、每步 4 candidates。這是固定工作量診斷：一次
方向 proposal 計算兩次 swap delta，故另列 `delta_evaluations`。
不是等 wall-clock、完整 GPU→C→Z3 的新解速度驗證。
報表 `elapsed`/`polish-s` 僅計 polish 與 tensor 匯出，後續完整驗證和
compressed pool 配對時間另記為 `evaluation_seconds`。

在本機 CPU、L86、128 lanes、250 steps、每步 8 candidates、phase=32 的
五個 seed 診斷中，兩模式的 lane 平均 score 如下：

| Seed | Baseline | Alternating |
| --- | ---: | ---: |
| 123 | 143.8 | 109.9 |
| 456 | 147.9 | 110.3 |
| 789 | 148.2 | 109.0 |
| 1024 | 150.1 | 108.8 |
| 2026 | 150.2 | 108.1 |

兩者精確壓縮 job 均為 0；交替操作的 CPU polish 耗時較長。
baseline 的 folded-error 統計取 objective-best，alternating 取另行保留的
compression-best，這反映交接設計差異，不能當成只改 neighborhood 的獨立
消融結果。以上支持繼續測試，不足以宣稱首解提速，也不據此更換預設值。

目前環境 MPS/CUDA 不可用。CPU 短測只能確認精確性與初步走勢，不能
推論 GPU 加速或 L86 首解時間。下一步應在實際 GPU 上，用同一 seed 集合、
相同 wall-clock 和 Z3 設定比較 unique jobs/min、GF2 free bits、verified
**new** solutions/time；不能只看 best score。
