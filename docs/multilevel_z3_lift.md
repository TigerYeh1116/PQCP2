# 多層精確壓縮 completion

對 L 可被 4 整除的搜尋，啟用 Z3 後預設新增
`L/4 -> L/2 -> L` 背景支線；L=68 為 `17 -> 34 -> 68`。
不需要已知 PQCP，不修改原始 ±4 target、PyTorch loss、C 退火或 verifier。

## 精確條件

令 `T=target_profile(L,k,eta)`，全程仍要求原題的兩個實際 shift
`k,L-k` 具有 `4*eta`。使用 aliasing identity：

`PAF(x^(m))[r] = sum_q PAF(x)[r+q*(L/m)]`。

1. 共用候選觀察 hook 將 FKM/PyTorch/C 產生的序列壓縮為 factor-4
   模式，使用完整 integer signature 跨候選配對，必須符合 `fold_4(T)`。
2. `solver/multilevel_lift.py` 以有限 one-hot 分裂枚舉中間模式：
   `c4[i]=c2[i]+c2[i+L/4]`，`c2` 元素只能是 `-2,0,2`。
   要求完整中間 PACF pair 等於 `fold_2(T)`，包含 shift=0 和自對稱 shift。
   若有指定 parity content，也精確要求
   `sum(c2[even])=L/2-2*w_even`，odd 同理。
3. 中間解重新用 Python integer PACF 確認，才交給既有
   `solve_compressed_lift` 展開到完整 binary 序列。展開位置是
   `i,i+L/2`，不是相鄰位置。
4. child UNSAT 才表示該中間模式沒有合法展開；阻擋該中間模式後繼續。
   child UNKNOWN 仍記為未解分支，嘗試其他模式也不會使其變成 UNSAT。
5. 完整 SAT 必須通過既有 independent verifier、指定 target、兩層壓縮
   及 content consistency checks，才可由共用等價去重 writer 寫入 L.txt。

沒有直接對所有 17 維模式作 exhaustive enumeration；來源仍是現有的
候選池。對給定 factor-4 模式，中間層無上限枚舉加 exact child solver
則覆蓋所有 binary lifts。這與專用的壓縮 candidate generator 是不同功能。

## 預設與回退

按執行並於 Z3 提示按 Enter（預設是）即啟用；不可被 4 整除的長度保留原路徑。
當 L/4 為奇數時，另預設套用 [雙向精確壓縮相容性](cross_compression.md)，
L=68 即同時檢查 factor-4 與 factor-17 的行列和及 PACF 相容性。

- `--no-z3-multilevel-lift`：只關閉新多層支線。
- `--no-z3-cross-compression-pruning`：只關閉雙向相容性剪枝。
- `--no-z3-compressed-lift`：關閉兩種精確壓縮支線，保留原 completion。
- `--z3-multilevel-max-models 16`：每 job 預設最多嘗試 16 個中間模式。
- 現在預設使用 [分段預算與待解佇列](staged_z3_completion.md)：
  `--z3-slice-seconds` 設首次預算，`--z3-timeout` 設每分支 check 的最大預算。
  `--no-z3-staged` 才回到多層 job 共用剩餘時間的一次性模式。
  construction 與 verification 不在 check timeout 內。

達到時間或 model 上限只回報 UNKNOWN；只有中間層完整耗盡且所有
child 都 UNSAT 才回報該固定壓縮模式/content 的 UNSAT。
候選池容量、排隊上限及沒有配對都不是無解證明。
原 factor-2 支線與 radius completion 保留。Z3 使用次數現在按實際
`check()` 次數計數，多層 job 可以包含多次 check。

## 觀察與保存

輸出 `多層精確壓縮 17 → 34 → 68 已啟用` 僅表示功能開啟；
`精確 factor-4 配對 → Z3 lift queue` 表示 job 已排隊；
預設 `Z3 分段求解` 回報該輪子工作、預算及排隊／暫存狀態；
回退模式的 `Z3 multilevel lift` 回報中間模式數及 Z3 check 次數。
快照仍保存於 `results/compressed_lift/`，`method=multilevel_z3_lift`、
`factor=4`，包含中間模式、未解 child 數、各階段 checks 與完整 SAT candidate。
正式結果沿用 L.txt 格式與對稱群去重。

參考：[New Results on Periodic Golay Pairs](https://arxiv.org/abs/2408.15611)。
只借鑑 multi-level compression 架構；所有 target 由本專案 PQCP 定義折疊。
此實作尚未證明能縮短 L=68 首解時間，必須以實際求解量與 verified 新解驗證。
