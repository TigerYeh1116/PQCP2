# 送入 Z3 前的壓縮工作等價去重

## 範圍與使用

`main.py` 選擇 Z3 時預設啟用，適用 factor-2 與 factor-4 精確 lift。
L=44、68 兩層都會使用；L=46 等不能被 4 整除的長度使用 factor-2。
不讀取已知 PQCP 作為 seed，不改 SA、PyTorch loss、Z3 公式或 verifier。

- `solver/compressed_equivalence.py`：純整數 canonical key 與數學推導。
- `CompressedLiftPool`：候選匹配後、排入 Z3 之前去重；一個有限容量的
  submitted history 記住已排入的軌域，不把未排入者標成無解。
- `CompletionWorkStore`：磁碟紀錄也以同一 canonical key 建索引，
  同一 L/seed 的等價工作不重複排入；含已完成及仍 UNKNOWN 的工作。
- `--no-z3-compressed-dedup`：回退到完全相同 pattern/target/content 才去重。
  不關閉分段排程；也不更動正式 L.txt 的完整序列等價去重。

保留原 job 的 raw key、檔名、來源與 target/content。canonical key
只用於 admission/index，**不把實際求解問題換成 canonical 座標**。
因此 SAT 仍按原始 job 驗證、寫入，既有中間模式及 blocked models 也不須搬移。
原本 radius completion 有 Hamming 球與其他座標相關限制，不使用此 key。

## 為什麼安全

依 `文獻/06_PACP.pdf` Project 2 第三題與現有 correlation/verifier：
S(0)=2L，只有 k 與 L-k 非零，值均為 4η。
令 L=m·d，c[r]=Σⱼx[r+j·d]，x 是 ±1 符號序列。

每條序列可獨立旋轉、反轉及整條取負；兩條可交換。
再允許兩條使用**同一個** gcd(q,L)=1 的 decimation：

```
x'[i] = s · x[ε·q·i + t mod L]       s, ε ∈ {−1,+1}
c'[r] = s · c[ε·q·r + t mod d]
```

q 在 Z/LZ 是可逆的，也置換壓縮類別內的元素，因此上述式子精確成立。
PACF 的旋轉、取負不影響數值；反轉利用 P(−u)=P(u)。所以

```
S'(u) = S(q·u)
k' = min(q⁻¹·k mod L, L − (q⁻¹·k mod L))
η' = η
```

注意程式採 x'[i]=x[q·i] 的約定，故 target 使用 **q 的逆元**。
不能只將 k 化成 gcd(k,L) 後忽略相同 q 對兩條模式的作用。

L 為偶數，所以可逆 q 必為奇數。設一條序列的 content 為
(w_even,w_odd)，每組位置都有 L/2 個元素：

- t 偶數：content 不交換；t 奇數：兩個 count 交換。
- s=−1：把當前每個 count w 替換成 L/2−w。
- ε=−1：反轉 i→−i 保持 parity；不是另帶一格偏移的 `[::-1]`。
- A/B 交換：模式和各自的 content 一起交換。

當 d 為奇數，t 與 t+d 的壓縮排列完全相同，但原序列 parity 相反。
因此兩種 content 排列都需枚舉。例如 L=44、68 的 factor-4 與
L=46 的 factor-2。d 為偶數時不可任意交換 content。

這些變換對完整 binary sequences 都有逆變換，並同時維持壓縮、
target 與 content 的關係，故給出整個 lift 解集合的一一對應。
合併的是**等價子問題**，不是「PACF 看起來一樣」或「分數相同」的候選。

## Canonical key

key 包含 `(L, factor, eta, transformed_k, row_A, row_B)`，
每個 row 同時包含完整壓縮 pattern 與 even/odd content。
對共同 q 枚舉，對每條 row 的獨立對稱變換取最小，再比較 A/B 交換。
因 transformed_k 排在 row 之前，只需對達到最小 k 的 q 計算 row key；
這是字典序最小化的等價計算優化，不是額外 pruning。

只識別上述群，不宣稱涵蓋所有可能等價。不同 factor、不同 η、任意
位置排列、A/B 各用不同非平凡 decimation、相同 PACF signature 都不直接合併。
此 key 不能移用到帶任意固定 bits 或額外固定 cross pattern 的問題，
除非連那些條件也一起正確變換。現有 cross-compression pruning 保留所有
數學上相容的 patterns，是完整條件的推論，不是額外固定某個 pattern。

## 舊 checkpoint 與 UNKNOWN

舊 JSON 格式、raw job key、檔名不變。啟動時先驗證所有 checksum，
再選每個等價類的一份代表：已完成 SAT/UNSAT 優先；否則優先可續跑者，
再比較已執行 checks/turns。出現同類 SAT 與 UNSAT 矛盾會報錯，不靜默忽略。

未選中的舊檔**不刪除、不覆寫、不標成 UNSAT**，以便回退。
也不把兩份不同座標下的 blocked models 生硬合併。
只有代表繼續求解；其 UNKNOWN 仍是 UNKNOWN，時間／中間模式上限仍不是證明。
新等價 job 只是避免重複排入，不會取得虛假的 UNSAT 結果。

## 驗證與限制

`tests/test_compressed_equivalence.py`：

- L=4、6：從完整序列操作顯式列出群軌域，與 key 分類雙向比較，
  不只確認「變換後 key 相同」，也確認不同軌域不會被合併。
- L=4、6、8：窮舉所有 binary pairs、content 與 target，檢查同 key
  的完整 lift 解數量、解的正式等價類集合都一致，包含無解子問題。
- 較大 L：直接變換完整序列，重算 compression、content 與 PACF，
  驗證 target 的逆元映射及奇數壓縮長度的 parity 情況。
- 測試候選池、正式 bridge、UNKNOWN 續跑、舊檔保留、損壞資料拒絕、
  已完成工作的重複抑制，以及 radius 不被此規則合併。

快取容量有限：row key 8192、pair key 4096。去重有 CPU 成本，
減少的排入數不等於首解速度提升比例。不同等價編碼在有限 timeout 下
也可能表現不同；目前沒有宣稱能保證 L=68 首解加速。
一個固定壓縮工作仍沿用既有「找到一個 SAT 即完成」行為，並未新增
枚舉該工作所有非等價 binary lifts 的功能。
