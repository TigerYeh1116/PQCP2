# factor-4／factor-17 雙向精確壓縮相容性

L=68 時，factor-4 壓縮長度為 17，factor-17 壓縮長度為 4。
新檢查預設接到多層 Z3 completion：

`factor-4 精確配對 → 雙向相容性 → 34 維中間模式 → 相容方向篩選 → Z3 完整 lift`。

預設 staged 路徑現在於「雙向相容性」後先加一個有限的
[C 矩形交換搜尋](rectangle_search.md)，固定兩種壓縮作局部搜尋。
miss 不排除任何模式，仍回到原本多層 exact 流程。

一般化適用於 L=4n 且 n 為奇數，例如 44=4×11、68=4×17。
其他長度保留原本路徑。不是新的 SA neighborhood，不依賴已知 PQCP。

## 必要條件及安全性

1. 因 gcd(4,n)=1，CRT 將每個原索引唯一對應到 `(i mod 4,i mod n)`，
   每條 0/1 序列等價於一個 4×n 二元矩陣。沒有改變原始循環相關的索引。
2. 設 factor-4 sign pattern 為 c，factor-n sign pattern 為 r，則矩陣
   column degrees 是 `(4-c[j])/2`，row degrees 是 `(n-r[s])/2`。
   總數必須相等。Gale–Ryser 條件為：將 row degrees d 降冪排序，
   對 t=1..4，`sum(d[:t]) <= sum(min(t,column_degree))`。
   這對「存在具有這些行列和的二元矩陣」必要且充分；對 PQCP 僅必要。
   排序只用於可行性判定，不會將任意行列排列宣稱為 PACF 等價。
3. 完整枚舉 factor-n 的四個元素。範圍是 `-n,-n+2,...,n`，總和由 c
   決定；若指定 parity content，精確要求 d0+d2=w_even、d1+d3=w_odd。
   排除平方和超過 `fold_n(T)[0]` 的 row word 安全，因另一條平方和非負。
4. 以完整 integer PACF signature 配對，精確要求
   `PAF(r_a)+PAF(r_b)=fold_n(T)`。不能套用 Golay 的零 sidelobe target，
   也不能忽略折疊後兩個 ±4 值可能重合（或進入 shift=0）的情況。
   程式直接折疊原始 `target_profile(L,k,eta)`。
5. 保留所有相容 row-pattern pairs；此枚舉沒有 heuristic 上限。
   若全部不存在，只能判定該固定 c_a/c_b/target/content 不可行。

## 中間層與完整 lift

對 factor-2 模式，位置 i 與 i+L/2 的 mod-4 residue 相差 2。
±2 固定兩個 bits，0 必須在這兩個 residue 間分配一個 1。
每個 parity group 的零方向獨立，因此扣掉固定 ones 後，兩個 residue
需要的 ones 非負且總數等於該 group 的零個數，即為精確方向可行性條件。

中間模式若與全部 row-pattern pairs 不相容，安全排除；否則把所有
仍相容的 pairs 以 OR 的形式交給既有 binary Z3 lift。每一 pair 包含
A/B 各四條 residue cardinality 等式。完整 SAT 再檢查 target、content、
壓縮結果與獨立 verifier，正式寫入仍走既有等價去重。

這比只做前置必要條件檢查多了一步：長度 4 模式確實限制 Z3 的展開方向。
它是完整 PACF 條件的數學推論；冗餘精確條件可能改善傳播，也可能增加
建模成本，必須實測，不能預先宣稱加速倍數。

## 使用及觀察

直接執行 main.py 使用預設 Z3／多層 lift 即啟用，L=68 會顯示：

`精確 factor-4／factor-17 雙向壓縮相容性剪枝已啟用。`

`--no-z3-cross-compression-pruning` 可獨立關閉作為對照。
Z3 timeout、UNKNOWN 的意義及 middle-model 上限仍沿用原流程。
`cross_pairs` 是相容模式對數；`cross_pruned` 是本次排除的中間模式數。
快照另存 `cross_pruning_applied`、`compatible_cross_pairs`、
`middle_cross_pruned`，SAT 時保存 `cross_a`、`cross_b`。
pattern count 不等於新的 PQCP 數量。

不適用一般帶固定格子的矩陣時，不能只靠 Gale–Ryser；本次中間層用的是
上述 antipodal 特殊結構的精確分配公式，沒有宣稱實作一般 partial-cell flow。

參考定理：[Matrices with prescribed row and column sums](https://arxiv.org/abs/1010.5706)。
CRT 與本題 target 的連接、奇偶方向分配依本專案 PQCP 定義推導。
