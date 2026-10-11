# 只固定區塊外支撐，讓全部正負號重新求解

此實驗使用主搜尋的 antipodal 三值 C/G 編碼；
與 [相同奇數列構造](L86_odd_kernel_lattice.md) 的偶數子序列平均不同。
目前沒有完整新解或 20% 新解加速證據。

## 必要條件與完整區塊列舉

U_A/U_B 是 C_A/C_B 的 0/1 支撐，總重量為 43。
每個非零三值符號在模 2 都等於 1，所以

\[
P_{C_A}(u)+P_{C_B}(u)\equiv
P_{U_A}(u)+P_{U_B}(u)\pmod2.
\]

目標在獨立位移 κ=min(k,43-k) 恰為 ±1，其他非零位移為零。
因此兩列完整 21 位支撐 PACF 簽章的 XOR 必須等於 1<<(κ-1)。
以 exact tuple/key 相等配對；所有相同簽章的不同支撐都保存。
不是把 homometric supports 視為等價，更不以 hash 或 Bloom filter 命中排除解。

每列選 m 個座標，只固定區塊外的支撐位元，區塊內列舉全部 2^m 配置。
允許重量來自既有經證明的可提升 contents，兩列重量仍相加為 43。
這些重量唯一決定每列 C 的負號數 n=(w-43+e+o)/2。
只在配對重量與簽章均正確後才求解全部 C/G signs。
區塊外的正負號也自由，因此它能跨越固定局部符號交換的限制。

使用 Gray code 一次翻一個支撐位置 p。
對 u=1..21（43 為奇數），沒有 self-edge，增量是

\[
\Delta P_U(u)\equiv U_{p+u}\oplus U_{p-u}\pmod2.
\]

這是目前 U 上的逐步更新；多個位置變動的交叉項已由每一步納入。
 native 程式用兩個 43-bit 旋轉與反射 mask 取得所有 21 位增量。
配置計數必須等於 2^m，輸出容量也是 2^m，沒有結果上限截斷。
每個匹配的 literal pair 再以獨立完整支撐 PACF 核對。

## 完整 signs 与導航

`complete_pqcp_supports` 精確求解 C 與互補支撐 G；
兩邊成功才重建原長度 86 並經 verifier 核對。
只有完整 UNSAT 可以存入 bounded dihedral orbit cache，UNKNOWN 永遠保留。
若取得新的 C witness，但 G 完整 UNSAT，則只當作下一個搜尋起點。
這不是 PQCP 發現，走向它也沒有全域收斂保證。

## 已完成實驗

`results/L86_coarse_support_blocks_v2_20261007/report.json`：
從先前 100,000-trial 長測的 9 組 exact C witnesses 出發，seed=123，
m=16、每起點 8 次，共 72 個完整區塊。
每列每區塊覆蓋 65,536 配置，重量條件各保留 32,768 配置。
共 16,361 組精確簽章配對，全部判定完整支撐無解，無 UNKNOWN。
支撐對稱無解快取命中 14 次。
找到 1 組起點尚未走過的 exact C witness，G 在 2,097,152 個 affine assignments
完整 UNSAT；沒有完整新解。
該新 C 的 literal masks 是 621200143491、3534170317498，
原長度目標 k=2、η=-1，contents 為 (6,26,11)、(11,17,15)。
原始兩列符號與兩個 sign 求解結果都在 `literal_completions.jsonl`。
報告 elapsed=110.256 秒，含此實驗載入與設定、不含長測原料的生成時間；
執行時有其他 GPU/測試工作，不能拿此時間直接宣稱加速。

接著從這個新 C witness 出發，擴大到 m=20，連續完成 4 個區塊。
每列每塊完整覆蓋 1,048,576 配置，各保留 524,288 配置；
共 199,911 組精確支撐配對全部完成符號判定，無 UNKNOWN。
走到 19 組未訪問 C supports，連同原中心共 20 組 literal C supports，
其餘差分條件全部完整 UNSAT，完整新解仍為零。
這是新訪問的 literal supports，不是完整 PQCP 等價類。
快取命中 33 次，報告 elapsed=511.963 秒左右（精確值見 report）。
來源、每個區塊、每筆判定及獨立核對的 C witness bank 保存於
`results/L86_new_coarse_support_blocks20_20261007/`。
原始 43 點整數 PACF、實際 masks 及負號數另外從記錄重算，核對全部 20 個 C 支撐。

## GS 支撐為何必須改變

若 C 支撐的 PACF parity 相對目標有 q 個獨立位移不一致，
因為 G 的互補支撐有相同的 unsigned parity sum，C/G 在這 q 個位置的誤差都必為奇數。
因此 E_C≥q、E_G≥q，E_full=(E_C+E_G)/2≥q。
這是所有 C/G signs 共用的固定支撐能量下界。
原 GS bank 的全部起點 q=8，恰達到 full energy 8；
只重排符號不可能進一步改善。
此結論由 `support_parity_energy_lower_bound`、完整 L6 binary 枚舉及 literal L86 正例核對。
另以 `experiments/audit_support_energy_bound.py` 逐個從原 binary strings 重算支撐 PACF 與完整能量，
所有 24 個 GS specimens 都恰達到固定支撐下界。
完整證據在 `results/L86_GS_fixed_support_lower_bound_20261007/report.json`，包含原始 strings。

已開始從 6 個 GS 等價起點探索較大的 m=18 支撐區塊，
初始支撐不要求是 exact C；所有配對仍嚴格滿足目標 parity，且所有 signs 自由。
來源快照及進度位於 `results/L86_GS_support_blocks18_20261007/`；
是否完整完成及是否有新解必須以該 report 的當前狀態為準。

該 GS 支撐長測後來已完成 96/96 區塊，333,395 組必要支撐配對全部完整 UNSAT，
無 UNKNOWN、沒有完整新解；取得 27 個未訪問 C supports，快取命中 66 次。
完整 elapsed=604.157646 秒，結果屬於此固定 seed 的有限支撐探索。

後續合理方向是從新 C 支撐擴大區塊，並與相同準備時間的 rollback 比較。
若兩組皆沒有完整新解，只能報右設限 time-to-solution，不能推算提升 20%。

## 驗證與範圍

專項測試以長度 3/5/11 的完整 Cartesian 支撐空間作為 oracle，
並在長度 43、m=16 的測例逐項核對全部保留 mask 的完整 PACF signature。
這是 selected-block completeness，沒有覆蓋區塊外全部支撐或一般 L86 全空間。
程式位於 `solver/cyclic_support_blocks.py`、`csrc/cyclic_support.c`；
實驗入口是 `experiments/search_support_blocks.py`，目前未變更主程式預設。
在 IDE 開啟該實驗入口按執行即可使用預設的 m=16、8 sweeps，
結果自動寫入帶時間戳的獨立 `results/L86_support_blocks_ide_*` 目錄，無需指令。
