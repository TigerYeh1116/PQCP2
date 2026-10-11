# 長度 86：相位計數與同時 C/G 布林求解

日期：2026-10-08。本文記錄新增推導、實作及實際驗證範圍。
截至下列已完成實驗，完整的新 L86 PQCP 等價類仍為 **0**；
因此尚未達成「新解發現速度提升至少 20%」的目標。

## 1. 文獻與適用範圍

Andraschko、Kotsireas、Kreuzer 的 ICMS 2026 論文
[Computing Periodic Golay Pairs Using SAT Solvers](https://icms-conference.org/2026/papers/paper80/main.pdf)
研究以布林權重編碼搜尋壓縮 Golay pairs，再檢查是否能提升。
Golay 的零旁瓣條件與本專案的 PQCP 目標不同；本文重新推導 PQCP 等式，沒有移植其零旁瓣限制。
該論文不是長度 86 PQCP 存在或加速的證據。

另一條相關方向是 [SAT Solving Using XOR-OR-AND Normal Forms](https://link.springer.com/article/10.1007/s11786-024-00594-x)。
XNF 對 XOR 較多的系統提供另一種編碼框架；目前專案新增後端使用既有 Z3，尚未安裝 Xorricane，
也沒有將他人基準的速度提升視為本專案的結果。

以下相位計數是本專案在既有完整 CRT 支撐族上自行推導的最佳化。

## 2. 精確相位計數：保留每一個原始抽樣

在 $R=\mathbb F_2[X]/(X^{43}+1)$，固定第一列支撐 $U$ 後，第二列支撐必須滿足

\[
VV^*=1+X^k+X^{-k}+UU^*.
\]

因為 $\operatorname{ord}_{43}(2)=14$ 且 $2^7\equiv-1\pmod {43}$，
非平凡 CRT 分量是三個 $\mathbb F_{2^{14}}$。
反轉在每個分量上是 $z\mapsto z^{128}$，norm 是 $z^{129}$。
非零 norm 的全部根是 $v_0h^e$，$0\le e<129$；$h$ 的階為 129。
零 norm 只有根 0。

取第一個非零分量。其根序列分成 43 個連續區塊

\[
B_t=\{v_0h^{3t},v_0h^{3t+1},v_0h^{3t+2}\},\quad0\le t<43.
\]

$X$ 在該分量有階 43，所以存在 $\ell\not\equiv0\pmod {43}$ 使 $X=h^{3\ell}$。
乘上 $X^r$ 將 $B_0$ 雙射至 $B_t$，其中 $\ell r\equiv t\pmod {43}$。
同時，其他分量的完整根族只是被置換，零分量及 $X+1$ 分量不變。
回到 literal 支撐，這正是 cyclic rotation，保留整數 Hamming weight。
因此每個區塊與其他完整根族的直積，都有相同的 weight-$w$ 支撐數 $K$。
原始完整族的 weight-$w$ 數量恰為 $43K$。

原抽樣流程先計算完整數量，再呼叫 `rng.randrange(43K)` 得到 literal 索引 $I$。
新流程只計算第一個區塊，接著呼叫 **同一個** `rng.randrange(43K)`，並令

\[
t=\lfloor I/K\rfloor,\qquad j=I\bmod K.
\]

在第 $t$ 個區塊取第 $j$ 個 weight match。
第一個非零分量之前的族都是 singleton，故原始 nested-loop 的 literal 順序恰按這些區塊分段。
新方法取得的是 **相同 literal mask、相同抽樣索引、相同 PRNG 消耗**，不只是等價類代表。
若全部非平凡分量都是零，區塊數為 1，照原樣处理。
若 $K=0$，所有區塊皆空，只排除該固定 $U$ 與指定 weight。

一般三個非零分量的計數工作從 $129^3$ 個組合降至 $3\cdot129^2$；
後續索引選擇也只掃描一個區塊。C/G 完整符號求解沒有更改。
每個選出的支撐仍由 Python 獨立重算整數 weight、parity norm signature 及原點 parity。

**限制：** 此證明需要完整 norm 根族。不能拿它裁掉固定中心 Hamming 球中的其他相位，
也不能把有限次抽樣當成全部支撐的窮舉。

實作：`solver/cyclic_support_sampling.py`。
`experiments/search_norm_supports.py` 已預設啟用相位計數；`--no-phase-count-pruning` 可回退。
`main.py` 的既有 Torch 搜尋沒有因此改成這個獨立抽樣器。

## 3. 同時 C/G 的四狀態模型

使用本專案原始 odd-half indexing。對每列 binary signs $a$ 定義

\[
c_j=(a_j+a_{j+d})/2,\qquad
g_j=(-1)^j(a_j-a_{j+d})/2,\qquad d=43.
\]

每個位置恰有一個非零：$(c_j,g_j)\in\{(1,0),(-1,0),(0,1),(0,-1)\}$。
以兩個布林變數表示：支撐 $S_j$ 及負號 $Z_j$，

\[
c_j=S_j(-1)^{Z_j},\qquad g_j=(1-S_j)(-1)^{Z_j}.
\]

共用 $Z_j$ 不會限制 C/G 符號，因為兩者的支撐互斥。
給定必要 parity contents $(e,o)$ 與 C negative count $n_C$，必須满足

\[
\operatorname{wt}(S)=d-e-o+2n_C,\quad
\sum_j[S_j\land Z_j]=n_C,\quad
\sum_j[\neg S_j\land Z_j]=e-n_C.
\]

第三式是 G 的 negative count；反向重建需用 $a_j=c_j+(-1)^jg_j$、
$a_{j+d}=c_j-(-1)^jg_j$，而非直接把 G 當成未調制的差。

令 $P(u)$ 是完整目標 profile，則

\[
N_{C_A}(u)+N_{C_B}(u)=\frac{P(u)+P(u+d)}4,
\]
\[
N_{G_A}(u)+N_{G_B}(u)=(-1)^u\frac{P(u)-P(u+d)}4.
\]

兩個整數 profile 都必須滿足，僅 C SAT 不是完整解。
對 active edge $(i,i+u)$，同號貢獻 $+1$，異號貢獻 $-1$，非 active edge 貢獻 0。
因此每個 norm equation 可精確寫成布林 pseudo-Boolean equality。
若支撐固定，active edge 數 $I(u)$ 已知，可縮成無負權重的 cut count：

\[
\sum_{\text{active edges}}[Z_i\oplus Z_{i+u}]
=\frac{I(u)-\rho(u)}2.
\]

右側非整數或不在合法範圍時才可排除。
odd $d$ 的反轉對稱是 $N(u)=N(d-u)$，故完整檢查 $u=1,\ldots,21$；
沒有假設自相關對平移 $d/2$ 不變。

模型可讓所有支撐保持自由，或固定 literal supports，或只固定 signed C_A。
完全自由的一列可加 $S_0=1$ 作 rotation gauge：該列至少有一個 active C，
獨立轉動该列保留 C/G norm 及 contents，其 binary 實現是相應的偶數平移。
固定支撐或固定 C_A 的列不使用這個 gauge。

固定支撐時也可加入完整 GF(2) 仿射參數化 $M=M_0+\sum_jz_jT_j$，
其中模 4 norm 及 negative-count parity 已由線性消元滿足。
每個 cut bit 是 $M_i+M_{i+u}$，在自由變數 $z$ 上是仿射函數。
對布林 bits $b_1,\ldots,b_m$，整數總和的第二個二進位位元滿足

\[
\operatorname{bit}_1\Bigl(\sum_i b_i\Bigr)
=\sum_{i<j}b_ib_j\pmod2.
\]

故 cut count 的模 4 第二位元，以及 negative count 的模 4 第二位元，
在完整自由變數空間上都是次數至多 2 的布林多項式。
cut count 已滿足所需 parity 後，第二位元相等正好是 norm 模 8 的必要條件。
常數、線性、二次係數由零向量、所有單位向量及所有單位向量對唯一決定。
新增 SAT 編碼重用已獨立核對的 `pqcp_quadratic_plan`，要求所有 syndrome bits 為零，
再以完整整數 cut cardinalities 決定 SAT/UNSAT；沒有把 mod 8 passing 當成完整見證。
自由維度超過原二次後端支援的 24 時，只加仿射條件，保留其他候選。

若固定支撐的一個 sign problem 已有完整整數見證，C/G 的獨立性允許重用它，
只求另一個 sign problem。先獨立檢查見證的全部 norm、支撐與 counts。
這保持「該支撐對是否存在完整提升」的真假；不代表列舉所有完整等價類。
若指定固定 C_A，重用的 C 見證必須保留同一個 C_A，否則拒絕。

實作：`solver/joint_support_sat.py`。
`complete_pqcp_supports` 的 `sat_unknown_completion=True` 只在原求解器回傳 UNKNOWN 時啟動。
預設仍關閉；啟用 UNKNOWN 補完時會加上上述仿射及模 8 必要條件。
SAT 後端尚無長度 86 加速證據。
有限 `sat_resource_limit` 用盡時保留 UNKNOWN。UNSAT 僅限實際編碼的 contents/固定支撐/固定 C_A 域。
每個 SAT 見證經全部整數 C/G PACF、contents、原始 binary full verifier 核對才接受。

## 4. 已完成驗證與長測

相位計數測試包含長度 3、5 與部分 11 的全部 weight-filtered literal indices，
並包含實際 43-point 支撐與種子逐索引比對。
新 SAT 模型以全部原始 L6 binary pairs 核對 contents、固定支撐、自由支撐、rotation gauge 與固定 C_A；
另核對 L14 的 parity-UNSAT、真正 native-budget UNKNOWN 的補完與見證重用。
L86 的 resource exhaustion 明確保留 UNKNOWN。

凍結來源的完整抽樣配對：`results/L86_phase_count_sampling_paired_20261008/report.json`。
三次各 16,384 trials，順序交替，全部逐行核對 literal inputs、整數求解結果、工作量及 SHA。
兩邊均無 UNKNOWN、無完整新解。中位數 22.392956875 秒降至 14.04472775 秒，
固定抽樣耗時減少 **37.2806%**。這是同一抽樣排程的吞吐量證據，不是已量得的新解發現速度。

獨立一百萬次新抽樣：`results/L86_independent_norm_phase_million_20261008/report.json`。
seed 271828，266.1157515 秒，991,137 個完整固定支撐 UNSAT，
8,863 個固定第一列的 weight family 為空，UNKNOWN=0，41 組 C 見證，完整新解=0。
壓縮 trace 保留全部一百萬筆，約 128 MiB；沒有用未保存的計數代替求解證據。

接著對 41 組見證的兩列 C_A 分別放開 G_A 與整列 B：
`results/L86_phase_million_coarse_free_completion_20261008/report.json`。
82 個錨點的全部 84 目標、共 6,888 目標完整 UNSAT，UNKNOWN=0，新解=0，142.529 秒。
`results/L86_phase_million_coarse_free_closed_20261008/certificate.json` 已逐錨點、逐目標及來源 SHA 稽核。
其後的 radius-one 搜尋沿用這 82 個完整證明，不重做已閉合錨點。

實際 43-point SAT 對照存於 `results/L86_joint_sat_fixed_domains_20261008` 與
`results/L86_joint_sat_reuse_coarse_domains_20261008`。
以各目錄 report 與進程終止狀態為準；未終止的 solver 不記為完整 UNSAT 或加速成果。

加入已驗證 C 見證、完整仿射空間及模 8 方程的版本，
`results/L86_joint_sat_affine_mod8_domains_20261008/report.json` 已完成全部 9 個實際固定支撐對。
SAT 與原 native 求解均判定完整 UNSAT、無 UNKNOWN，分類一致。
這 9 個域的 SAT construction+solve 約 281.32 秒，native 約 0.884 秒，SAT 明顯較慢。
故沒有將它設成一般搜尋預設，也不宣稱它提升新解發現速度。
保留 UNKNOWN 專用接口與原始 SAT 長測，供更大自由維度或不同編碼繼續研究。

另啟動 `results/L86_independent_norm_phase_ten_million_20261008` 的一千萬次新抽樣，
seed 314159265；只以其最終 report 判斷完成及完整新解數。
`results/L86_phase_million_coarse_radius1_20261008` 是 41 組新 C 見證的一階 L1 鄰域，
目前仍執行中，沒有將未結束部分列為已排除。

## 5. 按執行鍵

開啟 `experiments/run_l86_norm_search.py` 並按 IDE Run。
預設一百萬次、相位計數已啟用、完整 C/G 求解、全部 trace 壓縮保存、輸出獨立 timestamp 目錄。
每次 Run 記錄新的 seed，避免重做同一批抽樣；要重播可修改檔案頂端 `RUN_SEED`。
可直接修改 `RUN_TRIALS` 調整次數，不需要輸入命令。
精確 C 見證另存 `coarse_witnesses.json`，完整 PQCP 才存 binary inventory。

`experiments/benchmark_joint_support_sat.py` 是 SAT 後端對實際支撐的驗證入口，
不是目前首選的長測搜尋入口。其預設無 SAT timeout，可能花很長時間。

有限 miss 只排除已完成的域。目前仍需搜尋新的支撐或尚未閉合的 C_A 鄰域，
並等到真正的新完整類別出現後，才可進行達成目標所需的發現時間配對。
