# 完整有限域補全：擴大 D-optimal 與 Legendre 原料

這是本專案自行推導並實作的固定側精確補全。
新原料、非 perfect Legendre pairs 或更多構造配對，均不是完整 L86 新解。
本轮全部實驗仍沒有完整新解，20% 新解發現速度目標尚未達成。

## 1. 負號支撐的整數條件

43 點符號列 s_i=1-2b_i，其負號支撐 U 的重量 r，滿足
P_s(u)=43-4r+4N_U(u)，其中 N_U(u)=|U∩(U+u)|。
故一對支撐 U,V 是 SDS(r,s;λ)，恰等價於所有非零位移
N_U(u)+N_V(u)=λ。
必要的總和恆等式為 r(r-1)+s(s-1)=42λ。
若不成立，可用整數直接排除，不必枚舉。

在 R=F₂[X]/(X⁴³+1) 中把這個條件模 2 化，得到

\[
UU^*+VV^*=(r+s\bmod2)
+(\lambda\bmod2)\sum_{u=1}^{42}X^u.
\]

固定 U 後，V 的 norm target 完全已知。
既有 cyclic ring 的 X+1 與三個 degree-14 自反不可約因子，
給出全部 CRT preimages；每個非零域 norm 有 129 個 phases。
43 點旋轉是 norm-one subgroup 的 order-43 子群，
取其 coset 代表可保留 V 的全部旋轉軌道。
典型大小因此從 129³=2,146,689 變為 3×129²=49,923。
其他零 norm 分支也完整保留，沒有假設所有域分量非零。

Native 核對每個候選的重量與全部獨立整數 intersections，
保留所有精確模型。Python 再重算全部 42 個非零位移及符號 PACF。
輸出 buffer 按完整 family 大小配置，不用 nearest-distance cap，
也不把原料以外的支撐或未列舉族宣稱無解。

實作：`solver/sds_norm_completion.py` 與 `csrc/cyclic_support.c`。
小長度 3、5 的全部固定支撐、重量、λ 與全部 V 二元空間都已比對獨立 oracle；
43 點則檢查公開原料、所有旋轉 preimages 及完整 integer PACF。

## 2. 新 D-optimal43 原料

r=16、s=18、λ=13 時，兩列和為 11、7，非零 PACF sum 為 2。
此參數原料可核對於 [公開 order-86 表](https://documents.uow.edu.au/~jennie/D-optimal/order.86.html)。
該表五個 (16,18) entries 中，只有四個通過本專案全部整數位移核對；
未通過者一直保留在 rejected 記錄，沒有用於生成。

從四個有效 U 出發，枚舉至多兩次負／正位置交換，共 170,212 個 literal trades。
用共同 unit／U 平移取代表後有 42,321 個固定側子問題。
另一列 V 不受交換距離限制：每個子問題完整決定全部 V。
實際訪問 2,064,737,523 個必要 norm preimages，
571,225,099 個重量符合者進入整數 intersections，得到八個 SDS 等價類。

其中四個不在已驗證公開 parents；再核對先前專案的完整 catalog，
舊 catalog 有五個 SDS 等價類，新 bank 有三個不在其中。
這是本專案原料範圍的擴大，不聲稱文獻上的新設計。
八個原料對全部 perfect-O／relative-phase 的完整 L86 join 均沒有解。
其最低能量 seed 後續每列 radius-5 修復也已完整無解。

主記錄 `results/L86_new_doptimal_norm_trades2_20261007/report.json`；
舊 catalog 核對在同目錄 `novelty_against_previous_catalog.json`。
入口 `experiments/expand_doptimal43_norm.py` 可按 IDE 執行；
`DEFAULT_TRADES` 控制 1 或 2，完整源碼、原料副本與範圍都隨結果保存。

## 3. 奇數列不必各自 perfect

令 x=interleave(E_A,O_A)，y=interleave(E_B,-O_B)，則

\[
T(2u)=P_{E_A}(u)+P_{E_B}(u)+P_{O_A}(u)+P_{O_B}(u).
\]

若 E 為上述 D-optimal pair，只需 O 的 PACF sum 為 -2，就讓全部非零偶數位移為零。
各 O 列不必分別具有 -1 PACF。
O 支撐重量 21、21、λ=20 正好給出這種 Legendre pair，列和各為 1。
因此 full x/y 和為 12、6，滿足正號旁瓣的 DC 平方和 180。
剩下奇數位移由兩個完整 odd cross vectors 差進行精確配對；
所有 κ=1,3,...,41 及 relative phases 都保留，midpoint 43 必須為零。

有限域條件在此為 VV*=UU*，仍可用同一完整補全核心。
從全部 perfect43 平移代表出發，至多兩次交換的 U pool 有 9,330 個 unit／translation 代表。
完整訪問 465,732,054 個 norm preimages，得到六個 Legendre 原料等價類，
其中兩個不是兩列各自 perfect。

**商的覆蓋條件不能省略。**
在對 O 使用共同 unit／獨立反射商之前，先把八個 E pairs 關閉於全部共同 units
及獨立反射，得到 336 個 E 代表；獨立 E 平移由全部 relative phases 涵蓋。
O 的兩種 row assignments 都保留。
否則只取 O 商代表、固定少數 E 原料，會漏掉其餘配對，不能宣稱完整。
Native odd vector 核保留每個相同向量的全部 phase preimages，
並已用 3/5 點完整 binary space 與 43 點 literal oracle 核對。

完整兩次交換實驗決定 3,360 個構造族，3,034,080 次精確 odd lookups，
没有完整新解，最低非解 seed energy 為 36。
記錄 `results/L86_legendre43_norm_interleave_trades2_20261007/report.json`，
每個固定側補全逐行保存在 `completion_jobs.jsonl`。
入口 `experiments/search_legendre43_norm_interleave.py` 可直接按 IDE 執行。
這不是所有 Legendre43 或所有 PQCP86 的不存在證明。

## 4. 已實作的四列完整補全

上述仍分別要求 E sum-PACF=2、O sum-PACF=-2。
其實偶數旁瓣只要求四列 PACF 的總和為零。
若四列支撐重量為 16、18、21、21，這等價於四個 unsigned intersection counts
在每個非零位移相加為 33。
固定前三個支撐後，第四個 norm target 仍可精確推得：
V V*=J+1+U_1U_1^*+U_2U_2^*+U_3U_3^*（F₂）。
然後以整數 target N_V(u)=33-Σ_{j=1}^3 N_{U_j}(u) 完整補全。
這將允許 E/O 各自的 PACF sum 不為常數，而總和仍精確為零。
這個更一般的四列補全已實作於 `complete_sds_many_supports`，
原料的 E/O pair 不再各自要求固定 summed PACF。
`experiments/expand_pcs4_norm.py` 預設依次選四個 free rows；
固定其餘三列、在其中一列作至多一次重量保持交換，第四列全部位元重新完整求解。
同 intersection profile 的 cache 保留全部 literal fixed-triple preimages，
沒有把 homometric rows 當成同一個完整 odd join。
所有 O assignments、相對 reflections 及43個phases由 `pcs4_interleave.py` 完整保留；
E 的獨立 reflection 可用整列 reflection 吸收到 O reflection／phase。
小長度完整 literal oracle 已核對縮減前後的完整解等價類集合。

從37個輸入原料代表出發，四個 free rows 的一交換波決定158,829個 fixed triples，
得到126個原料類，其中89個不在輸入；全部完整 L86 joins 都沒有解，最低非解能量8。
第二波將126個原料作 roots，決定549,665個fixed triples，增加60類到186類，
elapsed=119.395979916秒，仍沒有完整新解，最低非解能量8。
分別見 `L86_pcs4_norm_all_rows_trades1_20261007`、`L86_pcs4_norm_wave2_20261007`。
新原料類只是本專案輸入 bank 的擴大，不聲稱文獻首創或 PQCP86 已解。
