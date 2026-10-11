# L86：固定三個半列、任意補全第四個半列

這是本專案從原始 PQCP 定義自行推導的精確補全。
輸入為非解中心，選擇 A-even、A-odd、B-even、B-odd 中的一列；
該列的全部 43 個位元及重量均可改變，其餘三列逐位固定。
UNSAT 只排除這個固定三列的子空間，不是全域不存在證明。

## DC／ALT 給出的全域必要內容

令四個實際符號半列的和為 e,o,f,p，各為 -43 至 43 的奇整數。
嚴格目標 T(0)=172，T(κ)=T(86-κ)=4η，其餘位移為零，故

\[
(e+o)^2+(f+p)^2=172+8\eta,\qquad
(e-o)^2+(f-p)^2=172+8\eta(-1)^\kappa.
\]

除以 4 後只有 45=6²+3²、41=5²+4² 的平方分解。
利用整列換號、交換及單列奇數旋轉，可選以下代表：

| κ parity | η | (e,o,f,p) |
|---|---|---|
| odd | +1 | (11,1,7,-1) |
| odd | -1 | (11,-1,7,1) |
| even | -1 | (9,1,9,-1) |
| even | +1 | (9,3,9,-3) |

奇數半列和要求 DC／ALT 的半和有相反 parity，排除其餘配對。
测试另枚舉全部四個奇整數的可行平方和，獨立核對這些形狀。
補全核心仍保留完整 signed content profiles，不把代表直接當成固定輸入。

必須區分符號慣例：這裡四列全部是實際 binary 位元，負號為 1。
上述代表的總負號數都是 77，所以 λ=77-43=34。
PCS4 原料使用正號 O_B，而 interleave 時實際取 -O_B；
其原料重量 (16,18,21,21) 的 λ=33，不能混用。

## 偶數位移精確決定未知 intersections

對負號支撐 U、重量 w，令 N_U(u)=|U∩(U+u)|，有
P_U(u)=43-4w+4N_U(u)。四個半列總負號数為 R，λ=R-43。
固定其餘三列後，未知列必须滿足

\[
N_U(0)=w,\qquad
N_U(u)=\lambda+T(2u)/4-\sum_{j\ne\mathrm{free}}N_{U_j}(u),\quad u\ne0.
\]

`target_content_profiles` 提供所有必要 w、κ、η；每個完整 intersection
profile 都先檢查整數範圍、對稱及 Σ_{u≠0}N_U(u)=w(w-1)。
接著使用 [有理 PSD 證書](L86_certified_profile_psd.md) 與完整 CRT norm preimages。
舊 backend 只商未知支撐的旋轉，在奇數方程中恢復全部 43 個 phases。
free-O 使用 phase，free-E 使用 -phase，保證固定另一半列逐位不變。
實際兩列 odd cross vectors 相加；這裡沒有 O_B 的額外換號。

新 [norm／trace backend](L86_norm_trace_pruning.md) 直接求 phase 已被固定的
literal masks，再核對全部整數 intersections 與原始長度 86 的完整 verifier。
兩種 backend 都保留全部可行完整模型，允許內容改變。

## 入口與覆蓋

`experiments/complete_l86_half_rows.py` 可在 IDE 按執行鍵；
預設讀最新根目錄 L86 長測 log 的全部已核對非解中心，依次補全四個半列。
結果、輸入、副本及 source manifest 保存在獨立 results 子目錄。
所有四個 free-components 的聯集允許一般 PQCP 等價中心取代表；
只選部分 components 時使用 literal centers，避免旋轉交換 parity 後漏掉子空間。
每個 fiber 的 raw space 是 2⁴³，fiber 彼此可能重疊，總數不能當成互異搜索量。

`complete_parity_half_rows` 共用一次中心驗證及 immutable 準備；
`complete_parity_half_row` 保留獨立驗證入口，沒有可繞過驗證的旗標。
PSD 預設開啟，trace 的最終預設以配對測試記錄為準；主程式搜尋尚未加入這個實驗分支。

最新 59 個中心、236 個完整 fibers 沒有完整新解。
L6 全部非解中心與每個 free-component 已對照全 literal binary 空間的解集合，
包含非空解集合及內容改變；兩種 backend 都通過。
