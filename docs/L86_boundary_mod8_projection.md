# 自由短區塊的模 8 線性投影與跨內容修復

這是本專案自行推導的精確必要條件。核心適用於任意二元前綴，
不要求 Golay、週期互補、反射或特定建構。
目前沒有完整 L86 新解或新解發現速度提升 20% 的證據。

## 1. 保留原始 86 點相關，沒有假設前綴項相消

固定兩列各 82 个 bits，最後 4 個 bits 自由。
用符號 x_i=1-2a_i 表示，補入的八個符號為 s_j=1-2b_j。
對獨立位移 u=4..43，同一列四個自由位置之間沒有 tail-tail edge。
故自相關和可寫成

\[
T(u)=K_u+\sum_{j=0}^7 \ell_{u,j}s_j,\qquad
\ell_{u,j}=x_{p_j+u}+x_{p_j-u}\in\{-2,0,2\}.
\]

兩個鄰居均為固定前綴位置，包括 u=43 時相同鄰居被計兩次的情況。
K_u 保留全部前綴相關；即使 82 點前綴是 periodic Golay pair，
改成 86 點後也不能把 K_u 擅自設為零。
實作直接計算全 86 点、自由 bits 暫為 0（符號 +1）的 baseline T_0。

兩列二元 PACF 的和必為 4 的倍數，因此

\[
\frac{T(u)}4=\frac{T_0(u)}4-
\sum_j\frac{\ell_{u,j}}2b_j.
\]

約化到模 2，即得到原問題模 8 的 40×8 線性必要系統。
若 k≥4，右邊是 k 位移的一個 unit vector；若 k=1,2,3，右邊均為零。
η=±1 在模 2 相同，因此先合併，再由完整整數 verifier 決定符號。
自反位移 43 必須為零；沒有把它當成可選例外位移。

Gaussian elimination 保留每個線性解及全部 nullspace 組合。
不一致只能排除該前綴及該目標組；全部目標組均不一致才排除這個固定前綴的任意補尾。
每個保留尾端配置重算所有獨立 PACF：只能有一個非零獨立位移、值 ±4、且半位移為零。
Python 再獨立驗證原始 86 點及完整 target profile。
沒有浮點、epsilon、近似 PSD 或隨機排除。

## 2. 直接補上原局部修復未覆蓋的內容變更

既有 swap／4+4 修復固定四個 parity counts。
新的自由區塊允許 parity counts 改變，但最後必須通過同一個嚴格 PQCP verifier。
對任意非解中心，獨立旋轉兩列，把選定的 cyclic window 放到尾端，
用上面的核心求解，再把結果旋回原座標。
所有區塊外 bits 必須原樣保留，重建與 Hamming distance 均另行核對。

m=4 時每列有 86 個窗口，共 7,396 個窗口對；
每對涵蓋全部 256 個 bits 配置，raw 計數是 1,893,376。
這是所選 cyclic-window union 的完整覆蓋，不是任意 Hamming≤8 的完整球。
m=1 則涵蓋每列任意至多一個 bit 改動，仍允許跨內容及符號目標。

中心去重只能使用保持 contiguous windows 的變換：
獨立旋轉／反射、整列換號與列交換。
一般 unit decimation 把窗口變成 arithmetic progression，不能刪去相应中心而宣稱鄰域仍完整。
已提供反例測試，確認此中心 key 不使用 unit decimation。
現有 24 個 GS specimens 在這個較小且正確的群下有 6 個中心代表。

實作：`solver/boundary_tail_search.py`、`csrc/boundary_tail.c`、`solver/boundary_tail_repair.py`。
完整 GS window 搜尋入口：`experiments/repair_gs_cyclic_windows.py`，可在 IDE 直接執行。
目前沒有加入主程式預設；配對計時工作結束後已完成全部六個中心的長鄰域實驗。

## 3. 另一個使用同一核心的構造族

[Đoković、Kotsireas 的原始論文 §3.2](https://arxiv.org/pdf/1310.5773)
列出兩個長度 82 periodic Golay parents。
本專案把論文的 orbit representatives 展開為 literal negative supports，
獨立核對其全部 82 點整數 PACF 為 (164,0,...,0)，才用於補四點的構造。
這些是 82 點原料，不是新 L86 解。

完整預定佇列保留兩個 parents、共同單位 q 的正負商共 20 個、
兩列各全部 82 個旋轉及所有 256 個尾端配置：
268,960 個前綴對、68,853,760 個 raw binary models。
反轉前綴並反轉尾端，等於先反轉全 86 點再旋轉 4 點，故安全保留其等價代表。
整列換號、列交換、共同交替換號也不遺漏新完整解等價類。
其他 periodic82 parents、插入到非尾端的布局及一般 L86 解不在此構造族。

原始兩個未旋轉、未 decimate 的前綴各自秩為 8；
全部目标組皆不一致，所以各自所有 256 個 tails 完整無解。
記錄 `results/L86_periodic82_boundary_source_checks_20261007/report.json`。
這只是兩個原始前綴，不是完整 orbit traversal 的結果。
入口：`experiments/search_periodic82_boundary.py`。

## 4. 目前驗證

核心與 wrapper 的專項測試包含：
L6 全部 prefix/tail binary space 的獨立 PQCP oracle、依赖 columns 的 nullspace、
66/86/94 點跨 bit64 的完整 256 tails、原料的 82 點完整 PACF、
prefix reversal 的完整 86 点相關恆等式，以及 small cyclic-window union 的全部 literal configurations。
cyclic-window 正例實際改變 parity contents，確保沒有沿用固定內容限制。
另加入一個非 PQCP 的 86 點 mod-8 可行前綴，強制大型投影非空，
避免只用通常全空的隨機前綴驗證大型排除邏輯。
各項最終結果應以實際測試輸出為準；測試通過也不代替新解加速證據。

原有 10 個測例已全數通過；新增 L86 非空投影測例另單獨通過。
本次再將全部 11 個測例一起跑過，均通過。
完整 periodic82 orbit 已完成 268,960 個前綴對，涵蓋 68,853,760 個 raw tails；
全部目標組皆被模 8 必要條件排除，無完整新解。
記錄 `results/L86_periodic82_boundary_complete_20261007/report.json`。
六個 GS 中心的全部窗口對也已完成，raw 配置計數合計 11,360,256，
模 8 必要投影全空，無完整新解。
記錄 `results/L86_GS_cyclic_windows4_20261007/report.json`。
這兩項是明確有限族的完整結果，不能推論一般 L86 不存在。
後續放寬為 [任意位置、內容自由的翻轉配對](L86_unrestricted_flip_join.md)，
不再限於連續窗口，且涵蓋較小半徑。
