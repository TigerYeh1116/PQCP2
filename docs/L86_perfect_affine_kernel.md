# 正號奇數旁瓣的整數仿射格構造

此為本專案自行推導。只研究兩列奇數索引子列相同且反號的 perfect-O 構造，
偶數子列 E_A、E_B 的和固定為 11、7；不限制一般 PQCP 搜尋。
這補足先前只處理奇數旁瓣為零的線性核研究。

沿用 [原有 interleave 展開](L86_odd_kernel_lattice.md)，
c=(E_A+E_B)/2、g=(E_A-E_B)/2，故 Σc=9、Σg=2，逐座標支撐互補。
perfect O 滿足 ΣO=1、P_O(0)=43、P_O(u)=-1。
指定奇數正號旁瓣 κ=43-2t，t=1..21。

## 1. 奇數條件轉為整數仿射方程

建立 22×43 整數矩陣 S：第一列為 O，其餘第 u 列為
S_u(i)=(O(i-u)+O(i+u))/2。
直接代入交叉相關式，全部奇數位移的指定值恰等價於 Sg=e_t。
其 Gram 為 diag(44,22,...,22)-J。
原核基底 B_u(i)=(O(i-u)-O(i+u))/2 滿足 BBᵀ=22I，SBᵀ=0。
逐一驗證模 2、11 的 B 列秩為 21，證明它生成完整整數核。

用 Smith 分解產生整數右逆 R，另用純 Python 整數逐項驗證 SR=I。
保存全部 S、B、R；若無法取得證書就中止，不能假設整數解存在或排除未知向量。
取 g_p=R e_t，全部整數解為 g=g_p+Bᵀα，α∈Z²¹。

最小實範數解為 g_min(i)=(1+S_t(i))/22，範數平方 3/22。
令 q_s=〈g_p,B_s〉，則 g_p=g_min+Bᵀq/22，得到精確恆等式

\[
22\|g\|^2=3+\sum_{s=1}^{21}(q_s+22\alpha_s)^2.
\]

Σc=9 要求 c 至少 9 個非零座標，故 ||g||²≤34。
因此只需枚舉整數 r_s≡q_s mod 22、Σr_s²≤745。
每個座標絕對值 ≤27，有限列舉使用整數平方與 suffix minimum bounds，
保留全部界內 residue vectors；重建 g 後再驗證原 Sg=e_t、Σg=2 及逐座標三值條件。
沒有把連續最小解或只搜尋短係數當成完整整數格。

## 2. 偶數條件與完整提升

偶數旁瓣全為零，故對 u≠0 必須 P_c(u)=1-P_g(u)。
P_c(0)=43-||g||²=w，負號數固定為 (w-9)/2，支撐為 g 的零座標。
既有精確 native GF2／mod8／cardinality 系統決定全部 c 符號。
UNKNOWN 不當作排除。每個 SAT 都要完整重建 86 點，
獨立驗證全相關向量正好等於 κ=43-2t、η=+1 的嚴格目標。

## 3. 完整有限族結果

完整 perfect43 原料共有 8 個平移代表。
每個代表保留 t=1..21，共 168 個仿射 fiber，沒有假設所有 t 都等價。
兩個 Paley 代表各得到 672 個 trit candidates；
其他六個代表的所有界內整數 sphere vectors 都不符合 trit 條件。
共 1,344 個固定 c 支撐子問題完整 UNSAT，沒有 UNKNOWN 或完整新解。

記錄 `results/L86_perfect_affine_O_complete_20261007/report.json`，
含完整證書、所有 fiber、固定支撐結果與來源快照。
程式 `experiments/search_perfect_affine_o.py` 可直接按 IDE 執行。
專項測試含長度 3、7 的整個三值空間、所有 perfect43 整數右逆、
完整 residue sphere 的小整數 cube 及原始 86 點奇偶 PACF 展開核對。

這只排除上述相同 O、指定偶數列和與正號奇數目標的構造族。
不同 O_A/O_B、不同比例的列和或一般 L86 PQCP 不在證明範圍。
不能以有限族 UNSAT 或候選数縮小宣稱新解速度提升 20%。
