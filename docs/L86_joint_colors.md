# L86：完整半週期可行空間中的 C/G 聯合搜尋

2026-10-07。此方法自行推導，已完成 literal 整數測試與真實 MPS 實驗；尚未加入 main 預設。
完整對照使用固定程式快照，沒有混入這項後續變更。

## 狀態空間及真正的必要條件剪枝

L=2d、d 奇數。每個 antipodal cell 用 c=(x_i+x_(i+d))/2、
g=(-1)^i(x_i−x_(i+d))/2 表示，只有四種狀態：
(c,g)=(+1,0)、(−1,0)、(0,+1)、(0,−1)。
每一列完整 binary word 與四色 word 一一對應。
由指定 even/odd ones e,o 及 n_c 個 11 pairs，四色個數為

c−: n_c，c+: d−e−o+n_c，g−: e−n_c，g+: o−n_c。

令兩列的 c-support 總數為 d，等價於原始 T(d)=0。
目標非零位置是 k、L−k，k<d；所以所有完整解都在此空間中。
使用已證明的 content bounds 只排除不可能存在完整解的 n_c 類別。
這不是為了容易找到低能量而額外假設對稱性。

## 封閉且連通的移動

固定 contents 時，交換同一列任兩個不同顏色，完整保持 c/g 的符號個數與 complementary supports。
任意同 contents 的排列都可由這種交換到達。
改變 contents 時，一列將 g+/g− 兩個位置換成 c+/c−，另一列反向轉換。
各列 e/o 不變，n_c 在一列增加 1、另一列減少 1，總 c-weight 仍是 d。
已驗證的 n_c 區間是連續的；相鄰可行 contents 的 donor counts 非負且可實現。
因此這些轉換連接所有保留的 contents，不因硬約束切斷合法狀態空間。

## 聯合 energy 與完整 target 的恆等式

C=P_cA+P_cB，G=P_gA+P_gB；欲求的 targets 為
C_target(u)=[T_target(u)+T_target(u+d)]/4，
G_target(u)=(-1)^u[T_target(u)−T_target(u+d)]/4。
對任何四色狀態，直接展開得到

T(u)=2C(u)+2(-1)^uG(u)，T(u+d)=2C(u)−2(-1)^uG(u)。

若 Ec、Eg 是兩個 normalized profiles 在 1..floor(d/2) 上的平方殘差總和，
完整 fixed-target energy 使用原專案的 1/16 單位，則

**E_full=(Ec+Eg)/2。**

原 binary half shift 及 origin 已由狀態空間精確滿足，所以沒有遺漏自對稱位移。
Ec+Eg=0 當且僅當完整目標成立；仍需獨立 full verifier 驗證及去重才寫入結果。
這讓搜尋同時靠近 C、G，不以只有 C 精確匹配當作最終成果。

實作 `solver/torch_joint_search.py` 共用已驗證的三值 swap 增量，
transfer 亦包含 two-site quadratic cross terms 與 origin；不套用錯誤的 binary flip 公式。
退火溫度、有限 proposals 及候選排序仍是啟發式，bounded miss 不能作不存在證明。
測試 `tests/test_torch_joint_search.py` 使用原長度 literal binary space 及整數 PACF 作 oracle。

## 本次文獻核對的適用範圍

[Dual-Step Optimization for Binary Sequences with High Merit Factors](https://arxiv.org/pdf/2409.07222)
（2025 期刊版本）討論 GPU 候選生成再由較自由的搜尋改善。
原題為 aperiodic LABS，skew-symmetry 是限制族，不保證包含一般 PQCP 解；
其 Bloom filter 也可能 false positive，不能作本專案的嚴格排除證明。
periodic PACF 對旋轉精確不變，亦不能照搬原文以旋轉跨越 aperiodic energy 的作法。

[A Knowledge Compilation Take on Binary Polynomial Optimization](https://arxiv.org/pdf/2311.00149)
（2026 期刊版本）給出依 hypergraph 結構的求解結果。
本專案尚未證明其 dense cyclic constraints 符合 bounded incidence treewidth 或 β-acyclic 條件，
不能直接引用多項式時間或效能保證。
上述四色必要條件、連通 moves 與 energy 恆等式都是另外推導。

## 已完成的檢查與後續修正

2048 lanes×8192 步、seeds 123/456/789，三次最佳 full energy 都是 20，沒有完整新解。
各次 top 32 候選接到 full cyclic repair，每個最多 8 supports，亦無完整命中。
這是有限導航與有界支持修復，不能推出不存在。

加入初始 binary pair 的反向編碼 `binary_joint_rows`，
只接受 half-shift zero 且具有相符四個 parity counts 的輸入。
由 GS 原料自行構造的近解可從 full energy 8 起跑；並非從既有 PQCP inventory 取出已知解。
退火溫度可以明確設定，零候選不受 export cap 或取消前未開始迭代影響。

原本 transfer 提案在隨機位置抽樣後檢查 g+/g− 及 c+/c−，可能大量抽到不合法位置。
`solver/torch_joint_transfer.py` 直接在所需 source colors 中抽樣，保留同一完整 move graph。
content boundaries 由 donor counts 及相鄰 n_c 區間精確判斷。
專項測試完整比對小狀態的兩種方式所得合法移動集合，並核對所有 increment。
`direct_transfers=False` 保留舊提案路徑，方便比較；此選項尚未加入 main 預設。

GS 近解的 1024 lanes×32768 步、三個 seeds、三種起始溫度（0.5、2、4）已完成九次，
最佳 full energy 均停在 8，沒有新解。T=4 的舊位置提案三次均沒有 transfer 獲接受。
相同 batch、步數與 seeds 的 direct-color 提案三次 transfer accepts 為 1、2、6，
最佳仍為 8、完整新解仍為零，且各次約 144..147 秒，舊方法約 130..133 秒。
因此直接抽樣改善了合法提案覆蓋，這次仍沒有 discovery 收益，不能稱為加速成功。

更接近原長度的交錯推導、精確局部配對及驗證範圍，見
[交錯構造與精確局部剪枝](L86_interleave_derivation_20261007.md)。
