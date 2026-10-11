# L86：經證明的三次消元與尾端線性剪枝

## 結果與目前採用狀態

新增 `solver/mod8_crossbred.py`，並完成數學證書與候選集合驗證。
這是研究原型，尚未接入預設搜尋；目前 Python 前處理成本高於既有原生求解器。
沒有找到新的完整 L86 PQCP，也沒有達成「找到新等價類快 20%」的驗證標準。

32 個真實 L43 符號問題的診斷，共抽樣 16,384 個前綴：

| 統計 | 原尾端商空間投影 | 新三次消元 |
|---|---:|---:|
| 保留前綴 | 3,133 | 351 |
| 保留尾端候選 | 4,070 | 426 |

新條件額外排除原條件保留前綴的 88.7967%。這是抽樣剪枝效果，**不是時間加速百分比**。
獨立 C Gray-code 參考程式對每個問題完整枚舉全部 \(2^{21}\) 配置，
32 個問題的全部 33 個模 8 解都通過新條件；其中含 16 個確實有整數符號解的正控制。
每條新增方程的原始多項式組合證書都以獨立的符號集合運算重播。

Python 消元及證書前處理中位數 64.356 ms，原生完整符號求解中位數 1.682 ms。
這表示目前原型會拖慢流程，不能因為候選減少就設為預設。
詳細記錄：`results/L86_mod8_crossbred_degree3_audit_20261008/report.json`。
樣本沿用含 C 解的 fixture bank，其 C/G 各半比例不代表隨機搜尋分布。

## 文獻查核：哪些能用，哪些需重新推導

1. Sakata、Takagi，**An Efficient Variant of F4 Algorithm for Solving MQ Problem**，
   TCHES 2026，DOI 10.46586/tches.v2026.i3.1284-1309。
   [作者 ePrint，2026-04-15 修訂](https://eprint.iacr.org/2023/1650)
   及 [東京大學 2026-07-22 發表資訊](https://www.i.u-tokyo.ac.jp/news/press/2026/202607222841_e.shtml)。
   查核到官方摘要及刊登資訊；本次未成功取得其最新 PDF 全文。
   它利用 Hilbert-driven 排程及 S-polynomial 選擇降低矩陣成本，
   官方摘要中的新記錄是 \(\mathbb F_{31}\) MQ，不能直接當作本專案
   \(\mathbb F_2\) 結構化 PACF 方程的效率或完備性保證。
   本實作不以預估 Hilbert 級數作為排除條件。

2. Joux、Mora，**The regular multivariate quadratic problem**，2025。
   [期刊全文](https://link.springer.com/article/10.1007/s10623-025-01717-6)。
   文中每個區塊恰好一個非零元素的 regular constraint，不等於本專案固定負號數。
   其中 hybrid specialization 與線性消元的組織方式可參考；
   泛型／semi-regularity 下的複雜度不能作為 PQCP 剪枝的證明。

3. Joux、Vitse，**A crossbred algorithm for solving Boolean polynomial systems**，2017。
   [作者 ePrint](https://eprint.iacr.org/2017/372)。
   此法較早，但直接對應「先多項式消元、再固定部分變數」。
   本次取得 ePrint 摘要；PDF 直接讀取被拒。
   方法細節另核對下列作者公開的完整論文，避免只憑摘要實作。

4. Bouillaguet、Delaplace、Trimoska，**A Simple Deterministic Algorithm for Systems
   of Quadratic Polynomials over F2**，SOSA 2022。
   [作者公開全文](https://perso.lip6.fr/Charles.Bouillaguet/static/publis/MQ_simple.pdf)。
   說明固定部分變數後線性化，並討論 Crossbred。
   本專案自行證明必要條件及完整尾端枚舉，不沿用其平均時間假設。

5. May、Ostuzzi、Ressler，**Just Guess: Improved (Quantum) Algorithm for the
   Underdetermined MQ problem**，EUROCRYPT 2026。
   [作者 ePrint，2026-02-20 修訂](https://eprint.iacr.org/2025/1788)。
   查核官方摘要。它針對 underdetermined MQ；本批常見 21 自由變數、最多 23 個
   模 8 方程，未符合直接套用的情境。未把純猜測失敗或量子 Grover 步驟當成排除證明。

## 自行推導

既有模 4 Gaussian elimination 後，符號位元由自由變數
\(z\in\mathbb F_2^q\) 精確參數化。既有 PACF 模 8 及每列負號數模 4
必要條件為二次 Boolean 多項式 \(f_1(z),\ldots,f_m(z)\)。
其推導見 `L86_mod8_quadratic_pruning.md`。

在 Boolean ring

\[
R=\mathbb F_2[z_0,\ldots,z_{q-1}]/(z_i^2-z_i)
\]

中，將 \(z=(p,t)\) 分成前綴與尾端。
對每個原方程生成 \(f_i\) 及選定的 \(z_jf_i\)，
所有單項式都以 \(z_i^2=z_i\) 化為 square-free，總次數至多三。
單項式用變數集合的 bit mask 表示，所以乘法是 mask 的 bitwise OR，
同一單項式出現兩次則 XOR 相消。

將單項式分為「尾端次數至少二」與「尾端次數至多一」兩類，
優先對前一類進行 GF(2) Gaussian elimination。
留下的方程具有

\[
g_\nu(p,t)=h_\nu(p)+\sum_{j=0}^{|t|-1}a_{\nu j}(p)t_j=0,
\qquad \deg h_\nu\le3,\quad\deg a_{\nu j}\le2.
\]

每條 \(g_\nu\) 另保存一個 bitset 證書，指出它是哪些
\(f_i\) 或 \(z_j f_i\) 的 XOR。
因此所有原方程的解必然滿足所有 \(g_\nu=0\)。
這個結論不需要假設原方程互相獨立或具有泛型代數結構。

對固定的 \(p=p_0\)，剩下的問題只是

\[
A(p_0)t=h(p_0)
\]

的 GF(2) 線性相容性。
若不相容，該前綴底下所有 \(2^{|t|}\) 個尾端都無法滿足原模 8 系統，
可以整個排除。若相容，保留 particular solution 加上**完整 nullspace**
的每個成員，再檢查原模 8 方程、精確負號數與完整整數 PACF。
最後 C 與實際 G target 的解仍須重建並通過原始 binary L86 verifier。

此前的全域三次消元未取得新的全域線性方程，與本次結果不衝突：
本次只要求對尾端線性，允許前綴係數仍為二次／三次多項式。

## 與原投影的關係

原方法商掉所有尾端二次係數的 span，得到對尾端線性的原二次方程組合。
本方法的輸入包含所有 \(f_i\)，因此保留下來的低尾端次數 row space
包含這些原投影條件。新增 \(z_jf_i\) 提供更多可消元的必要條件。
故在同一分割下，新方法保留的尾端集合是原方法保留集合的子集。
診斷對每個抽樣前綴逐一檢查了這個包含關係。

實驗使用 \(q=21\)、尾端六個變數，以及前八個前綴變數和全部尾端變數作乘數。
最多生成 \(23(1+8+6)=345\) 行。
全部三次以下 square-free 單項式共有
\(1+21+\binom{21}{2}+\binom{21}{3}=1562\) 個。
需要消去的尾端次數至少二的 columns 有

\[
(1+15)\binom62+\binom63=260.
\]

只使用部分乘數會減弱剪枝，不會破壞其必要性。
所有原始方程仍包含在生成矩陣內；未訪問的搜尋分支不能記為 UNSAT。

## 驗證與導入門檻

`tests/test_mod8_crossbred.py` 對 q=0..8 的隨機系統，逐一比較全部
原模 8 零點與新條件保留後的零點集合；亦驗證空方程、常數矛盾、
dependent columns、完整 nullspace 與部分乘數。
與 GPU 模 8 helper 的測試合計 **62 passed**。

要導入預設搜尋，下一步先將 bitset 消元移到原生 C／Metal，
保留同一證書格式與獨立重播驗證，再比較同一批實際搜尋 supports 的
完整 C/G 判定、前處理加求解總時間及峰值記憶體。
既有 GPU 商空間原型已通過 80 個小問題及 32 個真實問題的完整模 8 集合核對，
並確認混合 q=0/q=21 批次和緩衝區溢出回 UNKNOWN。
它尚未整合此三次消元，也尚未設為預設。

GPU 符號子問題與本次更強剪枝都只是搜尋的部分成本。
只有實際找到經完整驗證且等價類去重的新 L86 PQCP，
再與相同搜尋域、相同前綴的既有方法比較首次發現時間，
才能檢驗使用者要求的「找到新解速度提升至少 20%」。
