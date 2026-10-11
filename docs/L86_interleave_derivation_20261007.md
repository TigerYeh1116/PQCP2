# L86：交錯構造、整數配對與精確局部剪枝

2026-10-07。以下為另行推導並實作的研究分支。尚未找到完整新 L86 解，
也尚未證明新解發現速度提高 20%。本次新分支沒有取代 main 的預設搜尋。

## 文獻與資料能提供什麼

2026 ICC 的 [Huang 等人 PQCP 論文](https://repository.essex.ac.uk/42886/1/Final%20Version.pdf)
Definition 7 約束最大週期相關值；Project 2 第三題則指定恰好兩個非零 ±4 旁瓣。
兩者的可行集合不同，不能把該文的例子直接算成 L86 解。

2026 年 [Capelli、Del Pia、Di Gregorio 的 BPO 論文](https://doi.org/10.1007/s10107-026-02375-9)
提供有結構超圖及 cardinality constraints 的求解背景。
我們尚未證明本題符合其結構假設，因此不引用其多項式時間保證。

可實際作為原料的是 [Đoković、Kotsireas 的 GS 論文附錄 7.6](https://arxiv.org/html/1802.00556v1#S7.SS6)
與 [Koukouvinos 提供的 D-optimal 表](https://documents.uow.edu.au/~jennie/D-optimal/order.86.html)。
表中矩陣的 order 86 是兩列各長 43 的 D-optimal 設計，並非完整 L86 PQCP。
GS 的四列也是長度 43 原料；每組均先以原始 signs 重算全部整數 PACF。

D-optimal 表共 17 組，其中 16 組通過；第二類的 A_4/B_4 不滿足宣稱的相關恆等式。
它的單一等權交換修復鄰域亦完整檢查，沒有合格修復。保留原始來源，明確排除此組。
GS 附錄的 10 組四元組全部通過整數驗證，其中 3 組的 row sums 是 1、1、7、11。
另讀取 [Curtis Bright 的 Williamson 原料](https://cs.uwaterloo.ca/~cbright/williamson/)，
其 L43 兩組 row-sum 絕對值為 (1,5,5,11) 及 (5,7,7,7)，不能直接作本分支的 11/7/1/1 原料。

## 交錯恆等式：先精確滿足所有偶數位移

令 d=43，E_A、E_B、O_A、O_B 為 ±1 序列，定義

\[
x_{2i}=E_A(i),\quad x_{2i+1}=O_A(i+\delta_A),
\qquad y_{2i}=E_B(i),\quad y_{2i+1}=-O_B(i+\delta_B).
\]

直接拆開偶數及奇數索引，有

\[
T(2u)=P_{E_A}(u)+P_{E_B}(u)+P_{O_A}(u)+P_{O_B}(u).
\]

四列為 GS complementary quadruple 時，所有非零偶數位移恆等於零。
另一個充分條件是 E 的 D-optimal pair 在非零位移的 PACF sum 為 2，
而兩個 O 都是 PACF=-1 的 perfect rows。這是構造族的充分條件，不是一般 PQCP 的必要條件。

row sums 選 11、7、1、1，可得到 sum(x)=12、sum(y)=6，
Alt(x)=10、Alt(y)=8，因此 DC 平方和 180，alternating 平方和 164。
這符合正號、奇數 k 的必要 character identities，且四個 ones counts 是 (16,21,18,22)。

令 CC(E,O;t)=sum_i E(i+t)O(i)，則

\[
H(E,O;\delta,r)=CC(r+1-\delta)+CC(-r-\delta),
\quad T(2r+1)=H_A(r)-H_B(r).
\]

r=0..21 覆蓋全部獨立奇數位移；r=21 是自對称的半週期位移 43，必須為零。
完整目標因此等價於 H_A-H_B 在 r=0..20 恰有一個 +4，其他座標皆為零。
`solver/interleaving.py` 將每個完整整數向量映射至所有 offsets/preimages，
用精確 tuple equality 配對。字典 hash collision 不會造成漏解或誤判。

## 擴大有限原料族，而不是憑短測放棄

1. 完整生成 8 個 perfect-43 translation classes；各列重新驗證所有非零 PACF=-1。
2. D-optimal 合格 11/7 原料與所有 perfect O 的相對旋轉／獨立反轉：115,584 次精確配對，完整無命中。
3. 三組 GS 原料的全部相對旋轉、獨立反轉與兩個 O 的分配：21,672 次配對，無完整命中，產生目標能量 8 的近解。
4. 將合格來源的個別 rows 閉包於全部 unit decimations 與 rotations，再跨來源配對。
   catalog 有 126 個 weight-16、105 個 weight-18、1,141 個 weight-21 rows。
   651,511 次 even-profile 配對得到 258 個四元組構造族，全部完成 odd matching，仍無完整命中。

catalog 只商掉共同 unit decimation 及各原料的 rotation：全部奇數 k 及相對 offsets 仍被搜尋，
所以這個構造族的覆蓋不因商空間而減少。沒有單獨商掉可能改變 odd profiles 的 row reflection。
所有同 profile 的原料保留，不以第一個 homometric 代表取代其他 preimages。

上述族的完整無命中，只排除明確定義的構造族，不能排除一般 L86。
近解的能量是原專案獨立位移平方殘差的 1/16 單位；能量 8 不等於 PQCP。

## 精確局部剪枝：兩列可分離的 meet-in-the-middle

中心 (a,b) 有固定四個奇偶 ones counts。令 a'、b' 是各列候選，
Δ_a=P_a'−P_a、Δ_b=P_b'−P_b。完整 target 必須滿足

\[
\Delta_a+\Delta_b=T_{target}-T_{center}.
\]

這是原始 PACF 的整數恆等式，兩列之間沒有交叉項。
每列改變保持奇偶 counts 時，Δ/4 為整數。
字典記錄所有相同 Δ 的 literal flip sets；匹配後仍重算原始完整 binary PACF。

`exact_swap_repair.py` 完整檢查每列至多一次同奇偶、異符號交換。
每個 L86 中心有 895×913=817,135 個這類配置，21 個相容目標只需小側向量查詢。

`hamming4_repair.py` 再加入單列四位元改變，完整覆蓋固定 contents 的總 Hamming 距離 ≤4。
單列四位元配置分類為「同一 parity 兩個 0 與兩個 1」或「每個 parity 各一個 0 與一個 1」。
這兩類互斥且完整；兩列分別有 290,214 與 302,310 種。
原始整數 PACF 在某座標不可能符合目標時，才提早排除當前 literal 配置。
Native 使用 128-bit masks；跨越 bit 64 的 rotations 已與獨立整數 oracle 比較。

`hamming6_join.py` 進一步完整搜尋 4+2 與 2+4 flips。
以較小一列的 delta 建立 required vectors，較大一列的四位元配置分批產生，
每批有界且不保存 GPU tensors。不包括單列六位元改變，因此不能稱完整 radius-6 排除。

`hamming8_join.py` 搜尋 4+4 flips。
先以 12 個獨立位移作必要的 prefix equality，再檢查全部 43 座標。
目標位置在 prefix 以外時，零 target prefix 也保留；所以不會漏掉遠端非零旁瓣。
prefix 字典保留所有完整 profile 及所有 masks，沒有概率指紋或 Bloom filter。
此法不包括 6+2、2+6 或單列八位元，不能稱完整 radius-8 排除。

六個 GS 近解已完成 radius-4、4+2/2+4、4+4 的上述精確搜尋，均無完整解。
4+4 每個中心的兩列笛卡兒積有 87,734,594,340 個配置，
實際只需索引 290,214 個 profiles、串流 302,310 個 profiles，再作必要前綴匹配。
這個數量縮減是局部精確匹配的數學結果，不能當作新解發現速度提高 20% 的證據。

## 接到現有流程的位置及實驗記錄

`experiments/repair_gs43_supports.py` 已將 GS 近解接到既有 `repair_cyclic_supports(..., full_completion=True)`。
30 個目標各檢查最多 256 supports，7,680 次完整支持修復均 bounded miss；不能作全域 UNSAT。
`experiments/refine_gs43_seeds.py` 可將近解輸入 C/G 四色搜尋；初始 pair 需通過 content、half shift 及 inverse encoding 核對。
新 `direct_transfers` 只在同一移動圖中直接抽取合法來源顏色；並沒有額外禁止合法移動。
其 proposal probabilities 仍是啟發式，不宣稱退火收斂或平穩分布。

原始資料與 report 均放在 `results/L86_*_20261007/`，原本 44/46/68 inventories 未改動。
關鍵記錄：`L86_GS_interleave`、`L86_interleave_catalog`、`L86_GS_hamming4`、
`L86_GS_four_two`、`L86_GS_four_four`、`L86_GS_cyclic_repair`、`L86_GS_joint_refinement`。
本輪一般回歸為 1,328 passed、14 skipped；之後新增 4+2/4+4 的 6 個專項測試通過。
兩個既有環境 benchmark modules 沒有納入一般回歸。沒有把這些測試數當成搜尋效能證據。

目前繼續測試的下一條路徑為 [必要支持條件內的直接抽樣](L86_independent_norm_sampling.md)，
用未依附於近解的 literal CRT preimages 進入既有完整 C/G completion。
