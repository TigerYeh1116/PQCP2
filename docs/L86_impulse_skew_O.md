# 任意 skew O 的 impulse-difference 構造

此族允許 O 不是 perfect；它是獨立受限構造，沒有加入一般主搜尋的必要條件。
在 centered interleave 中取 E_A=c+g、E_B=c-g、O_A=O、O_B=O，
實際 y odd 半列為 -O。c,g 為互補支撐的 trits，g=δ₁+δ₋₁，Σc=9。
任意 sum-one skew O 滿足 O(u)+O(-u)=2δ₀(u)。
故 odd full profile 等於 2·sym-cross(g,O)=4g，
原始非零位置 κ=41、45，正好是嚴格 +4 目標。
同 unit 可將非零 impulse 位移化為 1；兩個位置與 midpoint 都完整保留。

## c 的完整枚舉縮減

偶數條件是 P_c(u)+P_g(u)+P_O(u)=0。
所有 O 的重量為 21，所以非零 P_O(u)≡3 mod 4；
P_g(u) 在 u=±2 為 1，其餘非零位移為 0。
因此 P_c(u)≡0 mod 4 於 ±2、其餘 ≡1 mod 4。

在 c 的支撐（排除 ±1）上寫 c_i=1-2b_i。
模 4 展開把二次項消去，未知只涉及 reflected pairs b_i⊕b_-i。
21 個位移方程的 GF₂ 矩陣 rank=20，決定全部 20 個 pair XOR：

opposite pairs 為 2,5,6,9,10,13,14,17,18,21；
equal pairs 為 3,4,7,8,11,12,15,16,19,20。
Σc=9 等價於 negative weight 16。十個 opposite pairs 固定貢獻十個負號，
所以 b₀=0、equal pairs 中恰三對同為負號。
全部 c 共 binom(10,3)·2¹⁰=122,880 個，逐位 orientation 全部保留。

## skew norm 求根

E_A 的负號支撐是 C，E_B 的是 C∪{±1}，重量16／18。
O 的負號支撐 U 重量21；四列偶數條件要求
2N_U(u)=33-N_EA(u)-N_EB(u)。
先检查整數／有理 PSD，再把完整 profile 模2。
skew 進一步要求 U+U*=J+1，所以每個非純量域有 u^q=u+1。
全部 q 個 affine coset 元素，按 u(u+1) 求根，每個 norm 0 或 2 根；
三個域合成至多八個 literal masks，全部核對 skew、重量及整數 intersections。
不能在 skew root 上用 rotation quotient，因 skew center 已固定。
每個保留者仍由全長 verifier 決定。

## 已完成的有限族結果

`results/L86_impulse_skew_O_complete_20261007/report.json` complete=true：
全部122,880個 c，PSD 排除120,882個，剩餘有1,492個 skew norm 空族；
4,048個 literal masks 進入整數檢查，完整新解零，elapsed=3.4831215秒。
所以這整個特定族已排除；沒有排除一般 skew O／其他 g／所有 PQCP86。
入口 `experiments/search_impulse_skew_o.py` 可在 IDE 直接執行。

## 下一個放寬：g 的重量為四

保持 Σg=2，可取三個+1、一個-1，不再強迫 O skew。
令 W=g mod2，未知 O 的 binary negative support 为 U。
odd 模4 必要條件變為 WU*+W*U=X^t+X^-t，
因此 [一般 norm／trace 完整求根](L86_norm_trace_pruning.md) 可使用。
但 c 的模4系統有更多自由位元，需要有理 PSD 與整數必要條件提前剪枝。
此 g-weight-four 搜尋已另行實作，見 [四點 g／任意 O](L86_same_O_weight4.md)。
上面的 finite-family UNSAT 不適用於它；其實際覆蓋及結果單獨保存。
