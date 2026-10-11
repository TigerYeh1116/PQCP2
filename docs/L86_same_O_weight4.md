# 同一 O、四點差分 g：完整有限族搜尋

本專案自行推導並實作，延伸 [兩點 impulse／skew O](L86_impulse_skew_O.md)。
取 centered E_A=c+g、E_B=c-g，實際 odd 半列 O、-O，
c、g 為互補 trits。限制 Σc=9、Σg=2、ΣO=1，
g 有三個 +1、一個 -1；O 是任意43點二元符號列，不要求 skew 或 perfect。

## 全部 g 及 c

所有列共同平移可將 g 唯一的負位置移至0，
正位置為其餘42點中任選3點，共11,480個 literal g。
共同 centered reflection 同時反射 c,g,O 並保持 ±1 target，
故保留 g 與其 reflection 的較小者，恰5,740個 representatives。
每個被商掉的整個 c/O fiber 都有反射的完整對應。
這不是把某一固定 g fiber 的 O rotation 商掉；後者沒有使用。

c 支撐39、和9，因此負號重量15。
偶數方程是 P_c(u)+P_g(u)+P_O(u)=0。
O 重量21，所以 P_O(u)≡3 mod4，得到 P_c(u)≡1-P_g(u) mod4。
令 A 為 c 的 support indicator，c=A-2b；
模4的二次項消失，形成 GF₂ 線性方程

\[
\sum_i b_i(A_{i+u}+A_{i-u})
\equiv [P_A(u)-(1-P_g(u))]/2 \pmod2.
\]

再加入 Σb_i≡15 mod2，完整 RREF 給 affine base 與全部 kernel columns。
原始每個方程都核對 base／homogeneous columns；columns 的獨立性亦核對。
Native Gray enumeration 訪問全部2^nullity個assignments，再保留 exact weight15。
nullity／整數重量是不同條件，不能僅枚舉部分 orientations。

## 任意 O 的 anchored norm／trace

E_A 負號支撐為 b∪g_minus、E_B 為 b∪g_plus，重量16、18。
未換號原料 O 的兩列重量都是21，λ=33，因此

\[
N_O(u)=[33-N_{E_A}(u)-N_{E_B}(u)]/2,\qquad u\ne0.
\]

整數範圍、ΣN_O=441 及 [有理 PSD](L86_certified_profile_psd.md) 先排除。
odd 目標要求 sym-cross(g,O)(u)=2δ_{±1}(u)。
寫 O=1-2U，因 Σg=2，得到完整整數條件

\[
\sum_i g_i(U_{i+u}+U_{i-u})=2-\delta_{\pm1}(u).
\]

模2後令 W=g mod2（四點 support），則
WU*+W*U=X+X⁴²。
與上面的 N_O mod2 同時送入 [完整 norm／trace 求根](L86_norm_trace_pruning.md)。
每個 g 的 partner／trace 固定，因此可預先計算三個域所有128個 subfield
norm targets 的0／2根及其CRT lifts，每個 c 最多八個literal O。
trace polynomial 在每個 primitive-43 域都非零：
若 α+α⁻¹=0，會有α²=1，與α的order43矛盾。
所以 zero-partner 分量確定排除該整個 g fiber，沒有忽略 norm fallback 解。

保留 masks 逐項核對 O weight21、全部 odd integer equations、全部 even intersections。
Native 使用64位整數及向外的有理 cosine coefficients，無浮點 pruning。
輸出 buffer 若不足會擴大重跑；只有全部 hits 已存才回 complete=True。
每個完整 hit 在 Python 重新算 even／odd，再進原始86點独立 verifier，
與 κ=41、45 的 +4 profile 完全比對，才保存新等價類。

## 測試與範圍

L5／7／9 的每個三正一負 g、每個可行 c weight，
全部 affine outputs 已對照 literal c signs 的原始整數 mod4方程。
43點 g=(+於1,2,3，-於0) 的完整fiber為rank20、nullity19。
Native 與独立Python逐階段計數一致：524,288個affine assignments，
32,768個c weight matches、32,724個PSD排除、34個norm／trace空族、
80個literal O，32個O weight排除、48個odd integer排除、完整解零。
因此測試包含非空有限域候選；不是只核對空輸出。
另核對11,480個g與5,740個reflection representatives的完整覆蓋。

入口 `experiments/search_same_o_weight4.py` 可按IDE執行鍵；預設全5,740個g。
`DEFAULT_MAX_G=0` 表示全部；正值是明確的輸入截取，不能宣稱全族排除。
輸入、每個affine證書、完整計數、源碼manifest、初始inventory存於獨立results目錄。
此族不涵蓋一般PQCP86、不同O、其他g重量或所有target signs；
完整negative結果也只排除此構造族。完整新解速度20%目標仍需發現率證據。

## 全族已完成，以及完整 odd survivors

`results/L86_same_O_weight4_complete_20261007/report.json` complete=true，
5,740個g representatives全部決定，210個mod4 C systems不一致；
其餘rank20為210個、rank21為5,320個。
Native完整訪問1,504,706,560個affine assignments，
136,635,821個exact C重量符合者，135,403,737個有理PSD排除。
1,077,751個norm／trace空族；1,234,664個literal O中，
642,092個weight不符、592,386個odd integer不符、186個even integer不符。
完整解零、沒有UNKNOWN，elapsed=176.217270625秒。
只排除上述同O／g重量四／正號target的整個有限族。

Native另加入完整odd survivor buffer；不足時擴大重跑，保留全部186個literal C/O。
`experiments/extract_weight4_odd_seeds.py` 先核對原始5,740筆輸入／ledger／總計，
再重播其中177個非空odd fibers，全部原先計數必须相同。
每個seed重新核對全部原始odd位移及even residual energy，
目標能量（平方和/16）恰4Σ_{u=1}^{21}(N_O(u)-wanted(u))²。
得到186個一般PQCP等價類的非解中心，最低能量160，沒有比先前能量8更低。
這是另一種全部odd方程已解的起點，不能稱新解或以其數量推論搜尋速度。
完整記錄 `results/L86_weight4_odd_seeds_complete_20261007/report.json`，elapsed=12.3727305秒。
接著以任意43位元半列補全放寬同O及g重量限制，結果另存。
