# 同時使用範數與交叉 trace 的完整半列補全

以下是本專案從原始偶數／奇數 PACF 自行推導，並非直接引用論文的 PQCP86 定理。
在 R=F₂[X]/(X⁴³+1)，star 表示 X↦X⁻¹。
R 分解成 F₂ 及三個 F_(2¹⁴)，在每個非純量分量 star=u↦u^q，q=128。

固定未知列的另一半列 W，同時要求

\[
UU^*=N,\qquad WU^*+W^*U=H.
\]

N 來自精確 intersection profile 模 2，H 來自奇數位移方程模 2。
因此它們只是必要條件，保留者还必須通過全部整數原始條件。

## 從實際 interleave 得到 H

將實際 odd 半列中心化：令 s=(43+1)/2=22，Ō_j=O_{j-s}。
E 不變。43 點 centered 位移 u 對應原始 odd 位移 (2u+43) mod 86。
對兩個 negative masks 的對稱 cross count 定義
C_(E,Ō)(u)=|E∩(Ō+u)|+|E∩(Ō-u)|。
固定另一整列後，未知／companion 的完整整數 cross target 是

\[
D(u)=\lambda+T((2u+43)\bmod86)/4-C_{\mathrm{other}}(u).
\]

H 的係數是 D(u) mod 2，W 是同整列的 centered 已知半列。
先核對 2max(0,w_U+w_W-43)≤D(u)≤2min(w_U,w_W) 及 D(0) 偶數。
这些也是必要條件，不能代替完整 cross counts。
free-O 的 CRT mask 需轉回實際 O_i=Ō_{i+s}。
cross trace 已固定相對 phase，不可再對 U 使用 rotation quotient。

## 每個域的完整求根

令 n,h,w 是 N,H,W 的域分量，n,h 固定於 F_q。

1. w=0,h≠0：無解。
2. w=h=0：回退到原本完整 norm preimages，保留全部 129 個或零 norm 的唯一根。
3. w≠0,h=0：令 u=w·v，trace 强迫 v∈F_q；
   v²=n/Norm(w)，平方在 F_q 為雙射，故唯一 v=(n/Norm(w))^(q/2)。
4. w≠0,h≠0：令 u=(h/w^q)z，則 z^q=z+1，
   z(z+1)=n·Norm(w)/h²。
   z 屬於完整 Frobenius affine coset，映射 z↦z(z+1) 的兩個根恰為 z,z+1；
   枚舉 coset 的全部 q 個元素，按 norm 保留兩根，得到完整查表。

在三個 partner 分量都非零時，最多 2³=8 個 literal masks；
純量分量的 norm 決定 U 的 parity，trace 純量必须為零。
零 partner 分量可能帶來更大集合，所以「最多八個」不是無條件上界。
每個域根經 norm／trace 整數 audit，CRT 後再次核對 literal cyclic norm／trace。
未知位元的重量、全部 integer intersections、完整 L86 verifier 最後逐項核對。

## 實作、回退與測量

Python 參考：`solver/norm_trace_completion.py`；
整數 C 算術：`csrc/norm_trace.c`、`solver/norm_trace_native.py`。
`norm_trace_preimages(...,native=False)` 保留 Python 參考路徑；
`complete_parity_half_rows(...,trace_backend=False)` 回到 PSD 加完整 norm enumeration。
C 負回傳是錯誤，會中止，不能當作 UNSAT；零 partner 的完整回退仍由 Python 處理。
來源雜湊決定編譯產物，舊凍結來源不受影響。

L3／5 全部 partner、全部對稱 norm／trace targets 已與全部 binary 空間比對；
L11 核對零、純量、一般 partner 的全部可達 fibers。
L86 的小長度 counterpart L6 已核對兩種 backend 的完整解集合。
另以每個 L43 域的全部 16,384 個元素、兩個非零 partners，
核對全部 F_q×F_q targets 的 native roots；記錄見狀態文件。

原 Python trace 配對 `L86_half_row_trace_paired_20261007`，
舊 PSD backend 中位 5.111799334 秒、新 trace 5.199291291 秒，慢 1.7116%。
候選數較少本身不足以證明加速，所以該結果沒有用來改主程式預設。
後續加入 native field 算術及共用中心準備，需看新的凍結來源配對記錄。
所有配對至目前完整新解均為零，尚不能聲稱達成完整新解速度提升目標。

新版配對 `L86_half_row_trace_rotations_paired_20261007` 完成五次同260fibers回放，
中位 2.009983541 → 2.002979750秒，仅减少0.34845%；
兩側完整模型集合一致、完整新解均零。trace backend仍非主程式或半列實驗的預設。
Native之後保留全部独立literal cyclic audits，並用預計算linear CRT columns减少多項式運算。
這個求根核心另用於 [四點g構造](L86_same_O_weight4.md)，固定trace后每個g的全部域表可共用。
