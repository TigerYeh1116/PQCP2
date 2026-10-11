# 偶數支撐列的整體變號軌道與尾端投影

2026-10-08。這是新增的精確符號空間最佳化，沒有假設序列反射對稱。
完整新 L86 等價類目前仍為 0；以下實驗只衡量相同搜尋域的耗時。

## 軌道與負號數

固定兩列 literal 支撐 $S_A,S_B$，總重為 43。
至少一列 $S_e$ 有偶數重 $w>0$。若一列的 negative subset 是 $M_e$，
整體變號就是 $M_e\mapsto S_e\setminus M_e$。
它保留全部 signed norm，將 negative count $t$ 改成 $w-t$；$w$ 偶數也保留 negative parity。
因此整體變號方向 $R_e$ 位於完整模 4/count-parity 仿射問題的齊次核。

原空間為 $M_0+\operatorname{span}(T_1,\ldots,T_q)$。
以 Gaussian elimination 精確找出 $R_e=\sum_j\alpha_jT_j$。
取 $\alpha_j=1$ 的一個位置，去掉該 $T_j$，得到

\[
H=\operatorname{span}\{T_i:i\ne j\},\qquad
\operatorname{span}(T_1,\ldots,T_q)=H\oplus\langle R_e\rangle.
\]

因此 $M_0+H$ 含每個整體變號軌道的一個代表，總共 $2^{q-1}$ 個。
這不是直接固定一個 sign 並忘掉另一方向：最後檢查負號數是否屬於
$\{n,w-n\}$，必要時整列變號回指定的 exact negative count $n$。
兩方向的全部整數 norm 相同，所以一次 norm 核對足以決定這個軌道的存在性。
回傳的是第一個見證，並非全部完整等價類。

對任意已限定的子空間，若 $R_e$ 不在其核中，則回到原後端。
這避免改掉已固定的條件或 conditioned sign pattern。
原 assignment budget 小於 $2^q$ 時，也直接回退原後端，保留原來的部分枚舉順序、首個見證及 UNKNOWN。

## 模 8 的 count bit

仿射問題已固定 $t\equiv n\pmod2$。
寫 $w=2s$、$t=2a+p$，可得

\[
\operatorname{bit}_1(w-t)\oplus\operatorname{bit}_1(t)
\equiv s+p\equiv w/2+n\pmod2.
\]

若 $w-2n\equiv2\pmod4$，兩方向的 count 第二位元必然不同；
要求其中一方向等於指定 $n\pmod4$，對軌道而言不再是額外限制。
因此可以移除這一個 syndrome bit，再用 $\{n,w-n\}$ 的 exact count 決定方向。
若 $w-2n\equiv0\pmod4$，兩方向的 count bit 相同，仍保留該必要條件。
其他 norm/count bits 全部保留；任何接受見證仍核對全部整數 norm 與指定 count。

## 與原 hybrid tail 的相容性

每個 cut bit 是兩端 negative bits 的 XOR。整列變號不改任何 cut bit，
故 $R_e$ 是所有 norm 二次多項式的共同 polar radical。
negative-count 第二位元沿 $R_e$ 也只改變常數，沒有新增 quadratic cross term。

原 hybrid 把 tail 內 quadratic coefficient vectors 的 span 商掉，得到必要線性問題。
首版只縮減一個變數、保留原 tail 大小，可能把原本的低秩 tail 改成較高秩 tail，增加 projected candidates。
現在由 $q,t$ 改成 $q-1,t-1$（tail 至少 1，並按實際維度裁切），保留原 prefix depth。
若去掉的方向在原 tail 中，其 quadratic span 相同；
若在 prefix 中，新 tail 的 coefficient span 是原 span 的子空間，提供更強的必要線性條件。
全部 tail solutions 及其 nullspace 成員仍枚舉，沒有假設 column 獨立。

## 實作與證據

實作位於 `csrc/ternary_quadratic_signs.c` 的 `pqcp_quadratic_orbit_hybrid_signs`，
Python 選項是 `run_native_system(..., even_sign_orbit=True)`，
固定支撐流程是 `complete_pqcp_supports(..., even_sign_orbit=True)`。
獨立抽樣器的 `--even-sign-orbit-pruning` 啟用，反向選項回退；目前預設關閉。

小長度 3、5、7、9、11 逐一核對全部原始 signed words：SAT/UNSAT、exact words、
完整空間覆蓋與 partial-budget rollback。實際 43-point C 見證與 G-UNSAT 域包含 sigma 4 及 6，
也核對所有整數 PACF、支撐與 counts。相關 20 項 focused tests 已通過。

三次 65,536-trial 配對：

- 首版 `L86_even_sign_orbit_pipeline_paired_20261008`，同域中位數耗時下降約 8.20%；
- 維持 tail span 的版本 `L86_even_sign_orbit_tail_preserved_pipeline_paired_20261008`，約 5.86%；
- 更強的 $t-1$ tail 版本 `L86_even_sign_orbit_stronger_tail_pipeline_paired_20261008`，約 7.02%。

第三項以凍結的原始 C loop 作回退核心，另加一個不被回退路徑呼叫的相容性 export。
兩邊支撐、targets、contents 與完整決定相同；所有中間 signed 見證重新核對。
CBC 首個見證可能不同，不能聲稱逐字相同；完整判定與原始 binary verifier 才是核對依據。
兩邊均無 UNKNOWN、無完整新解。因此此項尚無單獨提升 20% 的證據，保留可選介面。

另一次相同設定重複存於 `L86_even_sign_orbit_stronger_tail_shared_choices_paired_20261008`，
以其 terminal report 為準，不把仍在跑的 children 當成完成。
