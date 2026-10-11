# L86：交替符號與循環置換的充分構造

日期：2026-10-07。以下構造為本專案自行推導。
它是額外限制下的搜尋，不是一般 PQCP 的必要條件。
完整實驗沒有找到 L86 解，所以沒有加入 main 的預設搜尋。

## 從充分關係推導精確目標

令 x∈{−1,+1}^86，取奇數 unit multiplier v，並設

\[
y_i=(-1)^i x_{vi\bmod86}.
\]

直接換索引得 P_y(u)=(-1)^uP_x(vu)。
若 target 在 ±2k 為 +4，則獨立 even shifts 滿足
P_x(2t)+P_x(2vt)=4δ_{±k}(t)。
偶／奇子序列 E_r=x_(2r)、O_r=x_(2r+1) 都是長度 43 的 binary signs，
所以 P_x(2t)=P_E(t)+P_O(t)≡2 (mod 4)。
projective multiplier group (Z/43Z)^×/{±1} 的階數為 21。
若 v 不能遍歷全部 21 個 independent nonzero shifts，某個 odd-length cycle
不含 target；其上的 f(t)+f(vt)=0 強制 f=0，違反 f≡2 (mod 4)。
因此只需檢查 projective order 21 的 multiplier。

在 canonical positive-even profile，ΣE=9、ΣO=3。
odd PACF 在 42 個非半週期 odd shifts 上同值，記作 a；半週期為 b。
因 Σ_odd P_x=2(ΣE)(ΣO)=54，42a+b=54。
又 a≡2 (mod 4)、|b|≤86，唯一可能為 a=2、b=−30。

取 O'_r=O_(r+21)，定義 c=(E+O')/2、h=(E−O')/2。
c,h 為 complementary-support 三值序列，Σc=6、Σh=3。
展開 even PACF 得 P_x(2t)=2[P_c(t)+P_h(t)]；
展開 odd PACF 得

\[
P_c-P_h=J-16\delta_0.
\]

寫 P_x(2t)=2ε(t)，則 ε(t)+ε(vt)=2δ_{±k}(t)。
每個 cycle 只有一個 target 節點，長度為 21，因此其唯一解為：
節點 k、vk 上 ε=+1，之後沿 v-orbit 交替 −1,+1，直到最後為 −1。
得到 ε(t)∈{−1,+1}、11 個 positive 與 10 個 negative independent shifts。
所以 unary targets 為

\[
P_c(0)=14,\quad P_c(t)=(ε(t)+1)/2;
\qquad P_h(0)=29,\quad P_h(t)=(ε(t)-1)/2\quad(t\ne0).
\]

c 有 4 個負號，h 有 13 個負號。
只要在 complementary supports 上各找到一個精確 unary witness，即可組成 binary E,O'。
coarse／difference 的原索引為
c_A(i)=c_(22i)、g_A(i)=h_(22i)、c_B(i)=h_(22vi)、g_B(i)=c_(22vi)，
全部索引 mod 43。以既有 `reconstruct_odd_pair` 重建 x,y，
再獨立檢查完整 86 點 PACF、奇偶個數，以及 y_i=(-1)^i x_(vi)。
此流程不要求某個已知 L86 解作種子。

## 完整有限域 supports 枚舉

在 F₂[X]/(X^43+1)，U 為 c 的 support，故 UU*=P_c mod 2。
新 `cyclic_norm_family` 支援任何 symmetric F₂ target，
以原來已驗證的 CRT／field-norm 演算法列出所有 U。
每個非零 degree-14 component 的 norm fibre 有 129 個元素。
共同旋轉 c,h 不改 unary PACF 且保留 complementary supports，
所以可取每個 rotation orbit 一個代表，最多 129³/43=49,923。
它只用於完整 family 的對稱商，不能替換固定 reference 的 nearest-Hamming 搜尋。
weight=14 的所有代表再交给既有 native 精確符號枚舉，沒有捨棄 homometric preimages。

實作：`solver/automorphism_construction.py`；
重現：`experiments/search_automorphism_construction.py`。
UNKNOWN 及取消都保留不完整狀態；不以預算中止代替 UNSAT。

## 驗證與結論的範圍

`tests/test_automorphism_construction.py` 用 d=3、5、11 的全部 literal masks
驗證 general norm targets 與每個旋轉 orbit；
亦用長度 6、22 的 literal signed supports 驗證構造參數的 SAT／UNSAT，
並對產生的 binary pairs 直接重算原長度 PACF。
長度 6 與部分長度 22 的參數確實產生完整 PQCP。

L86 的 12 個 projective multipliers ×21 shifts 共 252 組參數全部完成，
掃描 12,580,596 個參數／rotation-orbit 組合，測試 453,936 個參數／weight-14 支持組合。
全部被 unary C 問題完整排除，沒有 UNKNOWN、沒有完整 L86 解。
11.05 秒僅是這個受限構造的完整枚舉時間，不能與一般 L86 搜尋速度比較。
記錄：`results/L86_automorphism_construction_20261007/report.json`。
此結論只排除上述 unary profiles 的构造，不是一般 L86 無解證明。
