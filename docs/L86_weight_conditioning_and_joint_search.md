# L86：負號個數剪枝、雙側位置擴展與精確 lift

2026-10-06～07。此階段仍未達成「找到全新 L86 PQCP 的速度提高至少 20%」。
已經驗證的是固定位置符號補全及 CRT 位置枚舉的速度改善，
並首次在此輪保存的 reference 重播中找到一個精確壓縮候選。
不能把子階段的效能百分比當成完整新解的 discovery rate。

## 文獻如何使用

另外閱讀 SAT 2025 的
[Problem Partitioning via Proof Prefixes](https://drops.dagstuhl.de/storage/00lipics/lipics-vol341-sat2025/html/LIPIcs.SAT.2025.3/LIPIcs.SAT.2025.3.html)
（Battleman、Reeves、Heule，DOI 10.4230/LIPIcs.SAT.2025.3）。
它包含依 cardinality constraint 語義分割 SAT 問題的方法。
這提供按真正負號個數分割的研究方向；以下 symbolic RHS 高斯消去、
整數枚舉與奇數 lift 公式為針對本專案自行推導，
不是聲稱直接實作該論文的 totalizer 或 proof-prefix 工具。

## 1. 精確的負號個數剪枝

固定非零位置後，必要的 mod 4 方程與兩側個數奇偶性經 GF(2) 消去，
得到所有符號解 s=c+Bf，f∈F₂^q。q 通常為 21。
舊核心枚舉全部 2^q 個 f，再逐一檢查兩側負號個數。

選一側 R，共 m 個非零位置，真正負號個數必須恰為 n。
列出全部 t∈F₂^m、wt(t)=n，再解

**B_R f=t+c_R。**

令 r=rank(B_R)。以 t 作為 symbolic RHS 做高斯消去：
依賴列產生 t 的線性相容條件；相容的每個 t 恰有 2^(q−r) 個 preimages。
因此新枚舉的集合恰等於舊 affine space 中滿足這一側精確個數的集合。
不同 t 的集合互斥，沒有重複，也沒有漏掉符合原方程與個數的解。

若 n=0，全部此側符號被固定為正號；若 n=m，則全部為負號。
這是必要條件，不是根據經驗的排除。
一般情形的枚舉上界為 binomial(m,n)×2^(q−r)。
相容性檢查還可能讓實際數量更少。

自動選擇條件是 binomial(m,n)≤4096、剩餘自由變數至少 7，
且上界不大於原空間的 80%；否則用舊核心。
此選擇只是避免枚舉分支的管理成本，不是對可行解增加限制。
原始 `csrc/ternary_signs.c` 仍保留，`conditioning=False` 可使用原枚舉。

計算預算現在計算實際測試的 conditioned assignments。
預算用盡且尚有相容 preimage 時為 UNKNOWN，不能排除固定 supports。
完整枚舉完成才可回報 UNSAT_FIXED_SUPPORT。

`tests/test_affine_weight.py` 以完整 binary 空間直接驗證：
symbolic RHS 相容性、每個 preimage、互斥性、任意 affine 依賴、
指定個數、整數自相關、空投影與預算狀態。

## 2. 實測：固定位置符號補全

以同樣的 1,410 個 L86 固定位置問題，先暖機，兩方法交錯執行 7 次。
每次皆完整比對 SAT／UNSAT／UNKNOWN 結果；SAT 會經獨立整數 PACF 審核。

| 指標 | 原枚舉 | 新枚舉 |
|---|---:|---:|
| 每輪 assignments | 2,956,984,320 | 1,999,230,592 |
| 使用個數 conditioning 的案例 | 0 | 581 |
| 中位數耗時 | 5.876 秒 | 4.376 秒 |

耗時降低 25.54%，此子階段吞吐量增加 34.29%。結果全部一致。
這個資料集中的所有 supports 仍是 UNSAT；不是 1,410 個完整 PQCP。
對較低殘差的 101 個問題，中位數耗時降低 29.59%。

記錄：`results/L86_weight_signs_256_20261006/report.json`、
`results/L86_weight_signs_16_20261006/report.json`。

## 3. CRT 位置枚舉的 ARM 向量實作

以四個 64-bit literal masks 為一批，用 ARM NEON 精確 bit popcount，
同時計算指定權重與位置距離。批次最小值讓不可能更新目前最小距離的批次
免於逐個輸出處理，但所有權重計數與所有最小距離 ties 都仍被完整計算。
使用整數等式，沒有近似、浮點誤差或碰撞雜湊。
非 ARM 平台保留原 scalar 分支。

32 個 L86 families、每個重複 10 次，交錯執行 7 輪。
中位數 scalar=0.649 秒、SIMD=0.262 秒，耗時降低 59.57%。
兩版的完整計數、最小距離與 literal output masks 完全一致。
小長度 oracle 特別把非平凡 CRT group 放最後，
覆蓋 SIMD 四個一批及 scalar tail，包含非零距離下界。
scalar 原始碼與比較記錄保存於 `results/L86_support_simd_20261006/`。

norm choices 使用最多 512 項的 exact LRU cache。
L43 的三個域，每個只有 128 個可能範數值，最多 384 項；
cache 不會隨 long-run reference 數量成長。

## 4. 兩側位置同時改動

新 experimental union 為

N₁(U₀)×所有指定權重的 V，加上所有指定權重的 U×N₁(V₀)。

N₁ 為原 supports 加上全部一搬移（位置距離 2）的 supports。
每個固定側仍由完整 norm family 求另一側，
先完整掃描所有切片的最小總距離，再解碼達成全域最小值的 literal ties。
這裡的「全域」只指上述 union，不指全部 PQCP supports。

允許指定 `minimum_total_distance` 以測試下一層。
下界本身是搜尋排程，不能直接用來宣稱較小層 UNSAT。
只有完整掃描、全部 ties 都保留且所有符號求解完成 UNSAT，
才會有 `layer_exhausted=True`，其證明範圍只有該距離層與指定內容。
預算、取消、UNKNOWN 或 ties 被截斷皆不構成該證明。

重播 16 個 native 低殘差 reference：

- 最小層共完成 362 個固定 supports，沒有壓縮精確解。
- E=4 reference 的固定一側最小距離原為 8；union 最小距離為 6，
  該層的 26 個 literal support pairs 均完成符號枚舉而 UNSAT。
- 距離 8 重播共測試 17,080 個 supports、24,154,861,936 個 assignments。
  15 個 reference 的該層完整耗盡；一個 reference 找到 SAT_COMPRESSED。
  整輪約 375 秒，這是擴大搜尋的成本，不是比原有小鄰域更快的證據。

記錄：`results/L86_joint_norm_20261006/`、
`results/L86_joint_norm_simd_layer8_20261006/`。

亦加入 content_changes 實驗：距離 2 可是搬移、加入兩位置或刪除兩位置。
由 parity content (e,o)，可行 support 大小 w 必須滿足

|d−e−o|≤w≤d−|e−o|，w≡d−e−o (mod 2)，
負號個數 n=[w−(d−e−o)]/2。

一側增加兩位置時，另一側指定大小減少兩位置，維持總 norm=d。
所有符號仍精確求解；不強迫特定新增位置放正號或負號。
目前只對 E=4 reference 完整測試最小層 77 個 supports，仍無解。
這不排除更遠距離或其他 contents。
記錄：`results/L86_joint_content_20261006/`。

以上較大的 union 與 content_changes 尚未加入 production 預設，
避免每輪無條件增加數十秒；目前 main 的預設新增的是個數剪枝與 ARM 枚舉。

## 5. 奇數半長度的 lift 轉換

定義 c_i=(x_i+x_(i+d))/2，h_i=(x_i−x_(i+d))/2。
則 c,h∈{−1,0,+1}，逐位置恰有一個非零，且

P_x(u)=2P_c(u)+2N_h(u)，
P_x(u+d)=2P_c(u)−2N_h(u)。

兩侧的 c 已滿足 coarse target 後，h 的必要且充分條件是

N_hA(u)+N_hB(u)=[T(u)−T(u+d)]/4。

N 是繞過 d 時乘上負號的 negacyclic 自相關。
**d 為奇數時，令 g_i=(−1)^i h_i，則 N_h(u)=(−1)^u P_g(u)。**
證明：若 j=(i+u) mod d、wrap=[i+u≥d]，
i+j≡u+wrap (mod 2)，所以原 negacyclic 的 wrap 負號與 modulation 的
wrap 負號抵消，只留下 (−1)^u。

g 在 c 的零位置上為 ±1，其他位置為 0。
若 c 有 n_c 個負號、z 個零，原 x 的 even ones 為 e，則

**g 的負號個數為 e−n_c。**

因為零位置 i 為偶數時 even bit 是原方向 bit b_i；
i 為奇數時 even bit 是 1−b_i，而 g 的 sign bit 正是 b_i XOR (i mod 2)。
odd ones 自動由 e+o=2n_c+z 保證。

所以 lift 精確等價於一個 ordinary periodic 符號問題：
support 為 c 的零位置，指定負號個數 e−n_c，
target 為 (−1)^u[T(u)−T(u+d)]/4。
可使用相同 GF(2) 高斯消去與全部自由變數整數枚舉，
不需要等待 Z3 判定。
此轉換與 `solver/odd_lift_native.py` 已實作；
`tests/test_odd_lift_native.py` 對全部 L6 coarse pairs，
以及真實 L14 coarse jobs，與原 binary 所有方向逐個枚舉比較，
包含 SAT、UNSAT、內容、完整原自相關及預算 UNKNOWN。
此 native lift 現已接入 L86 固定 coarse job 的預設完成流程：
先做 bounded exact enumeration，UNKNOWN 才回一般 Z3。
含 exclusions 或 cross constraints 時保留一般 Z3；`native_odd_lift=False` 可回退。
模 8 加強、預算與驗證範圍見 [新推導文件](L86_mod8_quadratic_pruning.md)。

本次 exact coarse job：`f0f7351bb4a166a84134`，k=1、η=−1。
原本 Z3 1 秒測試得到 LIFT_UNKNOWN；新的完整 native 枚舉
rank=22、21 個自由變數，完成 2,097,152 個 assignments，得到 LIFT_UNSAT。
暖機後兩次整個 native lift 分別約 0.024、0.026 秒。
因此這個 coarse witness 無法變成所需 parity content 的完整 L86 PQCP。
短逾時結果本身沒有被當成 UNSAT；排除依據是完整枚舉及其已驗證的轉換。

## 6. 下一個可提前剪枝的數學結論

固定 literal coarse supports 與 n_c，以及 parity content (e,o) 後，
g 的 supports、負號個數與 target 全部被決定。
**這個 g 問題不依賴 c 在非零位置上的正負號排列。**
因此 g 完整 UNSAT 可以提前排除這個支持／內容組合中的全部 coarse sign assignments。
反之，只要同一支持／內容組合的 c 問題與 g 問題各有精確解，
就能獨立取兩個 witness，以 x_i=c_i+(−1)^i g_i、
x_(i+d)=c_i−(−1)^i g_i 重建完整 binary pair，並由 verifier 最終確認。

這給出下一步比「先找 coarse、再花時間 lift」更直接的搜尋方法：
對 norm 生成的每個 supports，同時建立兩個必要線性系統，
先求解估計較省工作的一側；它完整 UNSAT 就排除整個組合，
只有 SAT 才求解另一側。UNKNOWN 一律保留未排除狀態。
這個完整支持補全介面已實作於 `solver/pqcp_support_solver.py`：
兩側先檢查線性可行性，依保守的枚舉上界選擇較省工作者先求解，
只有兩個精確 witness 才回傳經完整 verifier 的 `SAT_PQCP`。
目前 L86 的 production cyclic-norm repair 已預設使用這個完整支持補全介面。
它只在 C 與 G 都有整數見證、重建並通過完整 verifier 後才接受完整解。
此銜接本身尚未證明全流程 discovery 加速。
`tests/test_pqcp_support_solver.py` 用原始 binary 所有 L6 pairs，
及全部 2^14 個原始 L14 binary rows 建成的索引，
直接比較固定 supports 的完整 PQCP 可行性；不靠新的差值公式當 oracle。
另測試已發現 L86 coarse supports 的完整差值 UNSAT 排除。

距離 8 發現的 coarse 候選會讓舊 coarse 搜尋在該 reference 提早返回。
因此另外以這個 full-support solver 繼續測試原本尚未做符號補全的 supports，
記錄保存於 `results/L86_full_support_remaining_20261007/`。
原本尚未測試的 1,657 個支持組合已全部完成，皆為 UNSAT_BINARY_SUPPORT。
連同先前 127 個 coarse UNSAT 及 1 個完整 native lift UNSAT，
此 reference 的固定內容、距離 8 union 中全部 1,785 個 literal support pairs
已被完整排除。證明範圍與覆蓋核對保存於
`binary_layer8_certificate.json`；這不是整個 L86 問題的 UNSAT 證明。

## 7. 全域 partner 搜尋中的循環位移剪枝

對固定 U，V 的全部 cyclic rotations 都有相同 periodic PACF。
需要額外確認的是完整 binary 問題的 parity content 是否也保持。
當 d 為奇數、希望在 compression 上旋轉 t 時，
取原長度 2d 上的偶數位移 r：t 偶數則 r=t，t 奇數則 r=t+d。
原 x 的 even/odd ones 保持；c 與 g 都旋轉同樣的 t。
因此固定 U、固定 contents、固定 target 的 full-support 可行性在 V 旋轉下不變。

在非零 norm field component，旋轉 V 等於乘上 α（α 的階數為 d）。
norm-one group 的階數為 q+1，故 α=h^((q+1)/d·r)，r 與 d 互質。
只需讓第一個非零 field component 的 phase 留在
0..(q+1)/d−1，每個旋轉 orbit 恰有一個 CRT 代表。
若全部 field components 都是零，V 為常數且 orbit 大小是 1，原樣保留。

L43 的最大 family 從 129³=2,146,689 降為 3×129²=49,923，
這是 exact symmetry quotient，不是強迫解具有循環對稱。
`tests/test_cyclic_support.py` 對 d=3、5、11 的每個 U、每個 k，
將代表的全部 rotations 展開，與原完整 family 比較，
並確認不同代表的 orbits 互斥。
亦用全部原始 L6 binary words 驗證偶數原位移確實同時旋轉 c 與 g。

`solver/cyclic_orbit_search.py` 以此代表直接求完整 binary 支持補全，
兩方向皆保留；預算不足或截斷時只回 UNKNOWN，
全部代表完成 UNSAT 才回報 UNSAT_FIXED_SIDE_UNION。
它不保持「與未旋轉 reference 的最小距離」，因此沒有拿它替換原距離層證明。
這也是獨立實驗介面，目前未增加 main 的預設長計算。
重播記錄保存於 `results/L86_cyclic_orbits_20261007/`。

仍需找到經完整 verifier 且等價類去重後的全新 L86 解，
再與相同硬體、搜尋預算及 seed 分布的 baseline 比較 discovery rate。
現在 baseline 與新方法都沒有完整 L86 解，故沒有可驗證的 20% 新解加速比例。
