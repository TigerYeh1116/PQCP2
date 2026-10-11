# 長度 86 的有限域範數位置搜尋

日期：2026-10-06。這是依照本專案非零旁瓣目標自行推導的搜尋方法，
不是宣稱某篇論文已經解決長度 86 的 PQCP。

2026-10-07 新增 `cyclic_norm_family(desired,d)`，將同一套 CRT／norm 方法擴充到
任意 symmetric F₂ target 的 UU*=desired。其 `norm_target` 明確區分原固定側 PQCP 問題，
不應交給原 `nearest_supports` 當成固定 U／k 查詢。
此扩充用於獨立的 [automorphism 構造](L86_automorphism_construction.md)，
沒有替換原 nearest-reference 搜尋。general targets 以 d=3、5、11 的全部 masks 驗證。

## 文獻與適用範圍

- Lumsden、Kotsireas、Bright，
  [New Results on Periodic Golay Pairs](https://arxiv.org/html/2408.15611v3)，
  2025 年修訂，DOI [10.1090/mcom/4096](https://doi.org/10.1090/mcom/4096)。
  文中使用壓縮、多層壓縮、orderly generation 及精確匹配。
  本專案可以沿用「必要條件候選生成 → 精確補全」的架構；
  86=2×43，43 為質數，無法沿用多個小因數的多層壓縮優勢。
  文中的零旁瓣 periodic Golay 結構與猜想，不是本問題的排除定理。
- Andraschko、Kotsireas、Kreuzer，
  [Computing Periodic Golay Pairs Using SAT Solvers](https://icms-conference.org/2026/abstracts/Andraschko.pdf)，
  2026-03-04 的會議摘要。支持壓縮後 Boolean 編碼、Hamming weight 電路與 lift 的研究方向。
  本次能讀取的是摘要，不能據此推斷 L86 的效能。
- Jared Gobin，
  [Boolean Complementary Pairs](https://mspace.lib.umanitoba.ca/bitstreams/e3f73833-0322-4661-b9ef-f91d46391501/download)，
  University of Manitoba，2023 碩士論文。
  提供 ternary、Boolean 與 Laurent polynomial 表示的背景。
  本次下述「週期環、非零目標、43 次分圓因子範數」是額外推導；
  不把原文零自相關的因子分解當成可直接套用的定理。

## 1. 從原問題到位置多項式

用符號序列 x,y∈{−1,+1}^86 表示原始 binary pair。
令 P_x(r)=Σ_i x_i x_(i+r mod 86)。目標為

T(0)=172，T(k)=T(86−k)=4η，其餘 T(r)=0，η∈{−1,+1}。

以 d=43 壓縮，a_i=(x_i+x_(i+43))/2，b 同理，故 a,b∈{−1,0,+1}^43。
完整解必須滿足

C(r)=P_a(r)+P_b(r)=[T(r)+T(r+43)]/4。

因此 C(0)=43，C(k mod 43)=C(−k mod 43)=η，其餘為 0。
以下將 k 與 −k 合併成獨立位移 1≤k≤21。

令 U_i=[a_i≠0]、V_i=[b_i≠0]。
因為 a_i≡U_i (mod 2)，所有符號補全都必須滿足

P_U(r)+P_V(r)≡C(r) (mod 2)。

在 R=F₂[X]/(X^43+1) 中，令 U(X)=Σ_i U_i X^i，
並以星號表示 X→X^−1 的環自同構。上述完整 43 個奇偶條件等價於

**UU*+VV*=1+X^k+X^(43−k)。**

η 的正負在 mod 2 下相同；之後整數符號方程仍會使用真正的 η。

## 2. 分解成三個範數方程

2 在 (Z/43Z)× 的乘法階數為 14，且 2^7≡−1 (mod 43)。
因此 X^43+1=(X+1)f₁f₂f₃，其中三個 f_j 均為次數 14 的自反不可約因子。
程式以整數 bit polynomial 表示並驗證它們為

f₁=0x4ff9，f₂=0x5495，f₃=0x688b。

實作會驗證不可約性、因子乘積、CRT 冪等元及其所有剩餘值，
不是只信任以上三個常數。
X^43+1 與導數 X^42 互質，故分解沒有重複因子。
中國剩餘定理給出

R≅F₂⊕F_(2^14)⊕F_(2^14)⊕F_(2^14)。

若 α 是任一非平凡因子的根，α^−1=α^(2^7)。
因此星號在每個域上就是 v→v^128。
固定 U，在第 j 個域中定義

D_j=(1+X^k+X^(43−k)) mod f_j + u_j u_j^128。

D_j 在 F_(2^7) 中；要解的條件為

**v_j^129=D_j。**

- 若 D_j=0，域無零因子，因此唯一可能是 v_j=0。
- 若 D_j≠0，取 v₀=D_j^64。由 D_j^128=D_j，
  v₀^129=v₀²=D_j，故 v₀ 是一個根。
  範數映射 F_(2^14)×→F_(2^7)× 的核有
  (2^14−1)/(2^7−1)=129 個元素。
  選一個經完整階數驗證的核生成元 h，所有根恰為
  v₀h^t，0≤t<129。

在 X+1 因子上，U(1)+V(1)=1，所以 V(1)=1+U(1)。
這也正是兩側非零位置總數 43 的必要奇偶條件。

## 3. 完備性與減少枚舉的範圍

固定 U，每個域上的所有根均已列出，CRT 亦是雙射。
因此各域選項的 Cartesian product 完整、無重複地生成
**所有符合 UU*+VV*=目標的 V**。
最多有 129³=2,146,689 個，相較於未限制權重時的 2^43 個任意 V。

這是固定 U 的奇偶必要條件的完備參數化。
不是完整 signed ternary 或 binary PQCP 解的參數化，也不是全域搜尋完備性證明。
固定權重的原空間應比較 binomial(43,w)，而不是把 2^43 當成其大小。
不能把集合大小比例解讀成實際時間或找到新解的速度比例。

## 4. 接入現有流程

GPU 三值搜尋 → native 整數延續 → 原本局部位置修復 → 新範數分支 →
原本其他修復 → 現有 binary lift → 獨立完整 verifier。

新分支對一個 reference 做兩次生成：固定 A 生成 B，再固定 B 生成 A。
C 完整掃描各自的 CRT family，精確篩選另一側的指定非零個數，
並找出與 reference 距離最小的所有 literal masks。
兩個方向交替送入既有的 GF(2) 高斯消去與整數 Gray-code 符號枚舉。
每次最多送入 8 組；候選選擇與距離排序是搜尋排程，不是數學排除條件。

- `weight_matches` 是整個 family 中通過指定權重的數量。
- `nearest_ties` 是整個 family 中最小距離的同距離解數量。
- `complete_nearest` 只表示輸出容量足以保存全部同距離 masks。
- 每個符號枚舉的 `UNSAT_FIXED_SUPPORT` 只涵蓋該 literal mask pair 及指定符號個數。
- 新分支無解時只回報 `BOUNDED_MISS`；任一符號枚舉未完成則回報 `UNKNOWN`。
  不排除更遠位置、同時改動兩側、其他內容或其他 target。
- 全部輸出 masks 通過獨立整數 PACF 奇偶檢查。
  符號精確解通過完整整數 PACF 與內容檢查後才成為 `CompressedLiftJob`。
  壓縮精確解仍然必須完成 binary lift，不能直接寫成正式 PQCP。
- 現有符號補全一次只回傳一個 coarse witness。
  後續已推導出：對本專案的奇數半長度、固定 literal supports、固定符號個數與
  parity content，完整 lift 的可行性只依賴零位置與這些個數，
  不依賴 coarse 非零位置上的正負號排列。
  因此完整 lift UNSAT 可以排除同 supports、同個數的其他 coarse witnesses；
  逾時 UNKNOWN、不同 supports、不同符號個數或額外 cross-compression 限制，不能這樣排除。
  證明與原始 binary 全枚舉測試見
  [後續剪枝與 lift 推導](L86_weight_conditioning_and_joint_search.md)。

在 L86 的直接三值壓縮分支中預設每輪選一個 reference，最多 8 組符號補全。
從 IDE 執行 main.py，L=86、保留 Z3，並在「直接三值壓縮搜尋」選 y，
此分支會一起啟用，不需要另輸入命令。
啟動時會顯示「有限域範數位置搜尋」。
設定 `direct_compression_cyclic_top=0` 可回復原有分支；
整體直接三值壓縮的既有啟用方式沒有改變。

結果記錄於各 session 的 `L86_seed*_cyclic*.json` 及 cycle snapshots。
metrics 增加 cyclic_attempts、cyclic_patterns、cyclic_assignments、
cyclic_weight_matches、cyclic_new_jobs。

## 5. 驗證與目前證據

`tests/test_cyclic_support.py` 以直接 PACF oracle 對 d=3、5、11 的每個 U、
每個獨立 k，比較全部可能 V 與有限域生成的完整集合，並檢查無重複。
另外比較 native 的指定權重、最小距離、全部同距離個數與輸出容量語義，
以及 d=43 的跨域取樣與完整 family 的距離零案例。

`tests/test_cyclic_support_search.py` 比較小長度完整 signed-word oracle，
測試真實 L22 非零殘差候選變成精確壓縮 job、內容、PACF、
miss 不覆寫 reference、取消與 UNKNOWN。

此次回歸測試 1,186 通過、14 跳過；依既有環境略過兩個 benchmark 測試模組。
本機真實 MPS 亦完成一輪 L86 GPU→native→位置修復→其他修復流程：
新分支測試 8 組 supports、16,777,216 組符號自由配置，耗時約 0.041 秒。
記錄於 `results/L86_cyclic_norm_mps_smoke_20261006/report.json`。
這驗證交接與實際裝置相容性，不是長測的成功率或加速證據。

離線重播 16 個 native 延續後的低殘差 L86 reference：

- 舊方法：3 組 matched supports，6,291,456 組符號自由變數配置。
- 新方法：101 組 tested supports，211,812,352 組配置。
- 32 個固定一側的範數 family，最小位置距離分布：
  距離 4 有 3 次，距離 6 有 15 次，距離 8 有 14 次。
- E=4 的 reference，在固定 A 或固定 B 時，另一側的最小距離皆為 8。
  這只證明固定一側、保留內容大小的切片需要距離 8，
  不代表同時改動兩側也需要距離 8。
- 兩方法均沒有找到精確壓縮解，更沒有新的完整 L86 解。
  單次執行舊方法約 0.053 秒、新方法約 0.533 秒；
  新分支每個 reference 約 33 毫秒，增加搜尋廣度也增加計算成本。
  冷啟動編譯與硬體負載會影響時間，不能據此宣稱 discovery 加速。

記錄：`results/L86_cyclic_norm_20261006/report.json`。
較大的 256-reference 重播另存 `results/L86_cyclic_norm_replay256_20261006/report.json`。
256 個 reference 中，舊方法測試 42 組支持配置，新方法測試 1,410 組；
新方法完整符號枚舉了 2,956,984,320 組自由變數配置。
512 個固定一側切片中有 1 個沒有任何符合指定權重的 V，
所以該固定側、固定權重切片可被嚴格排除，但不能排除整個 reference 的其他改動。
其餘最小距離分布為 4：18 次、6：375 次、8：112 次、10：5 次、12：1 次。
此輪仍然沒有精確壓縮解，新方法耗時約 8.20 秒，舊方法約 0.34 秒。
目前不能宣稱任何「找到新 L86 解」的速度改善百分比。

## 6. 後續可推導的方向

固定一側的切片仍有限制。下一個自然擴展是對 U 的一搬移鄰域，
逐個完整生成另一側 V，再以兩側總位置距離排程。
這會涵蓋原局部方法，也涵蓋一側小改動、另一側大改動的配置；
但每個 reference 可能需數百次 family 生成，應先量測再擴大預設工作量。
另一個方向是保留較遠距離層及同一 support 的多個 signed witnesses，
讓 binary lift 的失敗能回饋補全排程，而不是重複只取第一個 witness。
這些是後續方向，尚未實作或宣稱有提升。
