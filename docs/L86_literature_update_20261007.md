# 2026-10-07 文獻核對與下一步

## 新讀取的 2026 PQCP 原始論文

Huang、Hsu、Liu、Chen，
[A Novel Design of Sparse Vector Coding Aided URLLC With Periodic Quasi-Complementary Pairs](https://repository.essex.ac.uk/42886/1/Final%20Version.pdf)，
2026 ICC，作者機構公開全文，6 頁。
Definition 7 只要求每個非零 PACF sum 的絕對值不超過 ε；
它的 Theorem 1 對 binary Golay pair 增加一個符號，得到奇數長度、ε=2 的 pair。
本專案 Project 2 第三題另要求恰有兩個 ±4 sidelobes。
因此論文中的一般 PQCP、奇數長度構造，以及通訊 BLER 效能皆不能直接充作 L86 解或搜尋加速證據。

值得後續自行推導的路徑是：從經独立 aperiodic verifier 核對的 Golay prefix，
加入少量未知尾端符號，在每個位移精確展開 PACF，而不是只檢查其最大值。
例如長度 80 prefix 加 6 個位置，只有 12 個新增 binary signs；
prefix-prefix 的兩列 aperiodic correlations 已相消，剩下 boundary terms 可建立精確約束。
在 independent shifts 6..43，新增符號不會互相相乘，形成 rational/integer linear 必要方程；
較小 shifts 才有 tail-tail quadratic terms。
需先證明／檢查剩餘線性系統是否允許 ±1 tail，再完成整數 PACF。
此額外推導已做成隔離實驗 `experiments/search_golay_boundary.py`。
以文中長度 10 GCP 為起點，每次包含全部兩列換號／反轉／交换，
再做 concatenation 或 interleaving doubling，得到長度 20、40、80 的 8、56、384 個代表。
各代表均獨立檢查全部 aperiodic correlation sums 為零。
固定前綴後，12 個 tail bits 及全部 k=1..42、η=±1 共同進入精確 Z3 約束。
384 個前綴全部完整 UNSAT，沒有 UNKNOWN；不是只測試少量隨機 tails。
這只排除該明確生成的前綴族，不能排除其他 Golay 前綴或一般 L86 PQCP。
未加入預設，也沒有宣稱此路線產生 L86 解。

`tests/test_golay_boundary.py` 用小前綴的全部 tails 比較原長度整數 PACF；
L86 則比較 0、全部 unit bits、全部 two-bit patterns 共 79 個配置。
因 expansion 與 PACF 都是最多二次的 multilinear functions，這些配置精確決定全部係數。
記錄 `results/L86_golay_boundary_family_audited_20261007/report.json`。

## 其他已核對來源

[Lumsden、Kotsireas、Bright](https://arxiv.org/html/2408.15611v3) 的 compression 與 orderly generation
可以沿用搜尋架構；原 periodic Golay 零目標不可直接用於本專案的非零目標。
43 為質數，86 也沒有多個小因數可獲得該文的 multilevel compression 優勢。

[Andraschko、Kotsireas、Kreuzer 的 2026 ICMS 摘要](https://icms-conference.org/2026/abstracts/Andraschko.pdf)
說明壓縮三值序列、Boolean 電路與精確提升。
[作者研究室](https://www.symbcomp.fim.uni-passau.de/en/veroeffentlichungen/scientific-papers)
列出 9 頁 extended abstract，但該清單沒有公開全文連結；本次僅能核對一頁摘要。

[Joux、Mora 2025](https://link.springer.com/article/10.1007/s10623-025-01717-6)
提供有限域二次系統的研究背景。
本專案的 PACF 模 8、尾端 quotient projection、exact cardinality 與最小 Hamming 半週期構造
都有另外寫出的推導，沒有把該文的 regular-system 複雜度直接當成加速保證。

已實作的推導分別見 [模 8 剪枝](L86_mod8_quadratic_pruning.md)、
[半週期候選投影](L86_half_shift_observation.md)、
[受限 automorphism 構造](L86_automorphism_construction.md)。

## 後續新增核對的搜尋文獻

[Pšeničnik 等，2026 預印本](https://arxiv.org/pdf/2607.09688)
以 Thompson sampling 分配區域搜尋資源，使用 GPU self-avoiding walks。
研究目標是 aperiodic LABS，不是本專案的週期兩列精確稀疏目標。
本文的連續 reward fractional Beta 更新不是對 PQCP 解的必要條件；
Bloom filter 可能有假陽性，不能拿來做不遺漏解的排除。
可借用的是資源排程，必須讓每個未排除區域仍有探索機會；
目前沒有把它當成數學剪枝或 L86 加速定理。

[Zhang 等，2025 v2](https://arxiv.org/pdf/2504.00987)
研究 massively parallel memetic tabu search 的 LABS 搜尋。
其 unrestricted search 與大鄰域評估提供探索更大區塊的動機，
但 Nvidia A100 的效率、LABS merit factor 與成功率不能轉為 Apple GPU 的 PQCP 指標。
本專案另外推導 [精確支撐區塊](L86_exact_support_blocks.md)，
保留所有必要簽章匹配，所有 C/G signs 重新精確求解。

[Lebedev，2026-09 arXiv 預印本](https://arxiv.org/pdf/2609.04589)
以 Gray map、反射限制、整數相關向量 MITM 和分層壓縮搜尋 quaternary Legendre pairs。
四元字母及常數 -2 目標不能直接給出本專案二元 L86 解。
尤其其四列表中，相關向量合併可以利用反射限制消除交叉項；
本專案不同的 odd interleave cross equations 仍取決於實際序列，
不能只保留一個 homometric 代表。
其受限 queue 的 exhaustion 也不等於一般不存在。

[Ramos、Hulak、de Queiroz，2026-07 預印本](https://arxiv.org/pdf/2607.20765)
研究長度 333 Legendre pairs 的固定共同 multiplier subgroup，
強調壓縮 entries 必須保留 orbit sizes 所允許的精確 value sets。
333 的 mod-3/mod-37 限制及 subgroup 排除不適用於質數半長 43。
可沿用的原則是：證明排除前要確認整數可行集完整，不能把 relaxed witnesses 當成原解。
本專案據此另行檢查 odd-kernel basis 是否 saturated，
並保存可用純整數乘法核對的右逆；推導見 [整數核](L86_odd_kernel_lattice.md)。

這些文件均區分「作者原始結果」與「本專案自行推導」。
構造研究入口維持在 experiments；一般 L86 搜尋的預設限制没有增加相同 O 或反射假設。
下節的交替換號商則已接上獨立可回退的主程式旗標，預設仍關閉。

## 交替換號從解後去重移到搜尋前

重新核對 [Lumsden、Kotsireas、Bright 原始論文 §2.1](https://cs.uwaterloo.ca/~cbright/reports/periodic-golay-pairs.pdf)
的 alternating negation；該文為零目標，符號保持問題可直接省略。
本專案另推導 T'(u)=(-1)^u T(u)，所以 odd-k 的 η 必須翻號，並同步變更 parity counts。
已接入 [交替換號目標商](L86_alternation_orbit_pruning.md)，保留全部新解等價類的代表。
40/53 是保留參數族比例，不是新解加速率；必須由完整配對長測驗證。

## 短邊界修復的另一條精確推導

上述原始文獻列出的 periodic82 原料，可核對於
[Đoković、Kotsireas §3.2](https://arxiv.org/pdf/1310.5773)。
補四個符號後改成 86 點時，前綴 PACF 不能假設仍相消。
本專案直接保留全長 baseline，將 u≥4 的尾端條件轉為模 8 線性必要系統，
保留完整 nullspaces 與所有目標位移／符號，再做全長整數核對。
同一核心也可作為任意非解中心的短 cyclic-window 修復，允許跨越原來固定 parity counts 的限制。
推導、測試及實際涵蓋的實驗範圍見 [邊界線性投影](L86_boundary_mod8_projection.md)。

## 2026 年高效能 Legendre 搜尋的新核對

Sojoodi、Kotsireas、Afsahi，
[Efficient High-Performance Computing Strategies for the Legendre Pairs Search](https://link.springer.com/article/10.1007/s10766-026-00820-y)，
International Journal of Parallel Programming 54, article 13，2026-05-16。
出版社摘要說明以 OpenMP／MPI／CUDA 平行化及優化，提供 H100／A30 相對 serial CPU 的結果。
本次取得的是出版社摘要與書目，沒有付費全文；摘要連出的作者程式庫回傳 404。
因此沒有聲稱已閱讀其完整演算法，更沒有將其倍數套用到 Apple GPU 或嚴格 L86 PQCP。

本專案另行推導 [任意位置翻轉的二次模 8 配對](L86_unrestricted_flip_join.md)：
從原始 binary PACF 精確展開，保留所有同簽章 preimages，允許 parity contents 改變。
此推導不是該文的定理；其必要條件及完整小空間核對都有本專案實作。
新實驗入口可以直接讀取目前的長測 log，不用既有 PQCP 當種子。

另完成 [相同 perfect O 的正號奇數仿射格](L86_perfect_affine_kernel.md)，
全部整數格由 SR=I 證書與 primitive kernel 保證，精確球界涵蓋所有可行 trit。
168 個 fibers、1,344 個固定支撐問題均已完整判定無解，無 UNKNOWN。
只排除其特定構造族，沒有把構造限制放進一般主搜尋。

## D-optimal 原料與更一般的奇數列

重新核對 [公開 order-86 原料](https://documents.uow.edu.au/~jennie/D-optimal/order.86.html)，
並參考 [Đoković、Kotsireas 的 D-optimal 原始論文](https://arxiv.org/abs/1408.6116)。
後者構造其他 orders 並研究 compression，沒有直接提供嚴格 PQCP86 搜尋加速定理。
本專案另外推導 [固定支撐的完整 SDS norm 補全](L86_sds_norm_expansion.md)，
不再只交叉匹配已公布的有限 rows。
從已獨立驗證的原料交換 U 支撐，V 在全部 CRT norm preimages 中重新完整求解，
再以整數 intersection/PACF 決定原料及原始 86 點解。

進一步自行推導：D-optimal E 與任意 Legendre O 的 summed PACF 可精確抵消偶數旁瓣；
不需強迫 O 每列 individually perfect。
已實際產生非 perfect Legendre 原料，並在完整對稱閉包及所有 relative phases 下搜尋。
仍未找到完整 L86 新解，沒有把更多原料或局部時間減少當成 20% 新解發現證據。

## 四列補全、skew 結構與範數／trace 的來源界線

重新閱讀 [Lumsden、Kotsireas、Bright 2025 修訂版 §2、§7](https://arxiv.org/html/2408.15611v3)：
該文使用 compression、orderly generation、PSD 篩選並研究零旁瓣 periodic Golay。
其 DC 平方和及 prime-half compression 定理的目標是零旁瓣；
PQCP86 的非零目標必須重新代入，不能直接使用 PSD≤172 的零目標上界。
本專案因此自行推導 DC=172+8η、ALT=172+8η(-1)^κ，
取得全部可行 parity-row sum 形狀，記於 [任意半列補全](L86_half_row_completion.md)。
也把「未知 intersection profile 的 Fourier 值非負」轉成
[完全有理數的負上界證書](L86_certified_profile_psd.md)，沒有用浮點誤差作無解判定。

[Đoković、Kotsireas 2018](https://arxiv.org/abs/1802.00556)
分類的是每個 block symmetric 或 skew 的特定 GS families。
此文提供43點原料與 skew 結構的動機，不能据此排除 arbitrary binary PCS4 或 PQCP86。
實作已放寬成固定任意三列、完整補全第四列，並完成兩波原料圖擴展；
不再要求 E／O 各自的 summed PACF 為常數。

[Đoković 2024 §1–2](https://arxiv.org/html/2404.14375v3)
的 spin／slide families 要求三個 blocks 有相同重量及 multiplier 關係。
本專案奇數目標的 canonical unsigned PCS4 重量為16、18、21、21，
沒有三個相同重量，因此這個 spin template 不能直接覆蓋所需原料。
論文列出的43點特殊參數搜索失敗，也不是本專案 PQCP86 不存在證明。

[Đoković 2025 Theorem 2](https://arxiv.org/html/2508.17141v1)
構造 q≡3 mod8 的 symmetric Hadamard order q(q+1)；q=43 時為1892。
這不是86點 PQCP的構造定理。其 skew／repeated-block 設計只作結構探索的參考。
本專案另行推導 [impulse g／任意 skew O](L86_impulse_skew_O.md)，
完整排除該有限族的122,880個 c，但未施加任何 general-search skew 限制。

[2026 Legendre HPC 論文](https://link.springer.com/article/10.1007/s10766-026-00820-y)
仍僅取得出版社摘要及書目，其 CUDA benchmarks 不能換算成本機新解速度。
本次確認公開 algorithm-image links 未成功取得，沒有假稱讀過付費演算法全文。

最後，[norm／trace 剪枝](L86_norm_trace_pruning.md) 是本專案自行推導：
每個 F_(2¹⁴) 分量同時求 u·u^128=n、w·u^128+w^128·u=h。
一般 nonzero partner 將每域129個norm phases減到最多兩根；
零分量保留完整fallback。這不是上述任何論文已證明的 PQCP86 加速率。
Python 舊配對稍慢，所以進一步改成純整數 C 算術及共用中心準備，再做相同完整子空間配對。
完整新等價類均為零時仍不计算「找到新解」的百分比。

## 2026 年七至九月的新文獻核對

[Ramos、Hulak、de Queiroz，2026-07-22](https://arxiv.org/html/2607.20765v1)
研究333點 Legendre pair 的固定共同 multiplier 子群，用精確 compression、
row-sum congruences 及有證書的有限枚舉排除特定子群。
它不直接決定86點PQCP；43為素數，本專案沒有其3／37 compression chain。
可借鑑的是要保留受限模型的完整值域、base／kernel／輸出證書，以及排除範圍。
新的 [四點g搜尋](L86_same_O_weight4.md) 因此保存原始mod4方程、
完整affine basis及每個fiber的實際計數，不把部分抽樣當成全族排除。

[Lebedev，2026-09-04，§3–4](https://arxiv.org/html/2609.04589v1)
以Gray map將quaternary Legendre轉成四個binary rows，
使用midpoint整數內容條件、reflection限制、PAF向量配對與divisor-chain compression。
其quaternary目標、旁瓣總和及字母表不同；本專案不能直接把其pairs当作PQCP86。
對它的同PAF去重也須小心：其模型中的symmetry消去imaginary cross條件，
本專案仍有odd cross方程，所以同PAF的literal preimages不能只留一個。
因此本專案profile cache繼續保留全部literal固定三側及相對phases。

[Bulutoglu、Baczkowski、Yauney，2026-04-24](https://arxiv.org/html/2604.22423v1)
研究Legendre等價操作的群，§1允許共同decimation、獨立平移／反射及交換。
本專案新g bank只用「同時反射c,g,O且保持target」的完整映射取商；
固定target、固定部分literal rows或bounded repair balls時，仍需另外核對商的覆蓋。

[2026 PQCP通信應用全文 Definition 7／Theorem 1](https://repository.essex.ac.uk/42886/1/Final%20Version.pdf)
採取所有非零PACF sums的幅度≤ε，並由Golay列追加一點得到odd-length、ε=2的pairs。
本專案要求恰兩個非零±4及其餘全部零，定義更嚴格；不能以名稱相同直接移植該構造。
這些較新方法的適用性已逐項區分，沒有據此聲稱86新解存在或加速20%。

## 2025 inverse-Gram講稿與本專案的新推導

另取得Goldberger的 [Hadamard 2025原始講稿](https://us.ticmeet.com/assets/archivos/d6f1d9b8-d39f-4888-925a-7eb81c4905cc/Goldberger.pdf)，
它討論integer matrix star-algebras、cyclotomic field descent及lattice／determinant methods。
這是會議講稿，不當成已取得完整論文。其generic Gram uniqueness及分鐘級例子，
不能直接套到本專案的兩個circulants／binary係數與非generic設計。
例如full PQCP的方形two-circulant block Gram為diag(G,G)，determinant自動是det(G)²，
單純「Gram determinant為平方」不產生新L86剪枝。
PDF圖像取得失敗；此處僅依可讀的文字／公式與本專案自己的推導，不解讀未取得的圖。

本專案另推導 [聯合real Gram不等式](L86_joint_gram_pruning.md)、
[p²的single norm條件](L86_prime_norm_obstruction.md)，及
[F_(257²)的共軛norm／trace二次根條件](L86_finite_field_trace_pruning.md)。
這些不是該講稿已證明的PQCP86演算法或加速率。
每個剔除使用literal integer／finite-field證書，zero branches有完整回退；
特別加入「兩個unknown norms相加不可使用single-norm障礙」的counterexample測試。

## 2026-10-08追加核對

[Verma，2026-09-22，A Quaternary Legendre Pair of Length 64](https://arxiv.org/html/2609.25573v1)
提供一組64點四元witness及Gaussian-integer逐lag核對。
它不是新的通用搜尋演算法；其非零相關總和為−2，與本專案恰兩個±4的binary目標不同。
不能從已知64點例子直接聲稱86點構造。其完整witness核對方式與本專案接受者的整數verifier相容。

[Pšeničnik等，Prioritizing Search Space Regions in the LABS Problem](https://arxiv.org/html/2607.09688v1)
頁首v1為2026-06-17，arXiv ID為2607.09688；保留來源日期差異，不將它說成10月新稿。
該文用Thompson sampling、GPU self-avoiding walks及Bloom filter安排非週期單列merit-factor搜尋。
這不證明本專案PQCP目標或未採樣区域不可能。

## 2026-10-08：壓縮列範數與固定負號圖割的自行推導

再次核對 [Kotsireas、Koutschan、Winterhof，Quaternary Legendre pairs II](https://arxiv.org/html/2408.16318v2)
的 §2.2／§4：其 cyclotomic 條件不能把「兩個未知範數的和」當成單個範數。
本專案因此從 Z_86≅Z_2×Z_43 的原始 PACF 等價式出發，
只對固定 c_A 所决定的單列 c_B 範數使用 mod4／mod8 提升；
g_A／g_B 總範數保留獨立耦合求解。
冪等元提升、全部有限域根、線性像／kernel 枚舉、單列空集合快取，
均是本專案推導與實作，不能歸為該論文已完成的 PQCP86 結果。

另外找到 [Perera、Kotsireas，2025，A low-complexity algorithm to search for Legendre pairs](https://www.sciencedirect.com/science/article/pii/S0024379525000102)。
可取得的作者摘要／出版頁說明以矩陣結構、Gershgorin 及 FFT-like DFT 降低 PSD 成本。
全文未取得，沒有聲稱已讀其完整演算法，也沒有將其 45／63 點時間移植成 86 點效能結論。
本專案已有批次 PSD 與 outward interval 證書；浮點 FFT 可協助候選排序，
但任何精確排除仍需可審核的誤差界或整數證書。

[Willms，2012，Autocorrelations of Binary Sequences and Run Structure](https://arxiv.org/abs/1206.4572)
研究週期／非週期自相關與 run structure 的關係。
在此觀點之外，本專案自行把「有零元素的固定支撐＋固定負號數」
化成各位移的路徑／循環圖，完整 DP 求所有可達 cut 數，
再合法延伸到兩個未知 signed norm 的集合和。
此延伸不假定相鄰 runs 已決定全部位移，也不強加反射對稱。

完整數學、程式、作用範圍與實驗證據見
[壓縮列與 2-adic 補全](L86_coarse_row_and_two_adic_completion.md)。
新 cut 法的 36 個實際固定支撐分支沒有額外排除、時間增加约 20.35%，
因此尚未預設啟用；完整區域與 affine 先行排程結果另行保存，
不能以這個小區域否定整個數學方法，也不能把它說成新解加速。

後續已有完整 c_A 區域三次配對：圖割先於 affine 的凍結版本耗時約為 5.124 倍。
19 個 cut 排除全部也被原 affine 系統排除，沒有額外收益；已保存完整來源與所有分支，
並重新以 affine 先行排程做完整配對，並非停止或假稱該方法有效。

對 2026 反射限制又自行推導出 [PQCP86 的反射範數障礙](L86_reflection_norm_obstruction.md)。
利用 signed reflection 列的 norm mod4 僅依賴支撐，
完整枚舉 43 方程／20 變數線性系統，兩個目標正負號的必要重量 slice 都為空。
因此不能讓兩條 C 或兩條 G signed rows 同時反射對稱；
這是本專案特定目標的推導，與原論文不同的 quaternary／Gray row 限制需分別核對。
不把其他問題的反射固定族，和一般搜尋中的合法反射等價取商混為一談。
Bloom filter false positives不能直接用作exact completion的UNSAT／去重證據，
skew-symmetric限制也不能直接從LABS換算成PQCP完整商。
目前沒有導入這些啟發式刪除；本專案的新區域選擇和完整數學判定仍分開記錄。

重讀 [Quaternary Legendre pairs II，§2.2／§4](https://arxiv.org/html/2408.16318v2)
核對其cyclotomic norm的rational prime-order限制。
其中整數norm／PSD product的條件不能直接拿來判定兩個unknown norms的和。
本專案目前只在single unknown norm及anchored cross-trace的合法場景使用自己的有限域推導。

另由本專案目標式推導並實作 [稀疏affine計算合併](L86_affine_profile_batches.md)，
以及 [三半列完整有限區域](L86_three_half_region.md)。
這些新增方法的數學適用範圍、literal解集合測試與新解計數各自記錄，沒有把文獻的其他目標加速率移植到PQCP86。
