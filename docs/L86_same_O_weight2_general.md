# 所有兩點 g、任意 O 的完整正號奇數構造

先前 [impulse／skew O](L86_impulse_skew_O.md) 只處理 g 的兩點與target的centered位置一致。
這裡允許它們不同：仍固定target ±1，g=δ_s+δ_-s，s=1..21，O 任意binary。
E_A=c+g、E_B=c-g、實際odd列為O及-O，Σc=9、ΣO=1；c支撐41、負號重量16。
共同平移可把任意兩個正位置的中點移至0，common unit可把非zero target移至±1。
在target固定後仍保留全部21種s，不把g與target再次一起化為1。
整個族也只包括同O、g重量二、正號odd旁瓣的構造，沒有全域排除之意。

## 唯一的 O 對稱部分

令 M=(O+O*)/2，是對稱trit列，M₀=±1。
完整odd方程等價於

\[
(\delta_s+\delta_{-s})*M=\delta_1+\delta_{-1}.
\]

令 z_k=M_(2sk)，43個點是同一循環。方程在j=2sk+s給出
z_(k+1)=h_k-z_k，其中 h_k=1 當j=±1，否則0。
循環長度43為奇數，因此
2z₀=Σ_{k=0}^{42}(-1)^(42-k)h_k，唯一決定z₀與全部M。
程式以整數核對完整原始convolution、所有M∈{-1,0,1}、M₀=±1及ΣM=1。

M_i=±1時，O_i=O_-i必須等於M_i；M_i=0時，該對的O必須相反。
只有s=1得到M=δ₀，亦即sum-one skew O。
其餘s給出不同且不必skew的完整binary O族；
ΣM_i²涵蓋1,3,...,41，每個family的自由antisymmetric pairs皆仍保留。
測試逐一重建binary O並核對原始86點的全部odd位移，不使用浮點。

## c 與 O 的完整核對

沿用 [四點g](L86_same_O_weight4.md) 的全部mod4 affine C，改成exact weight16。
每個C精確推得N_O(u)=[33-N_EA(u)-N_EB(u)]/2，
用有理PSD、anchored norm／trace 的全部0／2域根，
最後核對所有odd整數式及全部even intersections。
Native與原始skew族在s=1的計數完全一致：122,880個C，
120,882個PSD排除、1,492個norm空族、4,048個O進入完整整數核對。
可獨立對照舊skew實驗，不需要信任新增general-O程式的結論。

實作由 `complete_weight2_c_fiber` 共用已核對的native核心，
原本 `complete_weight4_c_fiber` 的驗證及C重量慣例保留。
入口 `experiments/search_same_o_weight2.py` 可在IDE直接按執行鍵，預設全21種s。
源碼、輸入、每個完整affine證書及所有計數另存獨立results目錄。
實際全族結果見 `results/L86_same_O_weight2_complete_20261008/report.json`。

該全21族已完成，22,020,096個affine assignments、2,580,480個c重量符合者；
2,538,522個PSD排除、36,340個norm／trace空族，44,944個literal O完整核對。
8,068個符合全部odd integer方程者都不符合even intersections，完整新解零；
沒有UNKNOWN，elapsed=0.515736秒。這排除整個declared weight-two g構造族，
但不排除不同O／更大g重量或一般PQCP86。

已從閉合ledger完整重播11個非空odd fibers，全部原始計數相同，
取出8,068個literal odd survivors，按一般PQCP等價得到4,034個非解中心。
所有original odd lags及native even deviation都再次獨立核對，最低精確能量96。
記錄 `L86_weight2_odd_seeds_complete_20261008` complete=true、elapsed=36.01409175秒。
這些中心供較一般補全探索，沒有當成完整新解，亦沒有低於先前能量8。

上述4,034個中心的四個任意43-bit半列共16,136個fibers也已全部補全，
完整新解零、沒有UNKNOWN，elapsed=115.97766591599999秒。
`L86_weight2_seeds_all_half_rows_20261008` 保存完整輸入、source manifest、每個fiber及console。
下一個較合理的放寬是兩個半列可改，詳見 [下一階段推導](L86_two_half_row_next_derivation.md)。
