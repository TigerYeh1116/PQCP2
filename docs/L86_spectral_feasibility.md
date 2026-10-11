# C/G spectral feasibility 的精確條件與限制

這是自行推導的實驗分支 `solver/torch_joint_feasibility.py`，尚未加入 main。
真解同時位於固定四色計數集合與 C/G Fourier spheres；這兩個集合非凸。
Douglas–Rachford 迭代只是導航，不能證明收斂或排除沒有命中的區域。

每個 Fourier frequency 的兩列向量需滿足
|F(c_A)|²+|F(c_B)|²=43+2η cos(2πjk/43)，
G 的右側把 η 替換為 η(−1)^k。
給定 Fourier 向量 f，最近 sphere 點的 radial formula 是 sqrt(target/||f||²) f。
零向量選一個固定 sphere 點。實際浮點 DFT／inverse 的輸出不作排除證據。

每個 antipodal position 只有 c+、c−、g+、g− 四種顏色。
固定每種顏色個数，最近 palette 點等價於指定 quota 的線性 assignment。
Native Hungarian 只提供 primal／dual 候選，並不直接當作精確最優證明。

`solver/colour_projection.py` 把所有 represented floats 與 dual potentials
以 `as_integer_ratio` 轉成共同 dyadic scale 的 Python arbitrary integers。
逐個檢查所選等式 u_i+v_j=cost_ij，及全部 inequalities u_i+v_j≤cost_ij。
同顏色 slots 的 cost 相同，所以用該顏色最大 v_j 可一次驗證整組 inequalities。
所有條件成立時，primal cost 與 feasible dual objective 相同，因 weak duality 而精確最優。
若證書失敗，回 Python integer Hungarian；不以容差接受不成立的等式。
最優性針對 represented floating inputs，不代表原始未取樣的實數座標。

每一次 discrete palette projection 都計算整數可表示的 C/G energy 並保存 best，
不再只在 observation interval 檢查，以免漏掉短暫的完整零解。
`observation_interval` 留作 API 相容參數，不再決定候選覆蓋。
取消時仍保留初始 discrete point。全部 lane 的 best 均在原始 binary 長度重算 energy；
只有 full verifier 通過的零解才算解。

已驗證小 count classes 的全部 palette 配置、exact integer fallback、DFT sphere formula、
完整 binary 重建及 interval 不影響候選覆蓋。
此前 MPS smoke 為 32 lanes×256 iterations、16,384 次精確 native assignment certificates、零 fallback；
最佳 full energy 182，沒有新解。此前版本使用 interval 25，不能把它當成每次投影皆被檢查的新版本測量。
