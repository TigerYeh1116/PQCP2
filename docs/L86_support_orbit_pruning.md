# 固定支持補全的獨立列 dihedral UNSAT 剪枝

此剪枝自行推導，實作於 `solver/support_orbit_cache.py`。
`complete_pqcp_supports(..., orbit_pruning=True)` 可使用；預設 False，保留既有路徑。
目前只啟用 L86，不把 UNKNOWN、SAT 或有界鄰域未命中當作排除證明。

對長度 d=43 的任何 signed ternary word w，令
w'_i=w_(εi+t mod d)，其中 ε=±1。直接換 summation index 可得
P_w'(u)=P_w(u)。負號個數、active weight 亦不變。
支持的補集會經同一 permutation 轉換，所以 C support 與 G support 同時保有此雙射。

給定兩列 supports U,V、同一 target 及兩列 C negative counts，
每列可以獨立 rotate/reflect。若原支持存在完整 C/G signs，
逐列施以同一 permutation 就得到變換後支持的 C/G witnesses；inverse permutation 給出反向雙射。
每列的 PACF 與 prescribed counts 保持，因此固定 target 的完整可行性不變。
這個證明不只比較 unsigned support PACF，也沒有把一般 homometric masks 合併。

key 保留完整 target、兩列 negative counts、各列 dihedral support representative，兩列仍有次序。
沒有跨 target k／η、跨 contents、任意 unit decimation、交換 rows 或 signed complement 作商。
cache 只存已由完整符號求解證明的 UNSAT 之 immutable provenance，至多 512 項且加 lock。
新查詢輸出原證明的 source masks；不沿用原支持上的 signed witnesses 或假造 enumeration work。
cache 不寫入 checkpoint，重啟後需重新建立。

小長度測試枚舉全部 3^5 signed words 的 rotations/reflections，逐個比較完整整數 PACF、negative count 與補集支持。
L43 真實案例另比較原支持、獨立旋轉後支持、關閉 cache 的完整求解結果；均 UNSAT。
budget=1 的 UNKNOWN 不存入 cache；不同 target/counts 與容量上限皆有專項檢查。

十萬支持產生的 9 個 exact C witnesses 已使用既有 N2(a)×N2(b) MITM 完整重查。
每個局部範圍只有原 witness，沒有新的 coarse pair；也沒有 orbit cache hit，不能宣稱這批資料有加速收益。
另以 `coarse_content_trades.py` 枚舉一列的兩個 zeros 變成 +1/−1、另一列 +1/−1 變成 zeros，
以完整 integer PACF delta=0 配對，允許 adjacent n_c 變化而保留 target。
9 個 witnesses 在此額外局部範圍仍無 trade；不是對其他 supports 或更大變化的排除。

記錄為 `results/L86_norm_coarse_trades_20261007/` 與 `results/L86_norm_content_trades_20261007/`。
這項剪枝目前是正確性已驗證的可選路徑，尚未有完整新解／20% discovery-speed 的效能證據。
