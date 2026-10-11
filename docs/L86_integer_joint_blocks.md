# 精確 C/G 四色區塊與 Apple Metal 整數核

這是隔離的研究入口，尚無完整 L86 新解或 20% 新解發現加速證據。
搜尋鄰域有精確數學定義；正溫度走法是啟發式導航，沒有收斂定理。

## 狀態空間與增量

Antipodal 編碼的每個位置只有 c+、c-、g+、g- 四色。
每列選至多 8 個位置，固定其餘位置的四色；
區塊內的偶/奇 binary ones 是 e=n_c+n_g、o=n_c+p_g，
由此列出全部可行四色計數及其全部排列。
兩列只在既有已證明的可提升 contents 內配對；總 C 重量必須為 43。
原始中心仍是可行候選，區塊最小能量不能比原中心更高。

令一列 w 在區塊 S 上的增量是 δ_i，對 1≤u≤21：

\[
\Delta P_w(u)=\sum_{i\in S}\delta_i(w_{i+u}+w_{i-u})
+\sum_{i<j\in S}\delta_i\delta_j
  (\mathbf1_{j-i=u}+\mathbf1_{i-j=u}).
\]

43 為奇數，獨立非零位移不會有重複的 antipodal edge。
每列 C/G 的 42 維增量分別記為 d_A、d_B，原始 residual 為 r。
每個配對的分數精確為 ||r+d_A+d_B||²。
此分數等於原始 binary 目標能量的兩倍（原能量以 1/16 的旁瓣平方和計）。
任何零分配對都逐一重建並核對原始 86 點整數自相關及固定目標。

## 後端核對與修正

浮點 MPS matrix multiplication 曾有極少量非整數誤差，即使輸入界限小且 precision=highest。
不能把數學上 float32 可精確表示整數的論述當成特定後端實作保證。
只改成 Torch int32 後，一次較長實驗仍出現區塊最小能量與完整 binary energy 不一致。
該失敗保留為未完成報告，且同 seed 重播沒有重現，原因未被完整確認。
不能把重播成功說成已查明錯誤根因。

現在 MPS 路徑以 `torch.mps.compile_shader` 呼叫直接 Metal signed-int32 kernels：
逐項算 PACF 增量，以 shared-memory 16×16 tiles 直接累積平方和。
所有 padded threads 也參與所有 barriers。
此邊界下每個中間整數的保守絕對值界為 5,193,216，小於 2³¹。
計算能量不使用浮點 matrix multiplication、epsilon 或四捨五入。
正溫度的 exp/cdf 僅決定導航選取，不能用于排除候選。

固定 L86 區塊的所有 7,280 個保留 row options 以獨立 literal PACF 核對，
所有 3,763,200 個 pair energies 以 CPU int64 核對，全部吻合。
記錄 `results/L86_joint_block_metal_integer_audit_20261007/report.json`。
這個驗證的範圍是該完整矩陣，不是全部可能 L86 區塊。

來源快照保存的較長實驗：`results/L86_joint_blocks_direct_metal_20261007/report.json`。
從 6 個 GS seeds 及 4 個 bounded joint seeds 出發，
temperatures=8/16、seeds=123/456/789、每個 chain 32 sweeps，
共 6 個 complete chains、1,920 個 complete blocks。
總評估 3,586,141,328 個局部配對分數，1,883 次導航移動、600 次 content 改變。
所有區塊的最小配對都通過獨立完整 binary energy 核對。
所有完整新解計數仍為零，最小 full energy 保持 8。
未把本次工作量或吞吐量轉成新解加速率。

API 依據 [PyTorch 2.8 compile_shader 文件](https://docs.pytorch.org/docs/2.8/generated/torch.mps.compile_shader.html)。
實作是 `solver/torch_joint_blocks.py`、`solver/mps_joint_block.py`、`csrc/joint_block.metal`；
獨立實驗入口為 `experiments/search_joint_blocks.py`。
與只固定區塊外支撐、讓所有符號自由的 [另一個區塊方法](L86_exact_support_blocks.md) 有不同鄰域。
