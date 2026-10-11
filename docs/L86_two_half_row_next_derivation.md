# 下一階段：改動兩個半列，避免固定三列的空子空間

這個階段已實作於 `solver/two_half_row_completion.py`；
IDE入口為 `experiments/complete_two_l86_halves.py`。
最新4,034個two-impulse非解中心的16,136個任意單半列fibers全部無解；
four-point的186個中心、744個fibers也是全部無解。
這意味各中心只改一個半列不能改善，不是所有兩個半列改動都無解。

可保留兩個literal半列，第三列允许radius-r內任意位置翻轉，第四列全部43位元自由。
對第三列的每個完整literal變化，再使用既有精確norm／trace或norm completion，
保留所有必要counts／κ／η及最後全長verifier。
這是固定兩列與明確有限第三列球的完整子空間；第四列沒有Hamming距離限制。

若使用一般PQCP等價中心，必須枚舉全部12個有序角色 (varied,free)，
其中varied≠free；獨立row rotation／swap會交換parity及角色。
全12角色及任意位置Hamming球在這些操作下有完整對應。
只保留「改A-odd、補B-odd」不能直接沿用已對一般群取商的center bank宣稱完整。

內容條件可以先排除：已知三列的weights不在完整target_content_profiles的index時，
這個fixed-triple完全不可能。其餘條件沿用原始even intersection及odd trace推導，
不能固定η或默認odd-κ，也不能把第三列的只保持重量swaps稱為全部bit flips。
例如11／7的even sums各改成9時可以跨到even-κ的內容族，這種分支需保留。

實作可新增「輸入三個已知半列」的補全API，未知位置用None；
它沒有輸入非解center，故只驗證三個binary rows及所有輸出的完整PQCPs。
不應為此公開一個可繞過驗證的skip_verification旗標。
原本center API仍需保持非解檢查及完整模型輸出。

先在全部小長度fixed triples對照完整literal解集合，
再選已獨立核對的最低能量新centers、r=1／2及全部12角色作凍結來源實驗。
選擇部分centers是範圍限制，需要記錄；同簽章preimages必須全部保留。
即使這些子空間沒有新解，也不能換算完整新解發現速度。

## 已實作的覆蓋與準備成本

`complete_three_parity_rows` 輸入四個physical half slots，恰一個為None。
它完整求三列固定的所有解，不要求partial assignment的填充列是非解。
因此沒有skip-verification旗標；每個完整輸出仍做原始全長獨立核對。
所有L6的literal固定三列已和完整binary空間比對，包含None補成全零時也有效的解。
兩半列搜尋在小長度全部radius0..3與literal oracle比對，另核對row rotation1造成的parity交換、
row swap及共同reflection之下完整輸出集合的對應。

完全相同的even target equations先合併，不丟掉其任何odd κ／η。
不必要的odd vectors／centered cross profiles延後到真正需要時才計算。
一個center的非zero flip masks在同varied role下有XOR單射性；
不同varied roles改動不同已知列，故不會產生同一literal triple。
只有zero-flip baselines重複，以最多四個free labels即可完整去重，不需儲存百萬key集合。
跨center的cache仍只商完全相同literal triples，容量有界，保留所有完整模型。

新結果ledger預設gzip level1，保留全部資訊並避免radius4產生龐大的未壓縮資料。
結果明確記錄中心選擇、完整ordered-role counts與全部未知43bits，
沒有將counts排除或cache命中當成新解發現率。

## 已完成的有限子空間

新two-impulse最低能量八個中心、radius2：90,912個有序triple指派，
48,824個實際eligible fibers，完整新解零，elapsed=8.7153975秒。
最新長測八個中心、radius2：48,648個eligible fibers，
126,970,443個必要norm visits、10個periodic support orbits，完整新解零。
radius3進一步完成1,275,648個triple指派、386,744個eligible fibers，
511,151,922個norm visits、66個periodic support orbits，完整新解仍零，elapsed=206.677832541秒。
各有獨立source snapshot與input，非全域不存在證明。

接著實作 [聯合Gram證書](L86_joint_gram_pruning.md)，並以相同fibers做配對測試。
實驗入口預設開啟這個已核對的必要条件，core API保留joint_gram=False回退。
已擴到radius4，這個更大子空間的實際結果另行更新。
