# AI Capstone Final Project Report

Team 26: 0816064吳中赫 0816169陳伯庭 0816043吳玟叡

---

## 1. Main Ideas

我們這組的想法，是想利用網路上所能收集到的各個畫風肖像畫，透過深度學習中的生成式對抗網路(Generative Adversarial Network, GAN)產生出一張屬於我們自己的肖像畫。產生出來之後，再透過深度學習的風格轉換(Nueral Style transfer)，結合各大著名畫家、不同畫風的畫作，產生出長的一樣，卻不同質感的肖像畫。就相當於各大著名畫家用他們的畫風來模仿了我們畫作，讓他們在逝世的數十數百年後，被我們請出來，用他們的風格來詮釋我們的畫作。

![image-20220610210128683](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220610210128683.png)

## 2. Data Collection

#### GAN

GAN的部分，我們這次的Dataset主要是從Kaggle還有自己爬蟲蒐集各種風格的肖像畫。爬蟲的部分，我們爬的是unsplash這個網站，因為它提供了高畫質的照片，如果用google可能蒐集到一下畫質較低的，對我們整體訓練的過程有害，產生不必要的noise。我們用request裡面的get請求肖像畫的圖片網址，然後用BeautifulSoup去解析respnse![img](https://lh4.googleusercontent.com/__bJIN4cRCPbUoi8uo_i1JU4HioMalHW5XB8rBl89njfqVnSU-bEdxOCHYmcBOC3kJoUZ4GRQJ3wr4IP4aXtj5t11Z6IgXWNOvDVGHrB_icMHmztFEGAGdCcPD03FZ8BhDXF5fJ4vAs4LKG4PA)接著我們從圖片的HTML結構可以看到圖片的class都是"YVj9w”![img](https://lh4.googleusercontent.com/zHXrntPdzq3PwyB-mQfjub1_nFYGpYw-ldYbYZo3ydkEamH7tb7pkV5WAXESc7umbVQ4Ue9TtUynqM7M1NbqR5JEO_yiVmy2zu40eIQwolN72pcydfl3OhRi1c9l_JHw4LyxQati9QMoJf-EAQ)所以我們爬取了所有class為"YVj9w"的圖片，總共5000張，然後把抓下來的圖片通通存到dataset這個資料夾，就結束了。

![img](https://lh6.googleusercontent.com/AQnfmF-GlIjBRrgtoOMInVPV6t1D2z2QcV5hyorX71hIQbqtf9KEvFlvwuyswTCkXYLSXxeVUt98uKWiYilA2dA7XkMDXeIE2YocLr68qhNXmmUPDfBV4Bn0_3Za_AY6oefhW0NGsns7vyYzkQ)

#### Style Transfer:

這部分我們選擇了梵谷、畢卡索、草間彌生各數張的名作作為我們的style images。這些畫家的有比較鮮明的畫風，所以我們預期使用這些畫作為style images較能圖顯style transfer的效果。<img src="https://lh3.googleusercontent.com/fIa8NUO7JtUfq0_s_XpBd-HqNrPp7yHupP0EX34gr_xTLUUFV7_xQI0k6Wv6dxglHq9FK9JP0ovJyYHLyOiTnPQatl1yrMscb9a6AJOuMNYX4r4NnM348nDZFd8iUWnFY8eWvLrumRtIvD_eyvwBDw" alt="img" style="zoom: 25%;" /><img src="https://lh4.googleusercontent.com/4o96XKeeniBJHNNS13bgqbzw3prtkv_t-2drS3mwk3H0weKyjWgyEbXSiDCn_OcMPBgSpP2wWN_kCgMUt7pqHIa7gynzQ8ftWJfZYMMJfCsUh2r1RQGQoHi2ELMnv6bJ_53nbV37CeHBD_n-Cw" alt="img" style="zoom: 50%;" /><img src="https://lh3.googleusercontent.com/VGJBLJk-CS0igTWKe8212nlJ03BlKkmKQ7minmeHvLklSYKoP1KGkG_F7B8F-6nZVS0k0UEp7gDC3Rns7oq9i5Pt_GFm8tH4ypIlA7Ft_R7L4Wg71UcNbQT4hqhSCnDfdqa8-0x5oVHYUQYr2A" alt="img" style="zoom: 50%;" />

## 3. Outlines

接著我們來更進一步的說明我們project以及整體的架構。首先是我們GAN模型的選用，我們一開始直接選擇設計GAN去生成肖像畫。GAN主要是先透過Genrator產生類似所需對應類別的圖片，再透過Discriminator決定生成出的這張照片像不像，然後重複這個循環直到Discriminator覺得生成的夠像為止。

<img src="https://lh6.googleusercontent.com/7Jiw5vNSlgw7DDWWL8GfdcoDeM3QcgyCgNw8ShVbP85xcCQyv1zRF4tvUI64MY8fjRpsQO0dJgLP4d-NLmj7scoY-ihPfEc_YlBqm4nLhhnuMNfa7g2J4An-PeBtYX5xEtUTP6CB9_ALorhldg" alt="img" style="zoom: 45%;" /><img src="https://lh5.googleusercontent.com/BBYqWNO_0m7OoCfJ1MEc4iRBGet5cBURburmagwG4NE-J95FXu_gYq5CYRsBQbAfOIXHNmPkw-LmQkJvBkAh7TgTbBFAEuSzFgztz9ieIPXAfzjN9VSeRcs2XpBHXULpdTnn0StvRGMOgjAw6w" alt="img" style="zoom: 67%;" />

但經過好久的訓練，都得不到理想的成果。我們仔細去研究，發現到後面我們的model幾乎沒有在學習，生成的圖片在後期的training過程甚至有越來越糟的趨勢，看起來是碰上了overfit。所以我們上網查了其他種GAN的變形，發現了Deep Convolutional Generative Adversarial Networks (以下簡稱DCGAN) 。

<img src="https://lh4.googleusercontent.com/Oayc1Z2S0Fb_-IVYcJwZGM3tqiWZ60xIfX6_sukJT7_10YpXsOjVkh1pTIJj_RJfU_q2wRHkd0XaHExQ7r1fvPpITM9UA_c0QeUsSB7vFY5FlmHdlu0K_OnudRELurYaRMFoeZA6GPzcrQk9UA" alt="img" style="zoom:80%;" />

DCGAN有點類似CNN + GAN，只是DCGAN的generator 跟discriminator都捨棄了CNN的pooling layer，用strided卷積代替，discriminator保留了CNN其他特性，generator則將convolutional layer替換成fractional-strided convolution。同時generator跟discriminator在每一層之後都加上了batch normalization，可以減少Training set初始化不良對模型訓練的影響，避免梯度爆炸，提高了訓練的穩定性，也是我們捨棄GAN的目的 - 避免overfitting。而我們設計DCGAN的細節會在下一個section講解。

用DCGAN得到產出的結果後，接著使用Style Transfer，來得到最終結果。我們這次選擇透過VGG19來完成style transfer的任務。利用VGG模型分別提取原圖及風格圖(著名畫家畫作)的特徵。VGG19有19層，分別是16個卷積層以及3個全連接層，結構圖如下。

<img src="https://lh3.googleusercontent.com/IJaa5_K-KJPI3Rxn-h0BrM3sWhRCqlXUyLthKbwdRQjWxvfUGdzHZxUmmCvuW0mJkAIWk82XZ1SSqvGRhG2AM4iTL3XY32Sg9mrbyqEwIVUOCYo-NQCXMDmc-oprJcJUEPhuAO2JFf0e7sRSYQ" alt="img" style="zoom:50%;" />

我們原本使用VGG16，因為擔心VGG19太深，訓練時間過長。但後來發現並沒有耗費多太多時間，效果其實比較好，就採用VGG19了。

## 4. Implementation 

#### Portrait Generator:

首先我們用torchvision裡面的datasets.ImageFolder這個套件來load我們的Dataset，我們發現pytorch也提供一個好用的套件，transform.compose，裡面可以統一圖片的大小，將圖片歸一化到[-1.0,1.0]之間，將圖片轉化成tensor好讓model進行讀取等。![img](https://lh3.googleusercontent.com/doIk3z9gHgXb1A5E7lfJuAG1rEDEPyyOeDuUxdbby3fu4FeEaNsQqi4bcoG9O8JgTFD_IXainOp6dmTP0jYc_i7JFHV1y5MqI-4xcKa4Vqyo8fkQwt92ygUW2sPNiAv4iNHZ6Go9XA80s0IlDg)輸出我們讀取的圖片長怎樣。可以看到64 * 64的圖片解析度還有進步空間，但調整成128 * 128的訓練時間過長，為了節省時間我們還是選用64 * 64的圖片。

<img src="https://lh5.googleusercontent.com/RgzAAxKykDrgi-tWo763kzw1O5OEmwJsXY-Xa8i9qV0MlddDSSGJsugxLC-HLUtjzCReAbH7PTDp_g_ebxthIjdS7VdxjqiJtRiA2zbyUaSK8GEqsJ6Mcm-5kgYLOPnSQc7DTP9bfjZJCDxAnA" alt="img" style="zoom: 50%;" />

接著就是我們model的設計細節，我們主要都是參考DCGAN論文是如何設計與實作的。

首先是Weight Initialization的部分，因為DCGAN論文內有提到Model的weight都必須由Normal Distribution(mean=0, stdev=0.02)隨機產生，所以我們寫了一個function可以直接把隨機的weight丟進model裡。

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220610222455559.png" alt="image-20220610222455559" style="zoom: 67%;" />

DCGAN - Generator: 主要用Pytorch作為框架，並按照倫文內的模型十件，經過上面的fuction後，都有做出noise是random的，然後用fractional-strided convolution來取代2D convolutional layer。![img](https://lh4.googleusercontent.com/75pEIMPyamiS_FsOy-qbf0JatgjwEwFyAKuUfGf_GIWEupU8HVypmBxOnnWNrAm_FhPAqvNzFqPdbCTnbxH74S5-iEtaj6k_0wKY38kX1lPvs0KgG_tvqAcB2V3yV24sVQNPWcCVA38__SN42g)DCGAN - Discriminator: 在DCGAN論文的Discriminator中，提取feature是由一連串的2D Convolutional layer, Batchnorm 跟 LeakyReLU負責的。最後輸出一個單一值，用來代表Discriminator認為是真或假的圖片。那在這邊我們上網查到一種做法，是在最後一層卷基層加上全連結層，而這層會被Generator用來try以及match。以下是我們的discriminator的內部結構，主要是以論文的為主，但input shape會改成3 (channel ) * 64 (image height) * 64 (img_weight)，也可以看到剛剛所說的結合CNN，用stride取代pooling，最後加上sigmoid activation function來回傳0(代表假的圖片)或1(代表真)。![img](https://lh4.googleusercontent.com/7-2oLMyPdtKEg6E4BG5DURUcng14FftWbGUtwN2PfPzPuoL3YIK55EuYuQqpTpWlcThrexcxsT3gdEQD5tgdydOG7OgY6KV8UjoOTp8h7YBhToMOOb9HxP2dfTHH5i9ERfnpBlHYZXszG2O5KQ)最後就是training，主要就是將Generator產出的結果丟進Discriminator，並重複循環到Discriminator以為產出的Artwork是真的Artwork為止。

<img src="C:\Users\user\Downloads\下載 (5).png" alt="下載 (5)" style="zoom: 67%;" />

#### Implementation - Style Transfer:

我們採用VGG19的model，並將image都resize到244*244，weight則是使用imagenet。content layers採用的是單層的block5_conv2 layer，style layers則是五層的block_conv1組成。接著我們就可以把我們的style image丟到style extractor裡面來產生style outputs。接著我們可以利用style content model來產生extractor，再將content image丟進去產生content output。至於後面計算gradient descent及loss function的部分則是參考tensorflow document去做撰寫，使用的optimizer是Adam，style weight設為0.01，content weight設為10000。一開始我們以一張貓咪的圖以及梵谷的《星空》做為測試，直到我們有辦法完整產生一張經過style transfer的貓咪圖後，才將我們在GAN產生的肖像畫拿來做training。由於GAN生成的肖像畫過多，所以我們只選取幾張圖來做style transfer。對於每一張我們選取的style image，我們用它與這幾張我們挑選的肖像畫做style transfer。對於某些GAN產生的，輪廓比較不明顯的圖片，透過style transfer我們除了能夠改變它的繪畫風格之外，也能讓肖像的輪廓更為明顯。

## 5. Results

首先這些是我們從GAN得到的畫作和原本的畫作作比較，可以發現這樣看的話說是真人畫出來的都會相信。

![img](https://lh6.googleusercontent.com/aRLzF9WEaSnXQV6J5chhnLO7PlYqVlxCjJzCkF2iyaqTRKTxpF689H2wFr-ysykQXTJ_xJkx3rQd4BqPTWBbGiK-Unoxwp02BNq5jTmy4KCDezBV834fxxnh9gPHPkKtUKLR-9qVNMchTgtvig)

接著我們就要把剛剛得到的Artwork，丟進Style Transfer。接著這些是我們要拿來當作參考風格的圖畫。

<img src="https://lh3.googleusercontent.com/fIa8NUO7JtUfq0_s_XpBd-HqNrPp7yHupP0EX34gr_xTLUUFV7_xQI0k6Wv6dxglHq9FK9JP0ovJyYHLyOiTnPQatl1yrMscb9a6AJOuMNYX4r4NnM348nDZFd8iUWnFY8eWvLrumRtIvD_eyvwBDw" alt="img" style="zoom: 25%;" /><img src="https://lh4.googleusercontent.com/4o96XKeeniBJHNNS13bgqbzw3prtkv_t-2drS3mwk3H0weKyjWgyEbXSiDCn_OcMPBgSpP2wWN_kCgMUt7pqHIa7gynzQ8ftWJfZYMMJfCsUh2r1RQGQoHi2ELMnv6bJ_53nbV37CeHBD_n-Cw" alt="img" style="zoom: 50%;" /><img src="https://lh3.googleusercontent.com/VGJBLJk-CS0igTWKe8212nlJ03BlKkmKQ7minmeHvLklSYKoP1KGkG_F7B8F-6nZVS0k0UEp7gDC3Rns7oq9i5Pt_GFm8tH4ypIlA7Ft_R7L4Wg71UcNbQT4hqhSCnDfdqa8-0x5oVHYUQYr2A" alt="img" style="zoom: 50%;" />

下方為使用一張GAN產生的肖像畫，經過style transfer的成果。1.肖像畫本身 2.與梵谷的《星夜》進行的style transfer 3.與畢卡索的 《哭泣的女人》進行的style transfer 4.與草間彌生的《花》進行style transfer後的結果

<img src="https://lh5.googleusercontent.com/gjoW8Nn5MGpK7urvrsxQkipJWr0GIMsceQjahRzzyKJh-GAMKw9OJrhoJe-Iqv3cDuw3JQ1KPeC5SLSviiwr8sswnSsKIcCNbyXmCUJqN44KW9ZSVKwwFDB9K-yt4zO411etGTUepKBus2PWZA" alt="img" style="zoom:170%;" /><img src="https://lh6.googleusercontent.com/dB0fXV38ZcvZ25Bnw73iDVmabxC3LysBmauDVpoxc2RAKYFWDwYISKk58JXw6Xlt_3Or1i0MhcmKh5K-u32LxvxYScy0EJ-jm4JIPp4UZ4srX3SpWCr56JvIh50k6OKV00JjB6-8VhAe4POK-w" alt="img" style="zoom:33%;" /><img src="https://lh5.googleusercontent.com/lKh8rOYNtuC22gcFQe954kaPcxG5JLsexh3GwBqjW-KtN9rF_DIS9GWMYJuvIonI-90_8YmDOI4o_5fUQ7Z5lQrYwB__SsvUT9VZkRXG-Bvrr3CghmYk_WF1xy6R10HtM2cYe2XrtgHVwQp5qA" alt="img" style="zoom:33%;" /><img src="https://lh6.googleusercontent.com/xaKiKsApjuzHBJb2ZNh8xe7S0w8Ys5LUwAaJJVWhKHD17gmRvGTn2a-mKrMJzlCI5_B8ofrH3Ochqh3HnOeqbdIGH9BZYlcRHc6usjuM_WpktRIoZGdpwTxJ_NR6f7Zrs2GIFndWCmr1-XMuCQ" alt="img" style="zoom:33%;" />

另外，我們有一個蠻有趣的發現。其實從style transfer後的圖可以很容易地將這些畫家的風格區分開來。可以看到梵谷的畫風非常地樸實，很好地將畫作中的各種元素融合在一起，有一種懷舊的調調；畢卡索的畫風充滿粗曠、扭曲的線條，以及豐富的色彩；草間彌生的畫風也很特別，由各種零碎的幾何圖形組成。1.草間彌生  2.畢卡索 3.梵谷 4.梵谷

<img src="https://lh4.googleusercontent.com/9IBlVJeL0I0oraQRGdEOG-shMkjBqS7JTxwjYmHgrk1hkJrnkSYTiVlzyMNWLncKrTZYapw7TP5MtDBLg5NfSq-pTin7Xqo5KbIIlyy4Ng3_NWFsFEg4kT1nHNqqmYIuTVxljMQofKVXU29QGA" alt="img" style="zoom:33%;" /><img src="https://lh4.googleusercontent.com/Hko9kNfxIWwyjL54jQmRCvXPwRPc6RYSJYzdbIs0_4HDhRybYS_44dFgEdE63fAnUrvBxRbAsujQ1b98yK49mKx5O3pCPacTWIysVuUDs6jF9qre8hIgaMohmvbH1UEq7u2ChKLHmEfhDjayWg" alt="img" style="zoom:33%;" /><img src="https://lh6.googleusercontent.com/0h0UNg0HS7OZN3Ba62j_tu1FfuBp0NylrAgizm8VlbtiMf0RF0wgH5ChjIfCmslBDb_AnILBXdSPVve9ikssEBtdanK0J1a30amNpbdHuPfF-1lK2rd6Ptvzgo4qhuziCwjCbrT4LYJtv8FIlA" alt="img" style="zoom:33%;" /><img src="https://lh4.googleusercontent.com/ag6Sw13v2vvCBIKEGySwxPK5uwN2gRDoXLavitVJV4d4wzBdRnF68jeXyZawySAp88FI9jy4oIIIg8y4SzSXBjtohs_Ffk-Vgv_0-e3sRzLzNth285j9vdYJFGMv0pYfXQ6sgKPSewj66hK_qA" alt="img" style="zoom:33%;" />

## 5. Conclusion & Future Work:	

在DCGAN這邊，我們可以看到雖然我們的Generator-loss呈現下降的趨勢，但生成的圖片從肉眼上來看還是有改進空間。有幾張可以分辨得出這是肖像畫，五官等等卻不夠細節。其中一個重要原因是因為我們只使用了64*64的resized image，所以訓練成效沒辦法達到很高。至於style tranfer的部分，我認為我們的成果有達到style transfer的效果，但是在保持原有肖像畫的特徵上略顯不足。所以未來我們可能嘗試在GAN的部分就做到更大解析度的output，如此style transfer才能有更精緻的結果。

---

Github Link:https://github.com/botingchen/DCGAN_Style_Transfer 

References:https://www.tensorflow.org/tutorials/generative/style_transferhttps://www.tensorflow.org/tutorials/generative/dcgan?fbclid=IwAR36fgQkpG_YPbZ_EhIYmGF92xFqev1xEkw1V-aUfMguGdBw9graH-q8FjQ

Presentation Video:https://drive.google.com/file/d/1700zEtckykeZzNt_zk_kqVd9ZMlcjvNN/view?usp=sharing





