## **RAVAN: Multi-Head Low-Rank Adaptation for Federated Fine-Tuning** 

**Arian Raje** _[∗]_ **Baris Askin Divyansh Jhunjhunwala Gauri Joshi** Department of Electrical and Computer Engineering Carnegie Mellon University 

## **Abstract** 

Large language models (LLMs) have not yet effectively leveraged the vast amounts of edge-device data, and federated learning (FL) offers a promising paradigm to collaboratively fine-tune LLMs without transferring private edge data to the cloud. To operate within the computation and communication constraints of edge devices, recent literature on federated fine-tuning of LLMs proposes the use of low-rank adaptation (LoRA) and similar parameter-efficient methods. However, LoRAbased methods suffer from accuracy degradation in FL settings, primarily because of data and computational heterogeneity across clients. We propose RAVAN, an adaptive multi-head LoRA method that balances parameter efficiency and model expressivity by reparameterizing the weight updates as the sum of multiple LoRA heads _si_ **B** _i_ **H** _i_ **A** _i_ in which only the core matrices **H** _i_ and their lightweight scaling factors _si_ are trained. These trainable scaling factors let the optimization focus on the most useful heads, recovering a higher-rank approximation of the full update without increasing the number of communicated parameters since clients upload _si_ **H** _i_ directly. Experiments on vision and language benchmarks show that RAVAN improves test accuracy by 2–8% over prior parameter-efficient baselines, making it a robust and scalable solution for federated fine-tuning of LLMs. 

## **1 Introduction** 

In recent years, the amount of data available on edge devices has increased exponentially, opening the doors for applications that perform machine learning (ML) at the edge. One such paradigm is federated learning (FL), a model training regime where edge devices, or “clients”, collaboratively train a model without sharing their local data with a central server [27]. FL training offers a way to perform large-scale ML on a distributed network of clients by utilizing on-device data while reducing potential privacy risks. The primary challenge in FL is to design methods that are robust in the presence of both data heterogeneity [24, 20, 23, 1]—variations in clients’ local data distributions—and computational heterogeneity [10, 8, 30]—differences in clients’ computing capacities. 

Recent literature has begun exploring the integration of large language models (LLMs) into FL frameworks, driven by the surge in LLM-based edge applications and the resultant need to leverage on-device data for training [40, 41]. Unfortunately, naively training LLMs in FL settings is intractable as a result of the memory constraints of edge devices and communication constraints of wireless networks. As a consequence, these works have examined the impact of parameter-efficient fine-tuning (PEFT) for LLMs in federated settings [17, 41]. These methods reduce the computational load of fine-tuning pretrained LLMs by scaling down the number of trainable parameters. A particularly 

> _∗_ Corresponding Author. `araje@andrew.cmu.edu` 

39th Conference on Neural Information Processing Systems (NeurIPS 2025). 

important PEFT method in FL is low-rank adaptation (LoRA) [16], where the update ∆ **W** is reparameterized as **BA** , the product of two low-rank matrices **B** and **A** . The original pretrained model parameters are frozen throughout fine-tuning, and only the LoRA **B** and **A** parameters ever receive gradient updates, resulting in fine-tuning that is vastly more parameter-efficient than full-parameter fine-tuning. LoRA-based methods are a promising alternative to full-parameter fine-tuning in FL since clients only have to train and communicate the LoRA parameters, simultaneously addressing computation and communication bottlenecks. 

However, LoRA-based methods are highly affected by client data heterogeneity (see Table 1) because re- stricting updates to a low rank subspace deprives the model of the capacity needed to fit the diverse directions introduced by heterogeneous data. Moreover, directly extending LoRA to FL, as done in FedIT [41], leads to an inexactness problem during aggregation. Since **BA** is a proxy for the true model update 

Table 1: Accuracy comparison on CIFAR-100 [22], non-I.I.D. clients (Dirichlet _α_ = 0 _._ 3) 

||**Method**|**I.I.D.**|**Non-I.I.D.**|
|---|---|---|---|
||Full-FT|89_._78|85_._17|
||FedIT|81_._75|68_._15|
||FedEx-LoRA|77_._82|66_._98|
||FFA-LoRA|78_._17|59_._89|



∆ **W** , averaging the **B** and **A** parameters separately would be inconsistent with the true model update: 

**==> picture [336 x 37] intentionally omitted <==**

where _C_[(] _[t]_[)] is the selected client set at round _t_ . Previous works that seek to address this exact aggregation issue suffer from accuracy loss and poor scalability in practical FL settings due to data and computational heterogeneity. FFA-LoRA [34] manages exact updates by freezing the **A** parameter at initialization but reduces the model expressivity relative to vanilla LoRA. FedEx-LoRA [32] adds the inexact residual _|C_[(] 1 _[t]_[)] _|_ � _c∈C_[(] _[t]_[)] **[ B]** _c_[(] _[t]_[)] **[A]**[(] _c[t]_[)] _[−]_[(] _|C_[(] 1 _[t]_[)] _|_ � _c∈C_[(] _[t]_[)] **[ B]**[(] _c[t]_[)][)(] _|C_[(] 1 _[t]_[)] _|_ � _c∈C_[(] _[t]_[)] **[ A]**[(] _c[t]_[)][)][ to the original] pretrained weights **W** to get an exact update every round. However, the method substantially increases the communication cost of fine-tuning since the updated model weights also have to be communicated every round. Critically, these LoRA-based methods can afford only small ranks within a limited - parameter budget. When the true update is high rank, this approximation disregards most of the update’s variance and limits accuracy. Fed-SB [33], motivated by the update approximation from LoRA-XS [5], introduces a third LoRA parameter between the standard **B** and **A** parameters and only fine-tunes this additional parameter. However, it necessitates an initial round of full-parameter finetuning to initialize **B** and **A** , which is prohibitively expensive in FL. Furthermore, the initialization may become stale as training progresses due to data heterogeneity and partial participation. 

Computational heterogeneity is an additional scalability challenge in practical FL systems. Clients may vary significantly in their hardware resources and computational capabilities, making it difficult for all clients to fine-tune models at the same scale and speed. The methods described above do not allow for LoRA parameters of different sizes across clients. HetLoRA [9] and FlexLoRA [4] allow clients to train varying-rank LoRA parameters, but the methods struggle in the presence of data heterogeneity and do not ensure exact aggregation. In this vein, our goal is to design a method for FL that 1) performs efficient computation and communication throughout the training procedure, 2) remains robust in the presence of heterogeneity, and 3) retains the property of exact aggregation. 

To this end, we propose RAVAN[2] , an adaptive multi-head LoRA method that sharply reduces the number of trainable parameters while maintaining accuracy in the presence of data and computational heterogeneity. We take inspiration from multi-head approaches such as HydraLoRA [35]; however, when naively ported to federated settings, those methods fail to guarantee exact aggregation and cannot raise the effective rank of the updates under a fixed parameter budget. Our design meets both requirements. RAVAN re-parameterizes each weight update ∆ **W** as a weighted sum of low-rank heads _si_ **B** _i_ **H** _i_ **A** _i_ , where the bases **B** _i_ and **A** _i_ are frozen at initialization and only the **H** _i_ parameters and lightweight scaling parameters _si_ are trained. We choose **B** _i_ and **A** _i_ with mutually orthogonal column and row spaces, respectively, and thus the heads combine to achieve a higher effective rank without exceeding the original LoRA parameter budget. When clients differ in resources, the most constrained devices can fine-tune only a subset of heads and leave the rest frozen. Uploading the products _si_ **H** _i_ preserves exact aggregation and the method incurs no extra communication cost. Extending prior efforts, RAVAN introduces an integrated framework that maintains parameter-efficient computation 

> 2RAVAN derives its name from the mythical 10-headed villain from the Hindu epic, _Ramayana_ 

2 

**==> picture [397 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
Top-64 Singular Values of W Top-64 Singular Values of W<br>10 [0]<br>10 [0]<br>10 1 10 1<br>(a) CIFAR-100 Spectrum (b) SVHN Spectrum<br>Centralized FL w/ I.I.D. ClientsFL w/ Non-I.I.D. Clients Centralized FL w/ I.I.D. ClientsFL w/ Non-I.I.D. Clients<br>Singular Value (log-scale) Singular Value (log-scale)<br>**----- End of picture text -----**<br>


Figure 1: Singular value spectra of the weight updates ∆ **W** for CIFAR-100 and SVHN [29] in three different training regimes. We display only the 64 largest values (hence the truncated plots). Moving from centralized learning _→_ FL (I.I.D. clients) _→_ FL (non-I.I.D. clients), the median shifts up and the distribution becomes broader, meaning a larger fraction of singular values remains near the higher end of the spectrum. The effective rank is, therefore, higher in the federated, non-I.I.D. setting. 

and communication and demonstrates robustness across diverse data and computational heterogeneity in federated settings. Across all benchmarks, RAVAN outperforms related federated PEFT methods in both I.I.D. and non-I.I.D. settings, demonstrating its strength in the presence of heterogeneity and client diversity. 

## **2 Problem Setup and Motivation** 

LoRA is a PEFT method that reparameterizes the weight updates to reduce the number of trainable parameters. It contends that the full-parameter weight update ∆ **W** full can be approximated as follows: 

**==> picture [252 x 25] intentionally omitted <==**

For notational ease, we write each weight matrix **W** and its corresponding update ∆ **W** full as a square matrix in R _[d][×][d]_ , but all derivations extend directly to the general rectangular case in which **W** _∈_ R _[m][×][n]_ . The method contends that ∆ **W** full exists in a low-rank subspace and can therefore be represented as the product of two low-rank matrices. In this context, the low-rank matrices, referred to as the LoRA parameters, have dimensions **B** _∈_ R _[d][×][r]_ and **A** _∈_ R _[r][×][d]_ . To perform LoRA fine-tuning, the pretrained model weights **W** are frozen throughout training and only **B** and **A** receive gradient updates. Since _r ≪ d_ , the number of trainable parameters decreases from _d_[2] to 2 _rd_ . 

**Importance of Higher-Rank Update Approximation.** A key limitation of LoRA is that constraining the approximation of the update ∆ **W** full to the low-rank subspace spanned by **BA** can limit its expressive capacity. When the rank _r_ is set too low, the approximation of ∆ **W** full may fail to capture the full complexity and variation present in the full-rank gradient updates. This limitation is amplified when we perform fine-tuning in a federated setting, as we observe in Figure 1 which shows the spectra of singular values of the _full-parameter_ weight updates, ∆ **W** full, in three different training regimes (centralized learning, FL with I.I.D. clients, and FL with non-I.I.D. clients). The model is trained to a target accuracy, and the weight update is decomposed using singular value decomposition (SVD). Figure 1 demonstrates that the “effective rank” of the weight updates is larger in the federated non-I.I.D. setting. Intuitively, the greater the diversity among client updates, the more the spectral mass is spread across singular vectors, thereby increasing the effective rank. This suggests that low-rank approximations of the weight updates discard more information and fail to capture many of the significant intrinsic dimensions of the true updates. 

**Improving the Effective Rank and Expressivity of LoRA.** A naive way to capture the richer spectrum of weight updates is to raise the LoRA rank _r_ , but that linearly increases the number of trainable parameters. To better approximate the higher-rank update, as proposed in LoRA-XS, we 

3 

Figure 2: **Left** : Within the same parameter count, the effective rank of the LoRA parameters increases when using an augmented third parameter and multiple heads. **Right** : Clients with various computational constraints can freeze certain heads to reduce memory consumption. 

can augment the traditional LoRA approximation with an additional parameter, **H** , as follows: 

**==> picture [252 x 25] intentionally omitted <==**

where **B** and **A** remain frozen and only **H** is trained. Suppose we have a trainable parameter budget of _N_ parameters. In the case of vanilla LoRA, _N_ = 2 _rd_ and _r_ = Θ(1). With the augmented version of LoRA that has parameters **BHA** and frozen **B** and **A** , we can instead use the much larger rank of _√N_ = Θ( _√d_ ). Additionally, if used in FL, this setup would avoid inexactness in the aggregation of the disparate client models since only the **H** parameter is averaged across clients. 

**Multiple Heads for Further Rank Improvements.** We can further improve the effective rank of the update approximation by using multiple concurrent augmented LoRA heads. Again, suppose we have a trainable parameter budget of _N_ parameters. We use _h_ heads, where each head _i ∈_ [1 _, . . . , h_ ] has the structure **B** _i_ **H** _i_ **A** _i_ and each **B** _i_ and **A** _i_ is frozen at initialization. With this reparameterization of the weight update, each head has rank _Nh_[.][Using the sub-additivity of rank, we instead have:] 

**==> picture [384 x 30] intentionally omitted <==**

Within the same trainable parameter budget, by using _h_ heads, we can improve the rank expressivity of the augmented version of LoRA by a factor of _√h_ . Furthermore, using heads of the form **B** _i_ **H** _i_ **A** _i_ , where **B** _i_ and **A** _i_ are fixed, retains the property of exact aggregation because the following is true: 

**==> picture [318 x 36] intentionally omitted <==**

Note that methods like HydraLoRA and LoRAMoE [12] that use multiple vanilla LoRA heads of the form **B** _i_ **A** _i_ (instead of the augmented **B** _i_ **H** _i_ **A** _i_ form) do not confer these same benefits of increased rank expressivity and exact aggregation. Suppose we have _N_ trainable parameters; each vanilla LoRA head would have dimensions **B** _i ∈_ R _[d][×]_ 2 _[N] dh_ and **A** _i ∈_ R 2 _Ndh[×][d]_ . We would then have the same maximum effective rank as vanilla LoRA because of the following inequality: 

**==> picture [359 x 30] intentionally omitted <==**

Since the number of trainable parameters _N_ = Θ( _d_ ) (recall that _N_ = 2 _dr_ for standard LoRA), this rank is _N/_ 2 _d_ = Θ(1), which is much smaller than the rank _√Nh_ = Θ( _√dh_ ) achieved by multiple augmented LoRA heads. 

4 

**Addressing Computational Heterogeneity.** In a realistic resource-heterogeneous federation, clients may have vastly different computational capacities. A PEFT scheme that forces every device to train the same size LoRA parameters will exclude the weakest clients or throttle the strongest. In the FL setting specifically, using multiple heads has the additional benefit of providing a way to manage computational heterogeneity. Clients with more significant resource limitations can freeze subsets of the heads and only fine-tune the remaining heads (Figure 2, **Right** ). This further reduces the memory requirements of local fine-tuning and prevents clients from having to drop out of the FL procedure due to stricter resource constraints. This partial freezing scheme still avoids inexactness in the aggregate updates, unlike previous heterogeneous-rank works in FL. 

## **3 Proposed Method** 

In this section, we present RAVAN, a method that uses multiple augmented LoRA heads to perform efficient LLM fine-tuning in the presence of data and computational heterogeneity. For pretrained weights **W** in the model _M_ , the forward pass is replaced by the following: 

**==> picture [143 x 45] intentionally omitted <==**

The pretrained **W** along with each **B** _i_ and **A** _i_ are frozen at the start of training. As a consequence, only the **H** _i_ parameters and the lightweight scaling factors _si_ are updated and communicated throughout the FL training procedure. The pseudocode of the proposed method RAVAN is given in Algorithm 1, and the following sections highlight key components of our framework. Specifically, we examine the importance of initialization in improving the update approximation (Section 3.1). We additionally analyze strategies for per-client head subset selection and aggregation for computational heterogeneity (Section 3.2). Together, these design choices let RAVAN match the communication cost of 

## **Algorithm 1** RAVAN 

**Require:** Clients _C_ , Model _M_ , Rounds _T_ , Local Steps _S_ , LR _ℓ_ , Rank _r_ , Heads _h_ 

1: **Initialization** : 2: INIT( **B** _i,_ **H**[(0)] _i[,]_ **[ A]** _[i]_[)][,] _[ i]_[ = [1] _[, . . . , h]_[]][ for] _[ M]_ 3: Freeze original model parameters, **B** _i,_ **A** _i_ 4: **Model Training** : 5: **for** _t_ = 1 **to** _T_ **do** 6: Reset _s_[(] _i[t]_[)] _←_ 1 _, i ∈_ [1 _, . . . , h_ ] 7: Select active client subset _C_[(] _[t]_[)] 8: Broadcast _{_ **H**[(] _i[t]_[)] _[}] i[h]_ =1[to] _[ C]_[(] _[t]_[)] 9: **for all** _c ∈C_[(] _[t]_[)] **in parallel do** 10: _Hc_[(] _[t]_[)] _←_ SELECTHEADS( _c, t_ ) 11: **for** _τ_ = 1 **to** _S_ **do** 12: Update _s_[(] _i[t]_[)] _[,]_ **[ H]**[(] _i[t]_[)] for _i ∈Hc_[(] _[t]_[)] 13: **end for** 14: _c_ sends _{s_[(] _c,i[t]_[)] **[H]**[(] _c,i[t]_[)] _[}] i∈H_[(] _c[t]_[)] to server 15: **end for** 16: **for** _i_ = 1 **to** _h_ **do** 1 17: Update **H**[(] _i[t]_[+1)] _←_ � _s_[(] _c,i[t]_[)] **[H]**[(] _c,i[t]_[)] _|Ci_[(] _[t]_[)] _[|] c∈Ci_[(] _[t]_[)] 18: **end for** 19: **end for** 

vanilla LoRA, while delivering higher-rank, resource-aware updates that preserve exact aggregation. 

## **3.1 Parameter Initialization** 

Fine-tuning efficiency hinges on the initialization of the LoRA parameters. The standard LoRA initialization sets **B** = **0** and draws **A** _∼N_ (0 _, σ_[2] ). In RAVAN, this initialization cannot be used since each **B** _i_ **H** _i_ **A** _i_ would remain **0** throughout training as **B** _i_ and **A** _i_ are frozen at initialization. Therefore, we must draw non-zero **B** _i_ and **A** _i_ and set **H** _i_ = **0** so that fine-tuning starts from the original pretrained weights but updates the LoRA parameters throughout the training procedure. In this section, we provide methods for effective initializations for the **B** _i_ and **A** _i_ parameters. These initializations do not require performing full-parameter fine-tuning of the original LLM weights such as the initialization presented in Fed-SB. Since each **B** _i_ and **A** _i_ parameter are fixed, the expressive power of the sum[�] _[h] i_ =1 _[s][i]_ **[B]** _[i]_ **[H]** _[i]_ **[A]** _[i]_[ is limited by the subspaces spanned by the column spaces of the] **B** _i_ and row spaces of the **A** _i_ . We test the following two methods to obtain orthogonal subspaces: 

- **Random Normal:** Set **B** _i ∼N_ (0 _, σB_[2][)][and] **[A]** _[i][∼N]_[(0] _[, σ] A_[2][)][.][In][high-dimensional][space,][their] column and row spaces are orthogonal with high probability. 

- **Gram-Schmidt:** For the **B** _i_ parameters, we concatenate the _rh_ columns of random normal initialized [ **B** 1 _, . . . ,_ **B** _h_ ] _∈_ R _[d][×][rh]_ and apply the Gram-Schmidt procedure in the column space. 

5 

This yields an orthonormal set _{_ _**b**_[˜] _k}[rh] k_ =1[.][The orthonormal set can be sliced back into] _[ h]_[ blocks of] width _r_ to form the **B** _i_ . For the **A** _i_ parameters, we can apply the Gram-Schmidt procedure in the row space of the concatenated **A** _i_ ’s. With this initialization, orthogonality holds deterministically. We benchmark our initializations against a constant initialization where **B** _i_ = **B** _j,_ **A** _i_ = **A** _j ∀i, j_ . We test an additional more flexible initialization benchmark where **B** _i_ = **MR** _i,_ **A** _i_ = **R** _i_ **N** for normally distributed **M** _∈_ R _[d][×][r] ,_ **N** _∈_ R _[r][×][d]_ and invertible **R** _i ∈_ R _[r][×][r]_ which are different for each head. We refer to this baseline as “shared subspace” since the initialization ensures that the column and row spaces of the **B** _i_ and **A** _i_ parameters are identical. On both vision and language tasks, the random normal and Gram–Schmidt initializations deliver the highest test accuracy, confirming that mutually orthogonal **B** _i_ and **A** _i_ increase the effective rank of the update approximation which translates directly into better downstream performance. Full numbers are reported in Section 4.3. 

## **3.2 Head Selection Strategies** 

RAVAN allows clients to choose subsets of the LoRA heads to fine-tune. This is particularly advantageous in FL where client devices often possess widely varying computational capacities. Suppose we have a participating client set in communication round _t_ , _C_[(] _[t]_[)] . For client _c ∈C_[(] _[t]_[)] , we define � ( _t_ ) **H** _c,i_[=] _[ s]_[(] _c,i[t]_[)] **[H]**[(] _c,i[t]_[)] _[∀][i][ ∈]_[[1] _[, . . . , h]_[]][.][At the beginning of each local training step, each client evaluates a] ( _t_ ) scoring function _ρ_[(] _c,i[t]_[)][=][ score][(] **[H]**[�] _c,i[,][ D][c]_[)] _[ ∀][i][ ∈]_[[1] _[, . . . , h]_[]][ on its local data] _[ D][c]_[.][In the following,] _[ ∥· ∥][F]_ is the Frobenius norm of the input matrix. The Frobenius norm of a matrix is calculated as the square root of the sum of the squares of all its entries. We employ the following three scoring functions: 

- **Random Scoring:** _ρ_[(] _c,i[t]_[)] _[∼]_[Unif][(0] _[,]_[ 1)][.] Heads receive random scores, so clients form their fine-tuning subset by uniformly sampling heads at random. 

- ( _t_ ) 

- • **Weight-Based Scoring:** _ρ_[(] _c,i[t]_[)][=] _[∥]_ **[H]**[�] _c,i[∥][F]_[.][Heads whose weights have the largest magnitude are] assigned a higher score and deemed more influential. 

- **Gradient-Based Scoring:** _ρ_[(] _c,i[t]_[)][=] _[ ∥∇]_ **H** �( _c,it_ ) _[L][c][∥][F]_[for a single mini-batch with all other heads frozen.] Heads whose gradients have the largest magnitude are deemed more influential. 

Client _c_ selects the top _Kc_ heads ranked by _ρ_[(] _c,i[t]_[)][and][forms][the][selection][set] _[H] c_[(] _[t]_[)] = _{i ∈_ [1 _, . . . , h_ ] _| i_ is among the top _Kc_ heads _}_ where the value of _Kc_ depends on client _c_ ’s computational constraints. During local fine-tuning, client _c_ only updates heads in _Hc_[(] _[t]_[)][.][Let] _[ C] i_[(] _[t]_[)] denote the set of clients that fine-tuned head _i_ in communication round _t_ . The server performs the update: 

**==> picture [294 x 49] intentionally omitted <==**

Equation (8) ensures exact aggregation of the _si_ and **H** _i_ parameters by directly averaging their product and reinitializing each _si_ = 1 _∀i ∈_ [1 _, . . . , h_ ] at the start of every communication round. An alternative measure to ensure exact aggregation is to fix each _si_ = 1 at the start of training. We would then directly average the individual **H** _i_ ’s so that **H**[(] _i[t]_[+1)] = _|Ci_[(] 1 _[t]_[)] _|_ � _c∈Ci_[(] _[t]_[)] � **H**[(] _c,i[t]_[)] �. We compare these two aggregation schemes in Section 4.3 and find consistent improvements with scaling factors. 

## **4 Experiments** 

## **4.1 Experimental Setup** 

**Dataset and Model Usage.** For image classification, we adopt **ViT-B/16** [11] (85 M parameters) and fine-tune on two benchmarks: (i) CIFAR-100 (50,000 train / 10,000 test images, 100 classes) and (ii) SVHN (73,250 train / 26,032 test digits, 10 classes). For natural-language tasks, we fine-tune **T5-Base** [31] (224 M parameters) on (i) 20 Newsgroups [28] (11,300 train / 7,532 test articles, 20 topics) and (ii) MRQA [14] (516,800 train / 58,221 test examples). The MRQA corpus is the union of six sources (HotpotQA, NaturalQuestions, NewsQA, SearchQA, SQuAD, TriviaQA). 

6 

**Federated Partitioning.** We create federated splits with _|C|_ = 20 or _|C|_ = 50 clients. For I.I.D. partitions, clients receive an equal-sized random subsample of the global training set. For non-I.I.D. partitions, we draw client-specific class proportions from a Dirichlet distribution with _α_ =0 _._ 3. For MRQA, which lacks class labels, the Dirichlet split is performed over the six constituent sub-datasets. 

Table 2: Performance comparison on CIFAR-100 and SVHN. 

|Method<br>Rank|**CIFAR-100 (Acc. %)**<br>20 Clients<br>50 Clients<br>I.I.D.<br>Non-I.I.D.<br>I.I.D.<br>Non-I.I.D.|**CIFAR-100 (Acc. %)**<br>20 Clients<br>50 Clients<br>I.I.D.<br>Non-I.I.D.<br>I.I.D.<br>Non-I.I.D.|**SVHN (Acc. %)**<br>20 Clients<br>50 Clients<br>I.I.D.<br>Non-I.I.D.<br>I.I.D.<br>Non-I.I.D.|**SVHN (Acc. %)**<br>20 Clients<br>50 Clients<br>I.I.D.<br>Non-I.I.D.<br>I.I.D.<br>Non-I.I.D.|
|---|---|---|---|---|
||20 Clients<br>I.I.D.<br>Non-I.I.D.||20 Clients<br>I.I.D.<br>Non-I.I.D.||
|Full-FT<br>N/A|89_._89<br>86_._86|89_._78<br>85_._17|95_._06<br>90_._29|94_._90<br>89_._49|
|_N_total = 1_._2M<br>FedIT<br>32<br>FedEx-LoRA<br>32<br>FFA-LoRA<br>64<br>Fed-SB<br>221<br>RAVAN<br>110<br>_N_total = 2_._4M<br>FedIT<br>64<br>FedEx-LoRA<br>64<br>FFA-LoRA<br>128<br>Fed-SB<br>313<br>RAVAN<br>156|83_._49<br>68_._66<br>80_._56<br>67_._45<br>78_._82<br>56_._34<br>79_._27<br>71_._48<br>**84**_._**42**<br>**76**_._**22**<br>83_._82<br>71_._01<br>79_._38<br>50_._47<br>81_._39<br>70_._31<br>83_._03<br>73_._12<br>**85**_._**04**<br>**77**_._**20**|81_._75<br>68_._15<br>77_._82<br>66_._58<br>78_._17<br>59_._98<br>79_._06<br>69_._51<br>**84**_._**02**<br>**73**_._**80**<br>84_._04<br>73_._23<br>79_._42<br>57_._86<br>82_._13<br>66_._81<br>83_._90<br>71_._13<br>**85**_._**55**<br>**77**_._**81**|88_._66<br>84_._00<br>91_._94<br>84_._30<br>91_._53<br>86_._03<br>90_._94<br>82_._25<br>**94**_._**13**<br>**90**_._**02**<br>91_._39<br>84_._68<br>91_._16<br>74_._04<br>91_._95<br>88_._06<br>92_._29<br>86_._89<br>**93**_._**92**<br>**89**_._**41**|91_._67<br>77_._53<br>91_._51<br>81_._63<br>91_._82<br>83_._30<br>92_._74<br>85_._30<br>**93**_._**75**<br>**89**_._**17**<br>92_._06<br>79_._31<br>92_._01<br>74_._84<br>92_._07<br>84_._24<br>92_._78<br>82_._46<br>**94**_._**28**<br>**84**_._**34**|



Table 3: Performance comparison on 20 Newsgroups and MRQA. 

|Method<br>Rank|**20 Newsgroups (Acc. %)**<br>20 Clients<br>50 Clients<br>I.I.D.<br>Non-I.I.D.<br>I.I.D.<br>Non-I.I.D.|**20 Newsgroups (Acc. %)**<br>20 Clients<br>50 Clients<br>I.I.D.<br>Non-I.I.D.<br>I.I.D.<br>Non-I.I.D.|**MRQA (F1 %)**<br>20 Clients<br>50 Clients<br>I.I.D.<br>Non-I.I.D.<br>I.I.D.<br>Non-I.I.D.|**MRQA (F1 %)**<br>20 Clients<br>50 Clients<br>I.I.D.<br>Non-I.I.D.<br>I.I.D.<br>Non-I.I.D.|
|---|---|---|---|---|
||20 Clients<br>I.I.D.<br>Non-I.I.D.||20 Clients<br>I.I.D.<br>Non-I.I.D.||
|Full-FT<br>N/A|71_._34<br>69_._29|71_._71<br>70_._13|62_._19<br>62_._25|62_._41<br>62_._51|
|_N_total = 2_._4M<br>FedIT<br>32<br>FedEx-LoRA<br>32<br>FFA-LoRA<br>64<br>Fed-SB<br>221<br>RAVAN<br>110<br>_N_total = 4_._7M<br>FedIT<br>64<br>FedEx-LoRA<br>64<br>FFA-LoRA<br>128<br>Fed-SB<br>313<br>RAVAN<br>156|**69**_._**07**<br>61_._98<br>69_._04<br>62_._52<br>68_._11<br>62_._36<br>67_._15<br>63_._10<br>68_._96<br>**65**_._**73**<br>**69**_._**36**<br>64_._41<br>68_._59<br>65_._11<br>69_._33<br>66_._22<br>68_._07<br>64_._18<br>69_._29<br>**66**_._**45**|67_._99<br>60_._67<br>**68**_._**19**<br>63_._33<br>68_._00<br>64_._86<br>66_._69<br>63_._98<br>68_._18<br>**65**_._**67**<br>68_._12<br>62_._67<br>67_._75<br>64_._31<br>68_._42<br>64_._86<br>67_._58<br>65_._59<br>**68**_._**89**<br>**66**_._**85**|61_._00<br>60_._57<br>60_._99<br>**60**_._**68**<br>60_._31<br>60_._40<br>59_._93<br>59_._73<br>**61**_._**18**<br>60_._45<br>61_._25<br>60_._75<br>61_._23<br>60_._36<br>61_._50<br>60_._50<br>60_._22<br>60_._11<br>**61**_._**82**<br>**61**_._**33**|61_._24<br>60_._52<br>**61**_._**40**<br>60_._56<br>61_._21<br>60_._14<br>59_._96<br>60_._01<br>61_._33<br>**61**_._**53**<br>61_._39<br>60_._26<br>61_._43<br>60_._06<br>61_._66<br>60_._12<br>60_._28<br>60_._60<br>**61**_._**73**<br>**61**_._**26**|



## **4.2 Main Results: Vision and Language** 

We consider an FL setting with partial client participation where, in each communication round, the server samples three clients uniformly at random. Every selected client performs 50 local training iterations before uploading its update. Note, we intentionally train for 50 mini-batches and not 50 entire traversals of the client’s training dataset so that each client performs exactly the same number of forward-backward passes. We evaluate two separate trainable parameter budgets. The upper half of Tables 2 and 3 correspond to the lower budget and the lower half to the higher budget. The RAVAN configuration uses 4 heads where each head **H** _i ∈_ R _[r][×][r]_ and _r_ is the specified rank. All results displayed in the following sections are the averages across 3 random seeds. 

RAVAN achieves the best performance among all PEFT methods in all vision configurations and in 11/16 of the language configurations. A key finding is that its advantage widens systematically in the statistically heterogeneous regime. For CIFAR-100 with 50 non-I.I.D. clients, RAVAN exceeds the performance of FedEx-LoRA by 7.2% and FedIT by 5.6% at the lower budget, whereas the corresponding I.I.D. gains are 6.2% and 2.3% respectively. The language results also display larger improvements in the non-I.I.D. paradigm. On 20 Newsgroups, the gap over Fed-SB reaches 2.6% with 20 non-I.I.D. clients in the lower parameter budget. On MRQA, the pretrained T5-Base already attains a strong F1 score, so all PEFT methods yield only modest absolute gains. Consequently, the scores of different approaches are tightly clustered in the lower-budget rows. Even in this regime, in the higher parameter budget setting, RAVAN outperforms every baseline in every MRQA configuration. 

7 

**==> picture [370 x 134] intentionally omitted <==**

**----- Start of picture text -----**<br>
Accuracy with Computational Heterogeneity Accuracy with Computational Heterogeneity<br>0.95 0.95<br>0.90 HetLoRA RAVAN (Random) RAVAN (Gradient) 0.90<br>FlexLoRA RAVAN (Weight)<br>0.85 0.85<br>0.80 0.80<br>0.75 0.75<br>0.70 0.70<br>0.65 0.65<br>0.60 0.60<br>0.55 0.55<br>(a) CIFAR-100 Performance Comparison (b) SVHN Performance Comparison<br>Bell-Shaped Uniform Skewed Right Bell-Shaped Uniform Skewed Right<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 3: Clients draw trainable parameter budget from bell-shaped, uniform, or skewed right distributions. All RAVAN variants outperform the baselines in every distribution. 

## **4.3 Ablation Studies and Analysis** 

**Initialization Comparison.** Table 4 compares the initializations for the fixed bases **B** _i,_ **A** _i_ as described in Section 3.1. The fully orthogonal Gram-Schmidt initialization is consistently best on the vision tasks, adding 1.2% over random normal on SVHN and outperforming the constant baseline by 20% on CIFAR100. On language tasks, the random normal initialization outperforms the constant baseline by 8.9% on 20 Newsgroups and the shared subspace baseline by 0.26% on MRQA. Thus, the proposed orthogonal initializations maximize the effective rank of the update and yield the strongest accuracy across all domains. While Gram-Schmidt outperforms other schemes on 

Table 4: Initialization comparison with 20 non-I.I.D. clients and lower parameter budget. 

||Method|**CIFAR-100**|**SVHN**|
|---|---|---|---|
||Random Normal|76_._22|90_._02|
||Gram-Schmidt<br>Constant<br>Shared Subspace|**78**_._**75**<br>58_._12<br>57_._39|**91**_._**25**<br>88_._01<br>84_._54|
|||||
||Method|**20 Newsgroups**|**MRQA**|
||Random Normal|**65**_._**73**|**60**_._**45**|
||Gram-Schmidt|64_._83|59_._71|
||Constant<br>Shared Subspace|56_._74<br>55_._64|60_._43<br>60_._19|



the vision tasks, the procedure is more computationally expensive in high-dimensional space. However, since this initialization is a one-time server-run operation at the start of training, amortized across the entire FL procedure, it adds virtually no overhead to the fine-tuning workload. 

**Computational Heterogeneity.** We emulate devices with unequal trainable parameter budgets by drawing each client’s trainable parameter budget from three fixed distributions (bell-shaped, uniform, and skewed right). Details on the individual distributions can be found in the Appendix. As displayed in Figure 3, in these settings, all RAVAN variants outperform the rank-adaptive baselines HetLoRA and FlexLoRA on both CIFAR-100 and SVHN. On CIFAR-100, the strongest baseline loses 11% of overall accuracy when moving from the bell-shaped distribution to the skewed right distribution. In comparison, RAVAN only loses 2% on average across its 3 variants. A key reason is that FlexLoRA performs a per-client SVD and HetLoRA performs hard-rank truncation when redistributing the global model to individual clients. RAVAN avoids these approximation operations, so its updates remain accurate even in the extreme skewed right case. We note, however, that weightbased scoring consistently lags behind the other two scoring mechanisms because it always selects - the same high magnitude heads across all clients. Random scoring and gradient-based scoring more evenly distribute updates across all heads and are better suited for heterogeneous fine-tuning in FL. 

Table 5: Effect of using trainable scaling factors with non-I.I.D. clients and lower parameter budget. 

|Method|**CIFAR-100**<br>20 Clients<br>50 Clients|**SVHN**|**20 Newsgroups**<br>20 Clients<br>50 Clients|**MRQA**<br>20 Clients<br>50 Clients|
|---|---|---|---|---|
|||20 Clients<br>50 Clients|||
|Constant<br>Trainable|74_._93<br>71_._53<br>**76**_._**22**<br>**73**_._**80**|**90**_._**35**<br>**89**_._**58**<br>90_._02<br>89_._17|65_._24<br>65_._39<br>**65**_._**73**<br>**65**_._**67**|60_._45<br>61_._44<br>**60**_._**60**<br>**61**_._**53**|



**Influence of Scaling Factors.** Table 5 compares the setting where scaling factors are set to a constant _si_ = 1 _∀i ∈_ [1 _, . . . , h_ ] throughout the fine-tuning procedure in comparison to the standard RAVAN algorithm where the scaling factors are trainable. Keeping the scaling factors trainable boosts 

8 

**==> picture [370 x 134] intentionally omitted <==**

**----- Start of picture text -----**<br>
Effect of Number of Heads Effect of Number of Heads<br>1.4 3.5<br>p Nh = d<br>1.2 SVHN 3.0<br>1.0 20NewsGroups 2.5<br>0.8 2.0<br>0.6 1.5<br>0.4 1.0<br>0.2 0.5<br>0.0 0.0<br>4 8 12 16 20 4 8 12 16 20<br>Number of Heads Number of Heads<br>(a) Lower Parameter Budget Comparison (b) Higher Parameter Budget Comparison<br>Accuracy Gain vs. 1 Head Accuracy Gain vs. 1 Head<br>**----- End of picture text -----**<br>


Figure 4: Comparison of performance when using different numbers of RAVAN heads at two different parameter budgets (SVHN: _N_ total = 1.2 M/2.4 M, 20 Newsgroups: _N_ total = 2.4 M/4.7 M). 

accuracy on three of the four datasets while never hurting performance by more than 0.4%. The improvement is largest when client updates are most diverse, specifically in CIFAR-100, as evidenced by Table 2. These lightweight scaling factors upweight useful heads and diminish the importance of heads whose fine-tuning subspaces provide less utility. Notably, we can achieve these performance gains without increasing the per-round communication cost or breaking exact aggregation. 

**Effect of Number of Heads.** From Section 2, we know that with a per-layer trainable parameter budget of _N_ , the effective rank of the RAVAN update is _√Nh_ . However, the maximum effective rank is still bounded by _d_ , where the pretrained weights **W** _∈_ R _[d][×][d]_ . Hence, adding heads improves the effective rank only while the value of _√Nh_ increases and remains below the dimension _d_ . Figure 4 confirms this behavior as accuracy generally only improves while _√Nh < d_ . In the lower parameter budget setting, this happens with a larger number of heads as _h_ can assume a larger value while still meeting this condition. In the higher parameter budget setting, we reach this saturation point much sooner. After _√Nh_ exceeds the value of _d_ , the effective rank of each individual head becomes smaller while the actual overall update does not become more expressive. This reduces representational power, especially at much larger values of _h_ , and weakens performance. Thus, optimally choosing the number of heads is a critical criterion for effective federated fine-tuning using RAVAN. 

Table 6: Accuracy comparison on GLUE benchmark with **LLaMA3.2-1B** . 

|Method|**MNLI-MM**|**MNLI-M**|**QNLI**|**QQP**|**SST-2**|**RTE**|**Average**|
|---|---|---|---|---|---|---|---|
|FedIT|84_._24|84_._62|82_._74|85_._96|94_._61|65_._70|82_._97|
|FedEx-LoRA|84_._15|84_._70|82_._74|86_._07|94_._61|65_._34|82_._94|
|FFA-LoRA|85_._05|**85**_._**78**|82_._07|84_._40|94_._38|62_._46|82_._36|
|Fed-SB|84_._88|85_._23|82_._84|84_._23|94_._95|**67**_._**15**|83_._21|
|RAVAN|**85**_._**24**|85_._65|**84**_._**00**|**86**_._**11**|**95**_._**18**|**67**_._**15**|**83**_._**90**|



**Scaling to Larger Model Architectures.** We demonstrate the scalability of RAVAN for larger model architectures by benchmarking the method against prior baselines on the GLUE benchmark [37] using **LLaMA3.2-1B** [13] (see Table 6). For each subtask, we use _C_ = 20 clients and follow the training procedure described in Section 4.2. We use a trainable parameter budget of _N_ total = 15 M, which corresponds to the rank configurations in the upper half of Tables 2 and 3. On average, RAVAN outperforms the next best baseline by 0.7% with the maximum gain on a single subtask being the 1.2% improvement over Fed-SB on the QNLI dataset. This demonstrates that RAVAN scales smoothly from 85 M to billion-parameter LLMs and complements state-of-the-art on-device models. 

## **5 Conclusion and Future Work** 

RAVAN offers a new avenue for performing FL fine-tuning in the presence of data and computational heterogeneity. By using multiple augmented LoRA heads and per-head scaling factors, RAVAN 

9 

improves the rank of the update approximation within a parameter budget, allowing the method to better approximate the full-parameter update without exceeding a device’s memory constraints. Partially freezing subsets of the heads allows clients to adaptively manage their own computational restrictions without sacrificing the accuracy of the aggregated model update. We believe that RAVAN and similar methods will open the doors for LLMs to capitalize on the vast amounts of edge data, a virtually untapped resource for model training and a critical future direction for the frontier of ML. 

While RAVAN outperforms existing PEFT benchmarks in FL, we identify three limitations and potential directions for improvement in the current method. First, while clients can fine-tune subsets of the heads to address device-level constraints and computational heterogeneity, the current framework necessitates that the same number of heads be selected in each layer of the original model. This reduces flexibility and can be remedied by a cross-layer scoring scheme that considers all RAVAN heads simultaneously. Second, RAVAN has yet to be tested in the context of differentially-private (DP) learning, and further study is required to validate its performance with stricter privacy guarantees. Finally, data-aware initializations of the **B** _i_ and **A** _i_ parameters may improve the performance of the method while retaining the current improvements in the rank of the update approximation. 

## **6 Acknowledgements** 

This work was supported in part by NSF grants CCF 2045694, CNS-2112471, CPS-2111751, SHF-2107024, ONR grant N00014-23-1-2149, a Google Research Scholar Award, the CyLab IoT Enterprise Security Initiative, and the CMU Prabhu and Poonam Goel fellowship. This work used Bridges-2 GPU at the Pittsburgh Supercomputing Center through allocation CIS250011 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296 [7]. We would like to thank Pranay Sharma, Siddharth Shah, Kevin Kuo, Aneesha Sampath, and Akash Dhasade for providing feedback and contributing to discussions for the project. 

## **References** 

- [1] Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina, Paul Whatmough, and Venkatesh Saligrama. Federated learning based on dynamic regularization. In _International Conference on Learning Representations_ , 2021. URL `https://openreview.net/forum? id=B7v4QMR6Z9w` . 

- [2] Paul Albert, Frederic Z. Zhang, Hemanth Saratchandran, Cristian Rodriguez-Opazo, Anton van den Hengel, and Ehsan Abbasnejad. RandloRA: Full rank parameter-efficient fine-tuning of large models. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL `https://openreview.net/forum?id=Hn5eoTunHN` . 

- [3] Sara Babakniya, Ahmed Roushdy Elkordy, Yahya H. Ezzeldin, Qingfeng Liu, Kee-Bong Song, Mostafa El-Khamy, and Salman Avestimehr. Slora: Federated parameter efficient fine-tuning of language models, 2023. URL `https://arxiv.org/abs/2308.06522` . 

- [4] Jiamu Bai, Daoyuan Chen, Bingchen Qian, Liuyi Yao, and Yaliang Li. Federated fine-tuning of large language models under heterogeneous tasks and client resources. In _The Thirtyeighth Annual Conference on Neural Information Processing Systems_ , 2024. URL `https: //openreview.net/forum?id=gkOzoHBXUw` . 

- [5] Klaudia Bałazy, Mohammadreza Banaei, Karl Aberer, and Jacek Tabor. Lora-xs: Low-rank adaptation with extremely small number of parameters, 2024. URL `https://arxiv.org/ abs/2405.17604` . 

- [6] Jieming Bian, Lei Wang, Letian Zhang, and Jie Xu. Lora-fair: Federated lora fine-tuning with aggregation and initialization refinement, 2025. URL `https://arxiv.org/abs/2411. 14961` . 

- [7] Timothy J. Boerner, Stephen Deems, Thomas R. Furlani, Shelley L. Knuth, and John Towns. Access: Advancing innovation: Nsf’s advanced cyberinfrastructure coordination ecosystem: Services & support. In _Practice and Experience in Advanced Research Computing 2023:_ 

10 

_Computing for the Common Good_ , PEARC ’23, page 173–176, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450399852. doi: 10.1145/3569951.3597559. URL `https://doi.org/10.1145/3569951.3597559` . 

- [8] Zheng Chai, Ahsan Ali, Syed Zawad, Stacey Truex, Ali Anwar, Nathalie Baracaldo, Yi Zhou, Heiko Ludwig, Feng Yan, and Yue Cheng. Tifl: A tier-based federated learning system. In _Proceedings of the 29th international symposium on high-performance parallel and distributed computing_ , pages 125–136, 2020. 

- [9] Yae Jee Cho, Luyang Liu, Zheng Xu, Aldi Fahrezi, and Gauri Joshi. Heterogeneous LoRA for federated fine-tuning of on-device foundation models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pages 12903–12913, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.717. URL `https://aclanthology.org/2024.emnlp-main.717/` . 

- [10] Enmao Diao, Jie Ding, and Vahid Tarokh. Hetero{fl}: Computation and communication efficient federated learning for heterogeneous clients. In _International Conference on Learning Representations_ , 2021. URL `https://openreview.net/forum?id=TNkPBBYFkXg` . 

- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2021. URL `https://openreview.net/forum?id=YicbFdNTTy` . 

- [12] Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Wei Shen, Limao Xiong, Yuhao Zhou, Xiao Wang, Zhiheng Xi, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang. LoRAMoE: Alleviating world knowledge forgetting in large language models via MoE-style plugin. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1932–1945, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.106. URL `https://aclanthology.org/2024. acl-long.106/` . 

- [13] Aaron Grattafiori et al. The llama 3 herd of models, 2024. URL `https://arxiv.org/abs/ 2407.21783` . 

- [14] Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eunsol Choi, and Danqi Chen. MRQA 2019 shared task: Evaluating generalization in reading comprehension. In Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eunsol Choi, and Danqi Chen, editors, _Proceedings of the 2nd Workshop on Machine Reading for Question Answering_ , pages 1–13, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-5801. URL `https://aclanthology.org/D19-5801/` . 

- [15] Pengxin Guo, Shuang Zeng, Yanran Wang, Huijie Fan, Feifei Wang, and Liangqiong Qu. Selective aggregation for low-rank adaptation in federated learning. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL `https://openreview.net/forum? id=iX3uESGdsO` . 

- [16] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2022. URL `https://openreview. net/forum?id=nZeVKeeFYf9` . 

- [17] Jiahui Hu, Dan Wang, Zhibo Wang, Xiaoyi Pang, Huiyu Xu, Ju Ren, and Kui Ren. Federated large language model: Solutions, challenges and future directions. _IEEE Wireless Communications_ , 2024. 

- [18] Nam Hyeon-Woo, Moon Ye-Bin, and Tae-Hyun Oh. Fedpara: Low-rank hadamard product for communication-efficient federated learning. In _International Conference on Learning Representations_ , 2022. URL `https://openreview.net/forum?id=d71n4ftoCBy` . 

11 

- [19] Ting Jiang, Shaohan Huang, Shengyue Luo, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. Mora: High-rank updating for parameter-efficient fine-tuning, 2024. URL `https://arxiv.org/abs/2405.12130` . 

- [20] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In _International conference on machine learning_ , pages 5132–5143. PMLR, 2020. 

- [21] Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M Asano. VeRA: Vector-based random matrix adaptation. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https://openreview.net/forum?id=NjNfLdxr3A` . 

- [22] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009. URL `https: //api.semanticscholar.org/CorpusID:18268744` . 

- [23] Qinbin Li, Bingsheng He, and Dawn Song. Model-contrastive federated learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10713–10722, 2021. 

- [24] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. _Proceedings of Machine learning and systems_ , 2:429–450, 2020. 

- [25] Yongle Li, Bo Liu, Sheng Huang, ZHeng ZHang, Xiaotong Yuan, and Richang Hong. Communication-efficient and personalized federated foundation model fine-tuning via tri-matrix adaptation, 2025. URL `https://arxiv.org/abs/2503.23869` . 

- [26] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation, 2024. URL `https://arxiv.org/abs/2402.09353` . 

- [27] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In _Artificial intelligence and statistics_ , pages 1273–1282. PMLR, 2017. 

- [28] Tom Mitchell. Twenty Newsgroups. UCI Machine Learning Repository, 1997. DOI: https://doi.org/10.24432/C5C323. 

- [29] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural images with unsupervised feature learning. In _NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011_ , 2011. URL `http://ufldl.stanford. edu/housenumbers/nips2011_housenumbers.pdf` . 

- [30] John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour, Mike Rabbat, Mani Malek, and Dzmitry Huba. Federated learning with buffered asynchronous aggregation. In _International conference on artificial intelligence and statistics_ , pages 3581–3607. PMLR, 2022. 

- [31] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ , 21(140):1–67, 2020. 

- [32] Raghav Singhal, Kaustubh Ponkshe, and Praneeth Vepakomma. Fedex-loRA: Exact aggregation for federated parameter-efficient fine-tuning of foundation models. In _NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability_ , 2024. URL `https: //openreview.net/forum?id=vOtEFcKIi5` . 

- [33] Raghav Singhal, Kaustubh Ponkshe, Rohit Vartak, Lav R Varshney, and Praneeth Vepakomma. Fed-sb: A silver bullet for extreme communication efficiency and performance in (private) federated lora fine-tuning. _arXiv preprint arXiv:2502.15436_ , 2025. 

- [34] Youbang Sun, Zitao Li, Yaliang Li, and Bolin Ding. Improving loRA in privacy-preserving federated learning. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https://openreview.net/forum?id=NLPzL6HWNl` . 

12 

- [35] Chunlin Tian, Zhan Shi, Zhijiang Guo, Li Li, and Cheng zhong Xu. HydraloRA: An asymmetric loRA architecture for efficient fine-tuning. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ , 2024. URL `https://openreview.net/forum?id= qEpi8uWX3N` . 

- [36] Van-Tuan Tran, Le Huy Khiem, and Viet Quoc Pham. Revisiting sparse mixture of experts for resource-adaptive federated fine-tuning foundation models. In _ICLR 2025 Workshop on Modularity for Collaborative, Decentralized, and Continual Deep Learning_ , 2025. URL `https://openreview.net/forum?id=IwNOUYgtuz` . 

- [37] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Tal Linzen, Grzegorz Chrupała, and Afra Alishahi, editors, _Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP_ , pages 353– 355, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-5446. URL `https://aclanthology.org/W18-5446/` . 

- [38] Ziyao Wang, Zheyu Shen, Yexiao He, Guoheng Sun, Hongyi Wang, Lingjuan Lyu, and Ang Li. FLoRA: Federated fine-tuning large language models with heterogeneous low-rank adaptations. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ , 2024. URL `https://openreview.net/forum?id=TcCorXxNJQ` . 

- [39] Yuxuan Yan, Qianqian Yang, Shunpu Tang, and Zhiguo Shi. Federa:efficient fine-tuning of language models in federated learning leveraging weight decomposition, 2024. URL `https: //arxiv.org/abs/2404.18848` . 

- [40] Yuhang Yao, Jianyi Zhang, Junda Wu, Chengkai Huang, Yu Xia, Tong Yu, Ruiyi Zhang, Sungchul Kim, Ryan Rossi, Ang Li, et al. Federated large language models: Current progress and future directions. _arXiv preprint arXiv:2409.15723_ , 2024. 

- [41] Jianyi Zhang, Saeed Vahidian, Martin Kuo, Chunyuan Li, Ruiyi Zhang, Tong Yu, Guoyin Wang, and Yiran Chen. Towards building the federatedgpt: Federated instruction tuning. In _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 6915–6919. IEEE, 2024. 

- [42] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In _The Eleventh International Conference on Learning Representations_ , 2023. URL `https://openreview. net/forum?id=lq62uWRJjiY` . 

13 

## **A Technical Appendices and Supplementary Material** 

## **A.1 Related Works** 

**Federated Learning.** Federated learning (FL) enables a distributed set of clients to collaboratively train a single global model using their local data [27]. The primary challenges in FL are data heterogeneity and computational heterogeneity induced by the statistical and hardware differences across clients, respectively. Several works aim to address these challenges by altering either the client training procedure or the server aggregation algorithm. FedProx [24], SCAFFOLD [20], and FedDyn [1] add a corrective regularization term to each client’s local objective—whether a proximal penalty - (FedProx), control variate correction (SCAFFOLD), or dynamic regularizer (FedDyn)—to keep local updates from drifting too far from the global model, a phenomena more often referred to as “client drift”. An alternative approach is to adjust the server aggregation scheme to reduce client drift. For example, FedVARP [49] incorporates historical client updates into the current round’s aggregation step to reduce the variance caused by partial client participation and heterogeneity. FedExP [50] varies the server learning rate by using an extrapolation rule that increases the server step size when consecutive aggregated updates point in similar directions and shrinks it when they diverge. 

**Parameter-Efficient Fine-Tuning.** Recently, many works have adopted the _pretrain-then-finetune_ framework, in which a general-purpose large language model (LLM) is adapted to a smaller downstream task [11, 31, 46, 48]. The excessive computational cost of fine-tuning LLMs has led to parameter-efficient fine-tuning (PEFT) methods that fine-tune a fraction of the overall parameters of the model. Adapter tuning [47], BitFit [43], and low-rank adaptation (LoRA) [16] have emerged as effective PEFT methods that can significantly reduce the number of parameters while maintaining task performance. Since its conception, many works have improved upon the initial LoRA formulation in various ways. Works such as QLoRA [45] and LoftQ [51] quantize the pretrained model weights to further improve the memory efficiency of LoRA-based fine-tuning. LoRA-XS [5], LoRA-SB [52], and MoRA [19] introduce an additional LoRA parameter while freezing _both_ the model backbone and the original LoRA parameters during fine-tuning. HydraLoRA [35], LoRAMoE [12], and MoLE [53] employ a mixture-of-experts architecture to traditional LoRA-based fine-tuning frameworks. 

**PEFT Methods for FL.** As newer applications look to use on-device data to fine-tune LLMs, PEFT methods for FL have become increasingly relevant. FedPETuning [54], FedPrompt [55], and FedIT [41] incorporate adapter tuning, prompt tuning, and LoRA into federated frameworks, respectively. More recently, methods like FFA-LoRA [34], FedEx-LoRA [32], Fed-SB [33], and RoLoRA [44] optimize LoRA in homogeneous-compute FL by addressing the inexactness in LoRA aggregation caused by separately averaging the **B** and **A** parameters. An orthogonal direction is explored by works such as HetLoRA [9], FlexLoRA [4], and FLoRA [38] that address computational heterogeneity in LoRA-based FL fine-tuning by allowing clients to train LoRA parameters with different ranks. However, an examination of cross-device fine-tuning remains limited in this context as methods like FLoRA have communication costs that scale linearly with the number of clients and communication rounds, making fine-tuning particularly difficult in large-scale FL systems. The goal of RAVAN is to design a PEFT method for FL that addresses data and computational heterogeneity, while scaling effectively to cross-device settings with a large number of clients and communication rounds. In this way, we overcome shortcomings in prior works and enable resource-aware fine-tuning. 

## **A.2 Broader Impacts** 

RAVAN enables accurate, efficient fine-tuning in a federated setting. While RAVAN has not yet been integrated into a real-world FL system, we identify two potential impacts of utilizing RAVAN: 

- **Edge Data for LLM Fine-Tuning:** RAVAN provides an opportunity for LLMs to utilize the data collected by edge devices. As edge applications become increasingly critical, capitalizing on these specialized datasets can make these applications more effective without forfeiting data locality. 

- **Efficiency and Improved Performance:** LLM fine-tuning is an expensive, energy-consuming procedure. RAVAN improves the efficiency of the process while retaining performance. Applied to realistic FL training regimes, RAVAN addresses some of these prior concerns. 

RAVAN should be implemented with safeguards to prevent misuse, privacy leaks, and harmful content. While RAVAN does not exacerbate these issues, they remain concerns in LLM usage more broadly. 

14 

## **A.3 Experimental Settings** 

**Hyperparameters and Optimization Details.** In this section, we highlight hyperparameter choices used in our experiments that were not discussed in Section 4. These descriptions, in addition to the provided code, should aid in the reproducibility of the stated results. 

Table 7: FL hyperparameter settings used for each model–dataset pair. 

||**Model/Dataset**<br>**ViT-B-16/CIFAR-100**<br>**ViT-B-16/SVHN**<br>**T5-Base/20 Newsgroups**<br>**T5-Base/MRQA**<br>**LLaMA3.2-1B/GLUE**|
|---|---|
|Batch Size<br>Max Sequence Length<br>Local Iterations<br>Communication Rounds<br>Total Epochs (per round)|32<br>32<br>32<br>32<br>16<br>–<br>–<br>256<br>256<br>128<br>50<br>50<br>50<br>50<br>50<br>50<br>50<br>100<br>20<br>20<br>1<br>1<br>1<br>1<br>1|



For RAVAN and each baseline, we run a learning rate hyperparameter sweep across the values _{_ 5e _−_ 5 _,_ 1e _−_ 5 _,_ 5e _−_ 4 _,_ 1e _−_ 4 _,_ 5e _−_ 3 _,_ 1e _−_ 3 _,_ 5e _−_ 2 _,_ 1e _−_ 2 _,_ 5e _−_ 2 _}_ and choose the most performant learning to represent in our results. Table 8 represents the optimal choices for each baseline in all settings. The following results each use the ADAM optimizer with momentum set to 0 _._ 9. 

Table 8: Optimal learning rate configurations for all baselines. 

(a) Lower parameter budget / I.I.D. clients. 

|Method|**Model/Dataset**<br>**ViT-B-16/CIFAR-100**<br>**ViT-B-16/SVHN**<br>**T5-Base/20 Newsgroups**<br>**T5-Base/MRQA**<br>**LLaMA3.2-1B/GLUE**|**Model/Dataset**<br>**ViT-B-16/CIFAR-100**<br>**ViT-B-16/SVHN**<br>**T5-Base/20 Newsgroups**<br>**T5-Base/MRQA**<br>**LLaMA3.2-1B/GLUE**|
|---|---|---|
|FedIT<br>FedEx-LoRA<br>FFA-LoRA<br>Fed-SB<br>RAVAN|5_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_4<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_4<br>1_×_10_−_2<br>1_×_10_−_2<br>1_×_10_−_2<br>5_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>5_×_10_−_3<br>5_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_4<br>5_×_10_−_4<br>5_×_10_−_4<br>5_×_10_−_4<br>1_×_10_−_4<br>5_×_10_−_5||
||(b) Higher parameter budget / I.I.D. clients.||
|Method|**Model/Dataset**<br>**ViT-B-16/CIFAR-100**<br>**ViT-B-16/SVHN**<br>**T5-Base/20 Newsgroups**<br>**T5-Base/MRQA**<br>**LLaMA3.2-1B/GLUE**||
|FedIT<br>FedEx-LoRA<br>FFA-LoRA<br>Fed-SB<br>RAVAN||5_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>5_×_10_−_4<br>1_×_10_−_4<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>5_×_10_−_4<br>1_×_10_−_4<br>1_×_10_−_2<br>1_×_10_−_2<br>1_×_10_−_2<br>1_×_10_−_2<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>5_×_10_−_3<br>5_×_10_−_4<br>1_×_10_−_4<br>5_×_10_−_4<br>5_×_10_−_4<br>1_×_10_−_4<br>1_×_10_−_4<br>5_×_10_−_5|
|||(c) Lower parameter budget / non-I.I.D. clients.|
|Method||**Model/Dataset**<br>**ViT-B-16/CIFAR-100**<br>**ViT-B-16/SVHN**<br>**T5-Base/20 Newsgroups**<br>**T5-Base/MRQA**|
|FedIT<br>FedEx-LoRA<br>FFA-LoRA<br>Fed-SB<br>RAVAN||5_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_2<br>1_×_10_−_2<br>1_×_10_−_2<br>5_×_10_−_3<br>5_×_10_−_4<br>5_×_10_−_3<br>1_×_10_−_3<br>5_×_10_−_4<br>5_×_10_−_4<br>5_×_10_−_4<br>5_×_10_−_4<br>1_×_10_−_4|
|||(d) Higher parameter budget / non-I.I.D. clients.|
|Method||**Model/Dataset**<br>**ViT-B-16/CIFAR-100**<br>**ViT-B-16/SVHN**<br>**T5-Base/20 Newsgroups**<br>**T5-Base/MRQA**|
|FedIT<br>FedEx-LoRA<br>FFA-LoRA<br>Fed-SB<br>RAVAN||5_×_10_−_3<br>1_×_10_−_3<br>1_×_10_−_3<br>5_×_10_−_4<br>1_×_10_−_3<br>5_×_10_−_4<br>1_×_10_−_3<br>5_×_10_−_4<br>1_×_10_−_2<br>1_×_10_−_2<br>1_×_10_−_2<br>5_×_10_−_3<br>5_×_10_−_4<br>1_×_10_−_3<br>1_×_10_−_3<br>5_×_10_−_4<br>5_×_10_−_4<br>5_×_10_−_4<br>1_×_10_−_4<br>1_×_10_−_4|



15 

**Baseline Descriptions.** We provide details for each of the baselines used in our experiments. We highlight how each baseline initializes, trains, and communicates the individual LoRA parameters. 

- **FedIT:** FedIT initializes the LoRA parameters with **B**[(0)] = **0** and **A**[(0)] _∼N_ (0 _, σ_[2] ). In communication round _t_ , each client _c ∈C_[(] _[t]_[)] locally trains the LoRA parameters resulting in local parameters **B** _c_[(] _[t]_[)] _[,]_ **[A]**[(] _c[t]_[)][.][After communicating these parameters back to the central server, the server performs] the following aggregation to generate the new global model: 

**==> picture [313 x 28] intentionally omitted <==**

- **FedEx-LoRA:** FedEx-LoRA initializes the LoRA parameters with **B**[(0)] = **0** and **A**[(0)] _∼N_ (0 _, σ_[2] ). In communication round _t_ , each client _c ∈C_[(] _[t]_[)] locally trains the LoRA parameters resulting in local parameters **B**[(] _c[t]_[)] _[,]_ **[A]**[(] _c[t]_[)][.][To address the exact aggregation issue, the server updates both the] global LoRA parameters as well as the model backbone: 

**==> picture [371 x 69] intentionally omitted <==**

While this ensures exact updates in every round, the updated model backbone **W**[(] _[t]_[+1)] also has to be communicated from the central server, increasing the communication overhead of the procedure. 

- **FFA-LoRA:** FFA-LoRA initializes the LoRA parameters with **B**[(0)] = **0** and **A** _∼N_ (0 _, σ_[2] ). However, the **A** parameter remains frozen at initialization and is never locally trained by the clients and communicated throughout the procedure. Thus, the only update throughout training is the update to the LoRA **B** parameter: 

**==> picture [250 x 27] intentionally omitted <==**

- **Fed-SB:** Fed-SB uses three LoRA parameters which, for the sake of consistency with prior notation, we call **B** _∈_ R _[d][×][r]_ , **H** _∈_ R _[r][×][r]_ , **A** _∈_ R _[r][×][d]_ . The weight update is reparameterized as **BHA** . To initialize the LoRA parameters, Fed-SB performs an initial round of full-parameter fine-tuning to obtain a full-parameter weight update ∆ **W** full. The weight update is decomposed using SVD to get ∆ **W** full = **UΣV** _[⊤]_ . **B** , **H** , and **A** are then initialized as **B** = **U** [: _,_ 1 : _r_ ], **H**[(0)] = **Σ** [1 : _r,_ 1 : _r_ ], **A** = **V** _[⊤]_ [1 : _r,_ :]. **B** and **A** are frozen at initialization, so the only update is the following: 

**==> picture [251 x 28] intentionally omitted <==**

- **FlexLoRA:** FlexLoRA allows each client _c ∈C_[(] _[t]_[)] to train LoRA parameters with client-specific ranks _rc_ . To aggregate the LoRA parameters, the server performs SVD on _|C_[(] 1 _[t]_[)] _|_ � _c∈C_[(] _[t]_[)] **[ B]**[(] _c[t]_[)] **[A]**[(] _c[t]_[)] = **UΣV** _[⊤]_ . To redistribute the LoRA parameters back to the clients, the central server sends each client _c ∈C_[(] _[t]_[+1)] the following LoRA parameters: 

**==> picture [329 x 13] intentionally omitted <==**

- **HetLoRA:** Let _r_ max be the highest rank supported by any client. Each client pads its local parameters to this common shape (with zeros in the unused columns and rows) before upload, so aggregation is still dimensionally consistent. The server weights the individual LoRA parameters based on their relative Frobenius norms: 

**==> picture [304 x 58] intentionally omitted <==**

The server then truncates the new global LoRA parameters **B**[(] _[t]_[+1)] and **A**[(] _[t]_[+1)] for each client _c ∈C_[(] _[t]_[+1)] so that **B**[(] _c[t]_[+1)] = **B**[(] _[t]_[+1)] [: _,_ 1 : _rc_ ] and **A**[(] _c[t]_[+1)] = **A**[(] _[t]_[+1)] [1 : _rc,_ :]. 

16 

**==> picture [318 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
Trainable Parameter Distributions<br>0.5<br>Bell-Shaped<br>Uniform<br>0.4<br>Skewed-Right<br>0.3<br>0.2<br>0.1<br>0.0<br>1 1 3 1 1 1 3 1 1 1 3 1<br>4 2 4 4 2 4 4 2 4<br>Fraction of Maximum Trainable Parameters<br>Proportion of Clients<br>**----- End of picture text -----**<br>


Figure 5: Fraction of clients assigned to each trainable parameter budget in each distribution. Skewed-left is omitted because it is never used in our experiments. 

**Computational Heterogeneity Setup.** To emulate computational heterogeneity in our FL setup, we vary the number of trainable parameters at each client. Let _N_ max denote the largest number of trainable parameters that any client can afford. Each client _c_ is constrained to a trainable parameter budget _Nc ∈{_ 4[1] _[,]_ 2[1] _[,]_[3] 4 _[,]_[ 1] _[}][N]_[max][.][This value is held constant throughout the FL procedure to mirror] fixed hardware limits. _Nc_ simply determines the trainable parameter budget per-client; all other hyperparameters are identical to the compute-homogeneous experiments. The bar plot in Figure 5 provides a visual summary of the client mix. Uniform serves as a neutral baseline where there is an equal proportion of clients at every trainable parameter budget; bell-shaped concentrates clients - around medium ranks, reflecting the case where most devices have moderate capability; skewed right stresses the system by placing a large share of the population at the lowest rank, leaving only a small fraction of high-capacity contributors. HetLoRA, FlexLoRA, and RAVAN each accommodate clients with different trainable-parameter budgets in distinct ways: 

- **HetLoRA and FlexLoRA:** The LoRA rank for each client _c_ is _rc_ = �( _Nc / N_ max) _· r_ max� where _r_ max is the maximum rank trained by any client. Since the number of trainable parameters scales linearly with the rank, this scaling ensures that every client keeps its update within the allotted budget _Nc_ while allowing higher-capacity devices to contribute proportionally higher-rank updates. 

- **RAVAN** : RAVAN uses _H_ LoRA heads per weight matrix. A client with budget _Nc_ , fine-tunes only �( _Nc / N_ max) _· H_ � heads and leaves the remaining heads frozen (e.g. for _Nc_ =[1] 4 _[N]_[max][the client] trains one quarter of the heads). 

Table 9: Layers equipped with LoRA adapters in each model backbone. 

|**Model**|**LoRA Target Modules**|
|---|---|
|ViT-B-16|`query`, `value`|
|T5-Base|`SelfAttention.q`, `SelfAttention.v`|
|LLaMA3.2-1B|`q_proj`, `v_proj`|



**LoRA Implementation Details.** For every model backbone, we insert LoRA adapters only in the self-attention projection matrices. The exact parameters for which we apply LoRA are described in Table 9. All other parameters are frozen and do not have associated LoRA parameters. 

**Compute Details and Cluster Description.** All experiments were executed on a GPU cluster managed by SLURM. Each training job used a single NVIDIA V100 32GB GPU with 256 GB RAM. Our environment used Pytorch 2.5.1 and Huggingface 4.47.1 for all experiments. With this setup, each experimental run took _≈_ 1 GPU hour with ViT-B-16 for both CIFAR-100 and SVHN, _≈_ 2 GPU hours with T5-Base for 20 Newsgroups, _≈_ 3 GPU hours with T5-Base for MRQA, and _≈_ 2 GPU hours with LLaMA3.2-1B for each GLUE subtask. All baselines were trained with identical hardware, batch sizes, optimizers, and communication rounds to ensure fair comparison. 

17 

## **A.4 Additional Experiments** 

**==> picture [159 x 252] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.95 Accuracy with Computational Heterogeneity<br>0.90 HetLoRAFlexLoRA RAVAN RAVAN (Weight)(Random) RAVAN (Gradient)<br>0.85<br>0.80<br>0.75<br>0.70<br>0.65<br>0.60<br>0.55<br>(a) CIFAR-100 / lower parameter budget<br>0.95 Accuracy with Computational Heterogeneityy with Computational Heterogeneity with Computational Heterogeneityputational Heterogeneityutational Heterogeneitygeneityeneityy<br>0.90<br>0.85<br>0.80<br>0.75<br>0.70<br>0.65<br>0.60<br>0.55<br>(c) SVHN / lower parameter budget<br>Bell-Shaped Uniform Skewed Right<br>Bell-Shaped Uniform Skewed Right<br>Accuracy<br>Accuracy<br>**----- End of picture text -----**<br>


**==> picture [159 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.9 Accuracy with Computational Heterogeneity<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>(b) CIFAR-100 / higher parameter budget<br>Bell-Shaped Uniform Skewed Right<br>Accuracy<br>**----- End of picture text -----**<br>


**==> picture [340 x 385] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.95 Accuracy with Computational Heterogeneityy with Computational Heterogeneity with Computational Heterogeneityputational Heterogeneityutational Heterogeneitygeneityeneityy 0.95 Accuracy with Computational Heterogeneity<br>0.90 0.90<br>0.85 0.85<br>0.80 0.80<br>0.75 0.75<br>0.70 0.70<br>0.65 0.65<br>0.60 0.60<br>0.55 0.55<br>(c) SVHN / lower parameter budget (d) SVHN / higher parameter budget<br>0.70 Accuracy with Computational Heterogeneity 0.70 Accuracy with Computational Heterogeneity<br>0.65 0.65<br>0.60 0.60<br>0.55 0.55<br>0.50 0.50<br>0.45 0.45<br>0.40 0.40<br>(e) 20 Newsgroups / lower parameter budget (f) 20 Newsgroups / higher parameter budget<br>0.610 F1 with Computational Heterogeneity 0.62 F1 with Computational Heterogeneity<br>0.605 0.61<br>0.600 0.60<br>0.595 0.59<br>0.590 0.58<br>(g) MRQA / lower parameter budget (h) MRQA / higher parameter budget<br>Bell-Shaped Uniform Skewed Right Bell-Shaped Uniform Skewed Right<br>Bell-Shaped Uniform Skewed Right Bell-Shaped Uniform Skewed Right<br>Bell-Shaped Uniform Skewed Right Bell-Shaped Uniform Skewed Right<br>Accuracy Accuracy<br>Accuracy Accuracy<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 6: Impact of computational heterogeneity on baselines and RAVAN across four datasets. Each row shows a single dataset (left: lower parameter budget, right: higher parameter budget). All settings match the descriptions from Section 4. 

**Computational Heterogeneity Experiments.** We evaluate all methods with 20 non-I.I.D. clients whose usable parameter budgets are drawn from three distributions. Across vision and language tasks, RAVAN’s variants consistently outperform the competing LoRA baselines. These results underscore RAVAN’s robustness to computational heterogeneity across different tasks and parameter budgets. 

18 

Table 10: LoRA ranks under lower vs. higher parameter budgets. 

|**Method**|**Lower Budget**|**Higher Budget**|
|---|---|---|
|FedIT|32|64|
|FedEx-LoRA|32|64|
|FFA-LoRA|64|128|
|Fed–SB|221|313|
|RAVAN|110|156|



Table 11: Accuracy comparison on GLUE benchmark with LLaMA3.2-1B. 

(a) 20 clients / lower parameter budget 

|Method|**MNLI-MM**|**MNLI-M**|**QNLI**|**QQP**|**SST-2**|**RTE**|**Average**|
|---|---|---|---|---|---|---|---|
|FedIT|84_._24|84_._62|82_._74|85_._96|94_._61|65_._70|82_._97|
|FedEx-LoRA|84_._15|84_._70|82_._74|86_._07|94_._61|65_._34|82_._94|
|FFA-LoRA|85_._05|**85**_._**78**|82_._07|84_._40|94_._38|62_._46|82_._36|
|Fed-SB|84_._88|85_._23|82_._84|84_._23|94_._95|**67**_._**15**|83_._21|
|RAVAN|**85**_._**24**|85_._65|**84**_._**00**|**86**_._**11**|**95**_._**18**|**67**_._**15**|**83**_._**90**|



(b) 20 clients / higher parameter budget 

|Method|**MNLI-MM**|**MNLI-M**|**QNLI**|**QQP**|**SST-2**|**RTE**|**Average**|
|---|---|---|---|---|---|---|---|
|FedIT|83_._74|83_._24|87_._72|85_._60|95_._30|68_._95|84_._09|
|FedEx-LoRA|83_._95|83_._41|87_._79|85_._65|**95**_._**41**|**70**_._**04**|84_._38|
|FFA-LoRA|85_._27|84_._69|89_._51|87_._10|95_._18|68_._23|85_._00|
|Fed-SB|85_._85|84_._76|89_._53|86_._09|94_._95|66_._79|84_._66|
|RAVAN|**86**_._**20**|**85**_._**34**|**90**_._**35**|**87**_._**22**|95_._18|**70**_._**04**|**85**_._**72**|



(c) 50 clients / lower parameter budget 

|Method|**MNLI-MM**|**MNLI-M**|**QNLI**|**QQP**|**SST-2**|**RTE**|**Average**|
|---|---|---|---|---|---|---|---|
|FedIT|84_._22|84_._24|87_._53|85_._87|94_._61|61_._73|83_._03|
|FedEx-LoRA|84_._25|84_._15|87_._77|85_._81|94_._61|62_._09|83_._11|
|FFA-LoRA|85_._92|85_._05|**89**_._**33**|**87**_._**40**|95_._30|60_._29|83_._88|
|Fed-SB|85_._71|84_._65|88_._05|86_._08|94_._15|**64**_._**98**|83_._94|
|RAVAN|**86**_._**03**|**85**_._**53**|88_._91|86_._95|**95**_._**41**|62_._09|**84**_._**15**|



## (d) 50 clients / higher parameter budget 

|Method|**MNLI-MM**|**MNLI-M**|**QNLI**|**QQP**|**SST-2**|**RTE**|**Average**|
|---|---|---|---|---|---|---|---|
|FedIT|84_._66|84_._26|88_._38|85_._87|95_._18|63_._58|83_._66|
|FedEx-LoRA|84_._74|84_._02|88_._50|85_._82|95_._41|58_._12|82_._77|
|FFA-LoRA|85_._35|84_._64|87_._21|87_._20|94_._50|61_._37|83_._38|
|Fed-SB|85_._91|85_._24|87_._68|86_._30|93_._58|**67**_._**15**|84_._31|
|RAVAN|**86**_._**17**|**85**_._**35**|**88**_._**87**|**87**_._**39**|**95**_._**87**|64_._62|**84**_._**71**|



**LLaMA Experiments.** In these experiments, we use the same hyperparameter settings described in Section 4 but vary the number of total clients and the the ranks of each baseline. Across all four GLUE configurations, RAVAN consistently matches or exceeds the performance of the strongest PEFT baselines using LLaMA3.2-1B (see Table 11). Additionally, while other PEFT baselines vary in performance across settings, RAVAN’s performance remains consistent in all configurations. This suggests that RAVAN maintains robust performance, demonstrating its ability to scale effectively to larger models and diverse FL scenarios. 

19 

**==> picture [171 x 262] intentionally omitted <==**

**----- Start of picture text -----**<br>
FL Training Curves<br>0.8<br>0.6<br>0.4<br>FedEx-LoRA<br>RAVAN<br>0.2 Fed-SB<br>FedIT<br>FFA-LoRA<br>0.0<br>0 10 20 30 40 50<br>Communication Rounds<br>(a) CIFAR-100 / 20 clients / lower parameter<br>budget<br>FL Training Curves<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0 10 20 30 40 50<br>Communication Rounds<br>Accuracy<br>Accuracy<br>**----- End of picture text -----**<br>


- (c) CIFAR-100 / 20 clients / higher parameter budget 

**==> picture [171 x 113] intentionally omitted <==**

**----- Start of picture text -----**<br>
FL Training Curves<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0 10 20 30 40 50<br>Communication Rounds<br>Accuracy<br>**----- End of picture text -----**<br>


- (b) CIFAR-100 / 50 clients / lower parameter budget 

**==> picture [171 x 114] intentionally omitted <==**

**----- Start of picture text -----**<br>
FL Training Curves<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0 10 20 30 40 50<br>Communication Rounds<br>Accuracy<br>**----- End of picture text -----**<br>


- (d) CIFAR-100 / 50 clients / higher parameter budget 

**==> picture [364 x 252] intentionally omitted <==**

**----- Start of picture text -----**<br>
FL Training Curves FL Training Curves<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0 10 20 30 40 50 0 10 20 30 40 50<br>Communication Rounds Communication Rounds<br>(e) SVHN / 20 clients / lower parameter budget (f) SVHN / 50 clients / lower parameter budget<br>FL Training Curves FL Training Curves<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0 10 20 30 40 50 0 10 20 30 40 50<br>Communication Rounds Communication Rounds<br>Accuracy Accuracy<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


- (g) SVHN / 20 clients / higher parameter budget 

- (h) SVHN / 50 clients / higher parameter budget 

Figure 7: FL training curves for CIFAR-100 and SVHN for all benchmarks. 

**Training Curves.** Figure 7 displays the training curves for the various PEFT benchmarks on CIFAR100 and SVHN using a varying number of I.I.D. clients and trainable parameter budgets. In comparison to the other PEFT methods, RAVAN converges faster and to a better overall performance, suggesting that it requires fewer communication rounds to reach optimal performance. 

20 

## **Appendix References** 

- [43] Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. BitFit: Simple parameter-efficient finetuning for transformer-based masked language-models. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 1–9, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-short.1. URL `https://aclanthology.org/2022.acl-short.1/` . 

- [44] Shuangyi Chen, Yue Ju, Hardik Dalal, Zhongwen Zhu, and Ashish J Khisti. Robust federated finetuning of foundation models via alternating minimization of loRA. In _Workshop on Efficient Systems for Foundation Models II @ ICML2024_ , 2024. URL `https://openreview.net/ forum?id=xT0acYbgOF` . 

- [45] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL `https://openreview.net/forum?id=OUIFPHEgJU` . 

- [46] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL `https://aclanthology. org/N19-1423/` . 

- [47] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, _Proceedings of the 36th International Conference on Machine Learning_ , volume 97 of _Proceedings of Machine Learning Research_ , pages 2790–2799. PMLR, 09–15 Jun 2019. URL `https://proceedings.mlr. press/v97/houlsby19a.html` . 

- [48] Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. In Iryna Gurevych and Yusuke Miyao, editors, _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 328–339, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1031. URL `https://aclanthology.org/P18-1031/` . 

- [49] Divyansh Jhunjhunwala, Pranay Sharma, Aushim Nagarkatti, and Gauri Joshi. Fedvarp: Tackling the variance due to partial client participation in federated learning. In _Uncertainty in Artificial Intelligence_ , pages 906–916. PMLR, 2022. 

- [50] Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi. Fedexp: Speeding up federated averaging via extrapolation. In _The Eleventh International Conference on Learning Representations_ , 2023. URL `https://openreview.net/forum?id=IPrzNbddXV` . 

- [51] Yixiao Li, Yifan Yu, Chen Liang, Nikos Karampatziakis, Pengcheng He, Weizhu Chen, and Tuo Zhao. Loftq: LoRA-fine-tuning-aware quantization for large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https://openreview. net/forum?id=LzPWWPAdY4` . 

- [52] Kaustubh Ponkshe, Raghav Singhal, Eduard Gorbunov, Alexey Tumanov, Samuel Horvath, and Praneeth Vepakomma. Initialization using update approximation is a silver bullet for extremely efficient low-rank fine-tuning, 2025. URL `https://arxiv.org/abs/2411.19557` . 

- [53] Xun Wu, Shaohan Huang, and Furu Wei. Mixture of loRA experts. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https://openreview.net/forum? id=uWvKBCYh4S` . 

21 

- [54] Zhuo Zhang, Yuanhang Yang, Yong Dai, Qifan Wang, Yue Yu, Lizhen Qu, and Zenglin Xu. FedPETuning: When federated learning meets the parameter-efficient tuning methods of pretrained language models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 9963–9977, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-acl.632. URL `https://aclanthology.org/2023.findings-acl.632/` . 

- [55] Haodong Zhao, Wei Du, Fangqi Li, Peixuan Li, and Gongshen Liu. Fedprompt: Communicationefficient and privacy-preserving prompt tuning in federated learning. In _ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 1–5, 2023. doi: 10.1109/ICASSP49357.2023.10095356. 

22 

## **NeurIPS Paper Checklist** 

## 1. **Claims** 

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: We justify the contributions of our method in Sections 3 and 4 which discuss the proposed method and experiments using the proposed method, respectively. Guidelines: 

- The answer NA means that the abstract and introduction do not include the claims made in the paper. 

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

## 2. **Limitations** 

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] 

Justification: The conclusion in Section 5 mentions three limitations of our work and potential directions for future research. 

Guidelines: 

   - The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. 

   - The authors are encouraged to create a separate "Limitations" section in their paper. 

   - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. 

   - The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. 

   - The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. 

   - The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. 

   - If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. 

   - While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations. 

3. **Theory assumptions and proofs** 

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

23 

Answer: [NA] 

Justification: Our work is empirical and therefore does not require proofs of key theoretical findings. 

Guidelines: 

- The answer NA means that the paper does not include theoretical results. 

- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced. 

- All assumptions should be clearly stated or referenced in the statement of any theorems. 

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. 

- Theorems and Lemmas that the proof relies upon should be properly referenced. 

## 4. **Experimental result reproducibility** 

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? 

Answer: [Yes] 

Justification: Section 4 and the Appendix provide all necessary hyperparameters and configurations necessary to reproduce the experiments listed in the results. We highlight the most important values used in our setting in Section 4. Guidelines: 

- The answer NA means that the paper does not include experiments. 

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. 

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. 

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. 

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example 

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. 

- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. 

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). 

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 

24 

## 5. **Open access to data and code** 

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

Answer: [Yes] 

Justification: Our code base is provided in the supplementary material zip file with a README that includes instructions on run commands for our method and all baselines. 

Guidelines: 

- The answer NA means that paper does not include experiments requiring code. 

- Please see the NeurIPS code and data submission guidelines ( `https://nips.cc/ public/guides/CodeSubmissionPolicy` ) for more details. 

- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). 

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( `https: //nips.cc/public/guides/CodeSubmissionPolicy` ) for more details. 

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. 

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. 

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). 

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 

## 6. **Experimental setting/details** 

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: Section 4 and the Appendix provide all necessary hyperparameters and configurations necessary to reproduce the experiments listed in the results. We highlight the most important values used in our setting in Section 4. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. 

- The full details can be provided either with the code, in appendix, or as supplemental material. 

## 7. **Experiment statistical significance** 

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [Yes] 

Justification: All results are reported as the average across runs with 3 random seeds. For the sake of horizontal space we don’t include the values directly in the tables, but provide standard deviation numbers in the Appendix. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

25 

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. 

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). 

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) 

- The assumptions made should be given (e.g., Normally distributed errors). 

- It should be clear whether the error bar is the standard deviation or the standard error of the mean. 

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. 

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). 

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 

## 8. **Experiments compute resources** 

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: The Appendix contains details on the resources required to run each experiment. All experiments were run using a single V100 GPU. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. 

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 

## 9. **Code of ethics** 

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics `https://neurips.cc/public/EthicsGuidelines` ? 

Answer: [Yes] 

Justification: We reviewed the code of ethics and ensured that our research followed these guidelines. We avoid privacy violations, have documented the research thoroughly, and all involved parties have been fairly compensated. Guidelines: 

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. 

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 

## 10. **Broader impacts** 

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

26 

Justification: Our paper highlights the potential impacts in this ongoing field of research in Sections 1 and 5. These sections highlight the bigger picture of our work and place it in the context of the larger field of research. 

Guidelines: 

- The answer NA means that there is no societal impact of the work performed. 

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. 

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. 

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. 

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. 

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 

## 11. **Safeguards** 

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [NA] 

Justification: Our work uses general-purpose models and datasets that have been adequately cited and referenced. We have avoid privacy violations and other security risks. 

Guidelines: 

- The answer NA means that the paper poses no such risks. 

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. 

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 

## 12. **Licenses for existing assets** 

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: All original material is created with the consent of the authors. Any other material has been cited and referenced. 

Guidelines: 

- The answer NA means that the paper does not use existing assets. 

27 

- The authors should cite the original paper that produced the code package or dataset. 

- The authors should state which version of the asset is used and, if possible, include a URL. 

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. 

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, `paperswithcode.com/datasets` has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. 

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. 

- If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 

## 13. **New assets** 

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [NA] 

Justification: We release no new models/datasets in this project. 

Guidelines: 

- The answer NA means that the paper does not release new assets. 

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 

- The paper should discuss whether and how consent was obtained from people whose asset is used. 

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 

## 14. **Crowdsourcing and research with human subjects** 

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [NA] 

Justification: We do not include human subjects in any experiments. 

Guidelines: 

   - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

   - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 

   - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 

15. **Institutional review board (IRB) approvals or equivalent for research with human subjects** 

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [NA] 

Justification: We do not use human subjects in any experiments. 

28 

Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 

## 16. **Declaration of LLM usage** 

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. 

Answer: [Yes] 

Justification: LLM usage was declared in the initial submission. We used LLMs for grammar edits and to generate the icon in the intro line. 

Guidelines: 

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. 

- Please refer to our LLM policy ( `https://neurips.cc/Conferences/2025/LLM` ) for what should or should not be described. 

29 

