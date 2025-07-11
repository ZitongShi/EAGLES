# EAGLES
> Zitong Shi, Guancheng Wan, Wenke Huang, Guibin Zhang, He Li, Carl Yang, Mang Ye

## Abstarct

Federated graph learning has swiftly gained attention as a privacy-preserving approach to collaborative learning graphs. However, due to the message aggregation mechanism of Graph Neural Networks (GNNs), computational demands increase substantially as datasets grow and GNN layers deepen for clients. Previous efforts to reduce computational resource demands have focused solely on the parameter space, overlooking the overhead introduced by graph data. To address this, we propose $\textbf{EAGLES}$, a unified sparsification framework.  Our approach applies client-consensus parameter sparsification to produce multiple unbiased sub-networks at varying sparsity levels, avoiding iterative adjustments and mitigating performance degradation seen in prior methods. Within the graph structure domain, previous graph sparsification efforts are based on single sparsification strategies and fail to address heterogeneity. To tackle these issues, we introduce two experts: graph sparsification expert and graph synergy expert. The former applies node-level sparsification using multiple criteria, while the latter learns node contextual information to merge them into an optimal sparse subgraph. This contextual information is also utilized in a novel distance metric to measure structural similarity between clients, encouraging knowledge sharing among similar clients. We introduce the $\textbf{Harmony Sparsification Principle}$ to guide this process, effectively balancing model performance with lightweight graph and model structures. Extensive experiments across multiple metrics and datasets verify the superiority of our method. For instance, it achieves competitive performance on ogbn-proteins while reducing training FLOPS by 82\% $\downarrow$ and communication bytes by 80\% $\downarrow$ .

## Citation
```
@inproceedings{
shi2025eagles,
title={{EAGLES}: Towards Effective, Efficient, and Economical Federated Graph Learning via Unified Sparsification},
author={Zitong Shi and Guancheng Wan and Wenke Huang and Guibin Zhang and He Li and Carl Yang and Mang Ye},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=Bd9JlrqZhN}
}
```
