TP - A imagem da inspeção está na base de referência e o sistema acertou exatamente qual é ela.
FN - A imagem da inspeção está na base de referência, mas o sistema errou ou não conseguiu identificar.
FP - A imagem da inspeção não está na base de referência, mas o sistema disse que ela existe e até apontou uma semelhante.
TN - A imagem da inspeção não está na base, e o sistema acertadamente reconheceu que ela não existe lá.

| **Métrica**           | **Significado Prático**                                                               | **Fórmula**                                     | **Quando Usar**                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Precision**         | Das vezes que o sistema disse que reconheceu uma imagem, quantas ele acertou?         | TP / (TP + FP)                                  | Útil quando **erros do tipo falso positivo** (identificar algo inexistente) são mais graves.          |
| **Recall**            | Das imagens que realmente estavam na base, quantas o sistema reconheceu corretamente? | TP / (TP + FN)                                  | Importante quando **não deixar escapar** imagens verdadeiras é mais importante (FN é crítico).        |
| **F1 Score**          | Combinação equilibrada entre precisão e recall                                        | 2 × (Precision × Recall) / (Precision + Recall) | Ideal quando há **desequilíbrio entre classes** e você precisa equilibrar precisão e recall.          |
| **Accuracy**          | Porcentagem total de acertos do sistema (tanto positivos quanto negativos)            | (TP + TN) / (TP + FP + FN + TN)                 | Boa métrica quando as **classes estão balanceadas** e **todos os tipos de erro têm o mesmo peso**.    |
| **Specificity**       | Entre os casos em que a imagem não existia, quantos o sistema acertou?                | TN / (TN + FP)                                  | Importante quando **evitar falsos positivos** é essencial, como em **sistemas de segurança**.         |
| **FPR**               | Frequência de erros ao reconhecer algo que não estava na base                         | FP / (FP + TN)                                  | Útil para medir a **taxa de alarme falso**, especialmente em contextos com muitas imagens negativas.  |
| **FNR**               | Frequência de falhas ao não reconhecer imagens que realmente estavam na base          | FN / (FN + TP)                                  | Crítica quando é **inaceitável perder imagens conhecidas**, como em **sistemas de inspeção crítica**. |
| **Balanced Accuracy** | Média entre a taxa de acerto dos positivos (Recall) e dos negativos (Specificity)     | (Recall + Specificity) / 2                      | Ideal em **cenários desbalanceados**, pois considera ambos os lados com igual importância.            |




Set(42|1) limite local 6|12
### 📊 Métricas de Avaliação — Dataset Flowers102 (Num Features = 60, Distance = 0.9, Threshold = 0.5)

| Feature Local Class      | TP   | FN   | TN   | FP   | Precision | Recall | F1 Score | Accuracy | Specificity | FPR   | FNR   | Balanced Acc | Conjunto |
|--------------------------|------|------|------|------|-----------|--------|----------|----------|-------------|--------|--------|----------------|-----------|
| **KeyNetFeatureSIFT**    | 1345 | 1691 | 0    | 3036 | 0.31      | 0.44   | 0.36     | 0.22     | 0.000       | 1.000  | 0.557  | 0.22           | Matches   |
|                          | 1469 | 1567 | 103  | 2933 | 0.33      | 0.48   | 0.39     | 0.26     | 0.034       | 0.966  | 0.516  | 0.26           | Scores    |
| **KeyNetFeatureSosNet**  | 1409 | 1627 | 614  | 2422 | 0.37      | 0.46   | 0.41     | 0.33     | 0.202       | 0.798  | 0.536  | 0.33           | Matches   |
|                          | 1351 | 1685 | 2887 | 149  | 0.90      | 0.44   | 0.60     | 0.70     | 0.951       | 0.049  | 0.555  | 0.70           | Scores    |
| **REKDSosNet**           | 2974 | 62   | 1458 | 1578 | 0.65      | 0.98   | 0.78     | 0.73     | 0.480       | 0.520  | 0.020  | 0.73           | Matches   |
|                          | 3031 | 5    | 2641 | 395  | 0.88      | 1.00   | 0.94     | 0.93     | 0.870       | 0.130  | 0.002  | 0.93           | Scores    |
| **REKDHardNet**          | 2939 | 97   | 490  | 2546 | 0.54      | 0.97   | 0.69     | 0.56     | 0.161       | 0.839  | 0.032  | 0.56           | Matches   |
|                          | 3023 | 13   | 1705 | 1331 | 0.69      | 1.00   | 0.82     | 0.78     | 0.562       | 0.438  | 0.004  | 0.78           | Scores    |
| **SingularPointSosNet**  | 3025 | 11   | 1408 | 1628 | 0.65      | 1.00   | 0.79     | 0.73     | 0.464       | 0.536  | 0.004  | 0.73           | Matches   |
|                          | 3031 | 5    | 2533 | 503  | 0.86      | 1.00   | 0.92     | 0.92     | 0.834       | 0.166  | 0.002  | 0.92           | Scores    |
| **SingularPointHardNet** | 3014 | 22   | 412  | 2624 | 0.53      | 0.99   | 0.69     | 0.56     | 0.136       | 0.864  | 0.007  | 0.56           | Matches   |
|                          | 3034 | 2    | 1525 | 1511 | 0.67      | 1.00   | 0.80     | 0.75     | 0.502       | 0.498  | 0.001  | 0.75           | Scores    |




### 🌲 Métricas de Avaliação — Dataset Madeira (Num Features = 60, Distance = 0.9, Threshold = 0.5)

| Feature Local Class      | TP   | FN   | TN   | FP   | Precision | Recall | F1 Score | Accuracy | Specificity | FPR   | FNR   | Balanced Acc | Conjunto |
|--------------------------|------|------|------|------|-----------|--------|----------|----------|-------------|--------|--------|----------------|-----------|
| **KeyNetFeatureSIFT**    | 1304 | 2656 | 370  | 3590 | 0.27      | 0.33   | 0.29     | 0.21     | 0.09        | 0.91   | 0.67   | 0.21           | Matches   |
|                          | 1729 | 2231 | 2982 | 978  | 0.64      | 0.44   | 0.52     | 0.59     | 0.75        | 0.25   | 0.56   | 0.59           | Scores    |
| **KeyNetFeatureSosNet**  | 1616 | 2344 | 2939 | 1021 | 0.61      | 0.41   | 0.49     | 0.58     | 0.74        | 0.26   | 0.59   | 0.58           | Matches   |
|                          | 1465 | 2495 | 3935 | 25   | 0.98      | 0.37   | 0.54     | 0.68     | 0.99        | 0.01   | 0.63   | 0.68           | Scores    |
| **REKDSosNet**           | 2449 | 1511 | 3830 | 130  | 0.95      | 0.62   | 0.75     | 0.79     | 0.97        | 0.03   | 0.38   | 0.79           | Matches   |
|                          | 2439 | 1521 | 3952 | 8    | 1.00      | 0.62   | 0.76     | 0.81     | 1.00        | 0.00   | 0.38   | 0.81           | Scores    |
| **REKDHardNet**          | 2503 | 1457 | 3775 | 185  | 0.93      | 0.63   | 0.75     | 0.79     | 0.95        | 0.05   | 0.37   | 0.79           | Matches   |
|                          | 2492 | 1468 | 3936 | 24   | 0.99      | 0.63   | 0.77     | 0.81     | 0.99        | 0.01   | 0.37   | 0.81           | Scores    |
| **SingularPointSosNet**  | 3697 | 263  | 3847 | 113  | 0.97      | 0.93   | 0.95     | 0.95     | 0.97        | 0.03   | 0.07   | 0.95           | Matches   |
|                          | 3693 | 267  | 3944 | 16   | 1.00      | 0.93   | 0.96     | 0.96     | 1.00        | 0.00   | 0.07   | 0.96           | Scores    |
| **SingularPointHardNet** | 3741 | 219  | 3547 | 413  | 0.90      | 0.94   | 0.92     | 0.92     | 0.90        | 0.10   | 0.06   | 0.92           | Matches   |
|                          | 3760 | 200  | 3913 | 47   | 0.99      | 0.95   | 0.97     | 0.97     | 0.99        | 0.01   | 0.05   | 0.97           | Scores    |

###  Métricas de Avaliação — Dataset Fibras (Num Features = 60, Distance = 0.9, Threshold = 0.5)
| Feature Local Class  | Conjunto | TP  | FN  | TN  | FP  | Precision | Recall | F1 Score | Accuracy | Specificity | FPR  | FNR  | Balanced Acc |
| -------------------- | -------- | --- | --- | --- | --- | --------- | ------ | -------- | -------- | ----------- | ---- | ---- | ------------ |
| **KeyNetFeatureSIFT**| Matches  | 35  | 163 | 62  | 136 | 0.20      | 0.18   | 0.19     | 0.24     | 0.31        | 0.69 | 0.82 | 0.25         |
|                      | Scores   | 41  | 157 | 113 | 85  | 0.33      | 0.21   | 0.25     | 0.39     | 0.57        | 0.43 | 0.79 | 0.39         |
|**KeyNetFeatureSosNet**| Matches  | 2   | 196 | 129 | 69  | 0.03      | 0.01   | 0.01     | 0.33     | 0.65        | 0.35 | 0.99 | 0.33         |
|                      | Scores   | 2   | 196 | 198 | 0   | 1.00      | 0.01   | 0.02     | 0.51     | 1.00        | 0.00 | 0.99 | 0.51         |
|**REKDSosNet**        | Matches  | 1   | 197 | 135 | 63  | 0.02      | 0.01   | 0.01     | 0.34     | 0.68        | 0.32 | 0.99 | 0.34         |
|                      | Scores   | 0   | 198 | 198 | 0   | 0.00      | 0.00   | 0.00     | 0.50     | 1.00        | 0.00 | 1.00 | 0.50         |
|**REKDHardNet**       | Matches  | 3   | 195 | 72  | 126 | 0.02      | 0.02   | 0.02     | 0.19     | 0.36        | 0.64 | 0.98 | 0.19         |
|                      | Scores   | 0   | 198 | 198 | 0   | 0.00      | 0.00   | 0.00     | 0.50     | 1.00        | 0.00 | 1.00 | 0.50         |
|**SingularPointSosNet**| Matches  | 114 | 84  | 198 | 0   | 1.00      | 0.58   | 0.73     | 0.79     | 1.00        | 0.00 | 0.42 | 0.79         |
|                      | Scores   | 113 | 85  | 198 | 0   | 1.00      | 0.57   | 0.73     | 0.79     | 1.00        | 0.00 | 0.43 | 0.79         |
|**SingularPointHardNet**| Matches  | 122 | 76  | 195 | 3   | 0.98      | 0.62   | 0.76     | 0.80     | 0.99        | 0.02 | 0.38 | 0.81         |
|                      | Scores   | 122 | 76  | 198 | 0   | 1.00      | 0.62   | 0.76     | 0.81     | 1.00        | 0.00 | 0.38 | 0.81         |