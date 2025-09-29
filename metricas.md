## 📊 Métricas para Avaliação de Autenticação Biométrica

No contexto de autenticação biométrica, a escolha das melhores métricas depende fortemente do risco de cada tipo de erro (**falso positivo** vs **falso negativo**).

### 🔐 Sobre o Problema

A autenticação biométrica é um problema binário, onde:

- **Positivo (1):** Usuário é quem diz ser (genuíno)
- **Negativo (0):** Usuário é um impostor

**Principais erros:**

- **Falso Positivo (FP):** Um impostor é aceito ❗️⚠️
- **Falso Negativo (FN):** Um usuário legítimo é rejeitado

---

### 🧠 Interpretação das Métricas

| Métrica              | Importância em Biometria                                                    |
|----------------------|------------------------------------------------------------------------------|
| **Precision**         | Indica o quanto das autenticações aceitas são realmente genuínas.           |
| **Recall (TPR)**      | Mostra o quanto dos genuínos foram aceitos. Importante para usabilidade.    |
| **F1 Score**          | Equilíbrio entre precisão e recall — útil, mas pode ocultar erros graves.   |
| **Accuracy**          | Pode enganar em bases desbalanceadas (ex: muitos negativos).                |
| **Specificity (TNR)** | Muito importante! Mede a capacidade de rejeitar impostores. ✅               |
| **FPR**               | Crítica! Alta FPR = muitos impostores sendo aceitos. ⚠️                     |
| **FNR**               | Alta FNR = usuários legítimos sendo rejeitados (ruim para UX).              |
| **Balanced Acc**      | Útil para bases desbalanceadas — média de TPR e TNR.                        |

---

### 🧮 Fórmulas das Métricas

**Definições:**

- `TP`: Verdadeiro Positivo – Genuíno aceito corretamente  
- `TN`: Verdadeiro Negativo – Impostor rejeitado corretamente  
- `FP`: Falso Positivo – Impostor aceito erroneamente ❗️  
- `FN`: Falso Negativo – Genuíno rejeitado erroneamente ⚠️

| Métrica              | Fórmula                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Precision**         | \( \text{Precision} = \frac{TP}{TP + FP} \)                             |
| **Recall (TPR)**      | \( \text{Recall} = \frac{TP}{TP + FN} \)                                |
| **F1 Score**          | \( \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \) |
| **Accuracy**          | \( \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \)               |
| **Specificity (TNR)** | \( \text{Specificity} = \frac{TN}{TN + FP} \)                           |
| **FPR**               | \( \text{FPR} = \frac{FP}{FP + TN} \)                                   |
| **FNR**               | \( \text{FNR} = \frac{FN}{FN + TP} \)                                   |
| **Balanced Accuracy** | \( \text{Balanced Acc} = \frac{\text{Recall} + \text{Specificity}}{2} \) |

---

### 🎯 Resumo Recomendado

| Objetivo                      | Métricas-chave                 |
|-------------------------------|--------------------------------|
| **Segurança** (evitar falsos aceites) | **FPR**, **Specificity**             |
| **Usabilidade** (evitar falsos rejeites) | **FNR**, **Recall**                |
| **Equilíbrio geral**         | **Balanced Accuracy**, **F1 Score** |
