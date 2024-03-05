# PapyrusTech
Referência a "Papyrus", um dos primeiros materiais de escrita, combinado com "Tech", indicando a tecnologia envolvida no reconhecimento de documentos antigos ou novos

Essa estrutura de pipeline foi usado para o Exame de Qualificação.

## Métodos do Pipeline

A seguir, são listados os métodos/algoritmos propostos para cada etapa do pipeline de processamento de imagens. Foi utilizado a implementação do [Kornia](https://kornia.github.io/) para várias das etapas, especialmente detecção e matching de características.

| Etapa do Pipeline       | Métodos/Algoritmos Propostos                 |
|-------------------------|----------------------------------------------|
| **Preprocessamento**    | - Some Noise                                 |
|                         | - Some Filter                                |
| **Detecção**            | - KeyNetDetector + HardNet (KeyNetHardNet)   |
|                         | - SIFTFeature                                |
|                         | - DISKFeatures                               |
|                         | - KoreanDetector + HardNet                   |
| **Matching Local**      | - match_snn                                  |
|                         | - match_smnn                                 |
| **Estruturação Global** | - Delaunay                                   |
| **Matching Global**     | - Floyd Warshall                             |

## Proposta de Estudo de Ablação

Um estudo de ablação envolve modificar ou remover componentes de um sistema para avaliar o impacto de cada um no desempenho geral. A seguir, uma proposta para realizar este estudo no contexto do pipeline de processamento de imagens descrito:

1. **Baseline Completo**: Utilizar todas as etapas do pipeline com um conjunto específico de métodos para estabelecer uma linha de base de desempenho.

2. **Ablação de Preprocessamento**:
   - Remover `Some Noise` e comparar o desempenho.
   - Remover `Some Filter` e comparar o desempenho.
   - Avaliar o impacto da remoção de ambas as etapas de preprocessamento.

3. **Ablação de Detecção**:
   - Substituir o detector (ex: usar `SIFTFeature` ao invés de `KeyNetDetector + HardNet`) mantendo o restante do pipeline constante e comparar o desempenho.

4. **Ablação de Matching Local**:
   - Alternar entre `match_snn` e `match_smnn`, avaliando o impacto no desempenho.

5. **Ablação de Estruturação Global**:
   - Omitir a etapa de `Delaunay` e avaliar o impacto no desempenho do matching global.

6. **Ablação de Matching Global**:
   - Remover a etapa de `Floyd Warshall` para entender seu impacto no desempenho.

7. **Combinações Variadas**:
   - Experimentar com diferentes combinações de métodos em cada etapa para encontrar a configuração que otimiza o desempenho.

Cada experimento deve ser documentado com as métricas de desempenho relevantes (ex: precisão, recall, F1 score). O objetivo é entender a contribuição de cada componente e identificar áreas para otimização.

As implementações específicas de detecção, matching local, e outras etapas onde mencionado, foram baseadas nas funcionalidades disponíveis no [Kornia](https://kornia.github.io/), uma biblioteca de visão computacional para PyTorch.
