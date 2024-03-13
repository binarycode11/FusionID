| preprocessor | local_feature_matcher | global_structurer | global_matcher | accuracy |
|----------|----------|----------|----------|----------|
|   X  |   GFTTAffNetHardNet+SMNN(95)  |   X  |   X |   83.14%  |
|   Row 2  |   Row 2  |   Row 2  |   Row 2  |   Row 2  |
|   Row 3  |   Row 3  |   Row 3  |   Row 3  |   Row 3  |
|   Row 4  |   Row 4  |   Row 4  |   Row 4  |   Row 4  |


### Passos futuro:
1. Testar outros métodos de local_features.
2. Diminuir o tempo de cada etapa(4.8s por busca).
3. Realizar busca em um batch todo.
4. Testar em GPU na parte inicial.
5. Organizar os testes.


43 batchs