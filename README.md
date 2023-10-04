# Script de Compressão de Imagens

Este script permite comprimir uma imagem usando uma técnica de compressão lossy que envolve a conversão para o espaço de cores LAB, aplicação de subamostragem de croma, realização de uma Transformada Cosseno Discreta (DCT) e compressão dos coeficientes resultantes usando o Zstandard.

## Uso

1. **Dependências**

   Antes de executar o script, certifique-se de ter as seguintes dependências instaladas:

   - Python 3
   - NumPy
   - Matplotlib
   - Pillow (PIL)
   - OpenCV (cv2)
   - SciPy
   - Zstandard (zstd)

   Você pode instalar essas dependências usando o `pip`:

   ```bash
   pip install numpy matplotlib pillow opencv-python-headless scipy zstandard
   ```


2. **Executando o Script**

    Para comprimir uma imagem, execute o script a partir da linha de comando com o seguinte comando:

        ```python compression.py <caminho_da_imagem>```

    Substitua `<caminho_da_imagem>` pelo caminho da imagem que você deseja comprimir. Por exemplo:

        ```python compression.py ./teste.jpg```

    O script executará a compressão e exibirá a imagem comprimida.

3. **Saída**

O script fornecerá as seguintes informações:

- Tamanhos das imagens comprimidas e originais em bytes.
- Tamanhos das imagens DCT comprimidas e subamostradas.
- A imagem reconstruída a partir dos coeficientes da DCT também será exibida.

## Como Funciona

O script segue estas etapas principais:

1. Carrega a imagem de entrada.
2. Converte a imagem para o espaço de cores LAB.
3. Aplica subamostragem de croma à imagem LAB.
4. Realiza uma Transformada Cosseno Discreta (DCT) na imagem subamostrada.
5. Aplica limiarização aos coeficientes da DCT.
6. Calcula e exibe os tamanhos das imagens comprimidas e originais.
7. Comprime as imagens usando o Zstandard.
8. Exibe a imagem comprimida e a imagem reconstruída a partir dos coeficientes da DCT.

## Licença

Este script é fornecido sob a [Licença MIT](LICENSE).

Sinta-se à vontade para usar e modificar o script de acordo com suas necessidades. Se encontrar problemas ou tiver sugestões de melhorias, por favor, abra uma issue ou contribua para o projeto.

Feliz compressão de imagens!
