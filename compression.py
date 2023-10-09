import sys
import utils
import zstandard as zstd


def principal(caminho_da_imagem, caminho_final, percentil=90):
    # Carregar uma imagem
    img = utils.carregar_imagem(caminho_da_imagem)

    # Converter a imagem para o espaço de cor LAB
    img2 = utils.converter_lab(img)

    # Aplicar subamostragem de croma à imagem LAB
    sub_amostrada = utils.subamostragem_croma(img2)

    # Aplicar DCT à imagem subamostrada
    amostrada = utils.dct_img(sub_amostrada, percentil=percentil)

    # Calcular os bytes usados pelas imagens DCT e subamostrada
    bytes_utilizados = (
        amostrada[0].tobytes() + amostrada[1].tobytes() + amostrada[2].tobytes()
    )
    bytes_subamostrados = (
        sub_amostrada[0].tobytes()
        + sub_amostrada[1].tobytes()
        + sub_amostrada[2].tobytes()
    )

    # Imprimir os tamanhos das imagens comprimidas e originais
    print(f"DCT e subamostrada: {len(bytes_utilizados)}")
    print(f"Subamostrada: {len(bytes_subamostrados)}")
    print(f"Original: {len(img.tobytes())}")

    print("")

    # Comprimir as imagens usando o Zstandard e imprimir os tamanhos comprimidos
    print(f"DCT e subamostrada comprimida: {len(zstd.compress(bytes_utilizados))}")
    print(f"Subamostrada comprimida: {len(zstd.compress(bytes_subamostrados))}")
    print(f"Original comprimida: {len(zstd.compress(img.tobytes()))}")

    # Reconstruir a imagem a partir dos coeficientes da DCT e exibi-la
    imagem_retornada = utils.idct_img(amostrada)
    utils.mostrar_img(imagem_retornada)

    # Salvar a imagem comprimida
    utils.salvar_imagem(caminho_final, imagem_retornada)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Uso: python compression.py <caminho_da_imagem> <caminho_final> <percentil>"
        )
    else:
        caminho_da_imagem = sys.argv[1]
        caminho_final = sys.argv[2]
        percentil = float(sys.argv[3])
        principal(caminho_da_imagem, caminho_final, percentil)
