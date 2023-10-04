import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.fftpack import dctn, idctn


# Definir uma função para a Transformada Cosseno Discreta 2D (DCT)
def dct2(a):
    # Deslocar os dados de entrada para ter média zero
    new_a = np.array(a, dtype=np.int16) - 128
    # Aplicar a DCT com normalização 'ortho'
    result = dctn(new_a, norm="ortho")
    return np.round(result).astype(np.int16)


# Definir uma função para realizar DCT em blocos em uma imagem
def block_dct(im: np.ndarray[np.int8], percentil):
    imsize = im.shape
    dct = np.zeros(imsize, dtype=np.int16)

    # Dividir a imagem em blocos de 16x16 e aplicar a DCT em cada bloco
    for i in np.arange(0, imsize[0], 16):
        for j in np.arange(0, imsize[1], 16):
            dct[i : (i + 16), j : (j + 16)] = dct2(im[i : (i + 16), j : (j + 16)])

    # Aplicar limiarização baseada em percentil nos coeficientes da DCT
    valores_absolutos = np.abs(dct)
    dct = dct * (valores_absolutos > np.percentile(valores_absolutos, percentil))

    return dct


# Definir uma função para a Transformada Cosseno Inversa Discreta 2D (IDCT)
def idct2(a):
    return np.array(idctn(a, norm="ortho") + 128, dtype=np.uint8)


# Definir uma função para realizar IDCT em blocos em uma imagem
def block_idct(im: np.ndarray[np.int8]):
    imsize = im.shape
    im_dct = np.zeros(imsize, dtype=np.int8)

    # Dividir a imagem em blocos de 16x16 e aplicar IDCT em cada bloco
    for i in np.arange(0, imsize[0], 16):
        for j in np.arange(0, imsize[1], 16):
            im_dct[i : (i + 16), j : (j + 16)] = idct2(im[i : (i + 16), j : (j + 16)])

    return im_dct


# Definir uma função para aplicar DCT em uma imagem
def dct_img(
    imagem_amostrada: dict[np.ndarray], em_blocos: bool = True, percentil: float = 92
):
    img = {}

    if em_blocos:
        # Aplicar DCT em blocos a cada canal de cor
        img[0] = block_dct(imagem_amostrada[0], percentil)
        img[1] = block_dct(imagem_amostrada[1], percentil)
        img[2] = block_dct(imagem_amostrada[2], percentil)
    else:
        # Aplicar DCT a cada canal de cor sem processamento em blocos
        img[0] = dct2(imagem_amostrada[0])
        valores_absolutos = np.abs(img[0])
        img[0] = img[0] * (valores_absolutos > np.percentile(img[0], percentil))
        img[1] = dct2(imagem_amostrada[1])
        valores_absolutos = np.abs(img[1])
        img[1] = img[1] * (valores_absolutos > np.percentil(img[1], percentil))
        img[2] = dct2(imagem_amostrada[2])
        valores_absolutos = np.abs(img[2])
        img[2] = img[2] * (valores_absolutos > np.percentile(img[2], percentil))

    return img


# Definir uma função para aplicar IDCT em uma imagem e convertê-la para o espaço de cor LAB
def idct_img(imagem_amostrada: dict[np.ndarray], em_blocos: bool = True):
    img = np.zeros(list(imagem_amostrada[0].shape) + [3], dtype=np.uint8)
    res = imagem_amostrada[0].shape

    if em_blocos:
        # Aplicar IDCT em blocos e redimensionar os canais de croma
        img[:, :, 0] = block_idct(imagem_amostrada[0])
        img[:, :, 1] = np.array(
            Image.fromarray(block_idct(imagem_amostrada[1])).resize(res, Image.LANCZOS),
            dtype=np.uint8,
        )
        img[:, :, 2] = np.array(
            Image.fromarray(block_idct(imagem_amostrada[2])).resize(res, Image.LANCZOS),
            dtype=np.uint8,
        )
    else:
        # Aplicar IDCT a cada canal de cor e redimensionar os canais de croma
        img[:, :, 0] = idct2(imagem_amostrada[0])
        img[:, :, 1] = np.array(
            Image.fromarray(idct2(imagem_amostrada[1])).resize(res, Image.LANCZOS),
            dtype=np.uint8,
        )
        img[:, :, 2] = np.array(
            Image.fromarray(idct2(imagem_amostrada[2])).resize(res, Image.LANCZOS),
            dtype=np.uint8,
        )

    return converter_rgb(img)


# Definir uma função para converter uma imagem para o espaço de cor LAB
def converter_lab(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


# Definir uma função para converter uma imagem de LAB para RGB
def converter_rgb(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


# Definir uma função para subamostragem de croma
def subamostragem_croma(img: np.ndarray):
    nova_img = {0: img[:, :, 0]}

    chroma_1 = Image.fromarray(img[:, :, 1])
    chroma_2 = Image.fromarray(img[:, :, 2])
    resolucao = chroma_1.size
    nova_res = [int(res * 0.5) for res in resolucao]
    nova_img[1] = np.array(chroma_1.resize(nova_res, Image.LANCZOS))
    nova_img[2] = np.array(chroma_2.resize(nova_res, Image.LANCZOS))
    return nova_img


# Definir uma função para carregar uma imagem
def carregar_imagem(caminho):
    if "CR2" not in caminho:
        img = cv2.imread(caminho, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# Definir uma função para exibir uma imagem
def mostrar_img(img):
    plt.imshow(img)
    plt.show()


# Definir uma função para salvar uma imagem
def salvar_imagem(caminho: str, matriz_imagem, aprimorar=True):
    imagem = Image.fromarray(matriz_imagem)
    imagem.save(caminho)
