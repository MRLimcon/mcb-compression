import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import dctn, idctn


def aplicar_fft(img):
    # Criar o array de fft
    fft = np.zeros_like(img).astype(np.complex128)

    # Realize a FFT (Transformada Rápida de Fourier) em cada canal de cor da imagem de entrada
    fft[:, :, 0] = np.fft.fft2(
        img[:, :, 0].copy().astype(np.float64) - 128, norm="ortho"
    )
    fft[:, :, 1] = np.fft.fft2(
        img[:, :, 1].copy().astype(np.float64) - 128, norm="ortho"
    )
    fft[:, :, 2] = np.fft.fft2(
        img[:, :, 2].copy().astype(np.float64) - 128, norm="ortho"
    )
    return fft


def aplicar_ifft(fft):
    # Inicialize uma matriz para armazenar a nova imagem após a inversa da FFT
    new_img = np.zeros(fft.shape).astype(np.complex128)

    # Realize a inversa da FFT em cada canal de cor para reconstruir a imagem
    new_img[:, :, 0] = np.fft.ifft2(fft[:, :, 0], norm="ortho") + 128
    new_img[:, :, 1] = np.fft.ifft2(fft[:, :, 1], norm="ortho") + 128
    new_img[:, :, 2] = np.fft.ifft2(fft[:, :, 2], norm="ortho") + 128

    return new_img


def superdimensionar_fft(fft_low_res, high_res):
    # Calcule a metade da resolução para a imagem de baixa resolução
    low_res = fft_low_res.shape[:-1]
    half_res = [int(val / 2) for val in low_res]
    top_res = [high_res[i] - low_res[i] + half_res[i] for i in range(2)]

    # Inicialize uma matriz para armazenar a FFT reconstruída para a imagem de baixa resolução
    # (Upscaling pela interpolação de Fourier)
    reconstructed_fft = np.zeros(list(high_res) + [3]).astype(np.complex128)

    # Itere sobre os canais de cor (R, G, B)
    for i in range(3):
        # Copie os valores da FFT do canto superior esquerdo até o limite de baixa resolução
        reconstructed_fft[: half_res[0], : half_res[1], i] = fft_low_res[
            : half_res[0], : half_res[1], i
        ]
        # Copie os valores da FFT do canto superior esquerdo até o canto inferior direito
        # na região de baixa resolução
        reconstructed_fft[: half_res[0], top_res[1] :, i] = fft_low_res[
            : half_res[0], half_res[1] :, i
        ]
        # Copie os valores da FFT do canto inferior esquerdo até o canto superior direito
        # na região de baixa resolução
        reconstructed_fft[top_res[0] :, : half_res[1], i] = fft_low_res[
            half_res[0] :, : half_res[1], i
        ]
        # Copie os valores da FFT do canto inferior esquerdo até o canto inferior direito
        # na região de baixa resolução
        reconstructed_fft[top_res[0] :, top_res[1] :, i] = fft_low_res[
            half_res[0] :, half_res[1] :, i
        ]

    return reconstructed_fft


def subdimensionar_fft(fft_high_res, low_res):
    # Calcule a metade da resolução para a imagem de baixa resolução
    high_res = fft_high_res.shape[:-1]
    half_res = [int(val / 2) for val in low_res]
    top_res = [high_res[i] - low_res[i] + half_res[i] for i in range(2)]

    # Inicialize uma matriz para armazenar a FFT reconstruída para a imagem de baixa resolução
    # (Downscaling pela interpolação de Fourier)
    reconstructed_fft = np.zeros(list(low_res) + [3]).astype(np.complex128)

    # Itere sobre os canais de cor (R, G, B)
    for i in range(3):
        # Copie os valores da FFT do canto superior esquerdo até o limite de baixa resolução
        reconstructed_fft[: half_res[0], : half_res[1], i] = fft_high_res[
            : half_res[0], : half_res[1], i
        ]
        # Copie os valores da FFT do canto superior esquerdo até o canto inferior direito
        # na região de baixa resolução
        reconstructed_fft[: half_res[0], half_res[1] :, i] = fft_high_res[
            : half_res[0], top_res[1] :, i
        ]
        # Copie os valores da FFT do canto inferior esquerdo até o canto superior direito
        # na região de baixa resolução
        reconstructed_fft[half_res[0] :, : half_res[1], i] = fft_high_res[
            top_res[0] :, : half_res[1], i
        ]
        # Copie os valores da FFT do canto inferior esquerdo até o canto inferior direito
        # na região de baixa resolução
        reconstructed_fft[half_res[0] :, half_res[1] :, i] = fft_high_res[
            top_res[0] :, top_res[1] :, i
        ]

    return reconstructed_fft


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
    return np.clip(np.round(idctn(a, norm="ortho") + 128), 0, 255).astype(np.uint8)


# Definir uma função para realizar IDCT em blocos em uma imagem
def block_idct(im: np.ndarray[np.int16]):
    imsize = im.shape
    im_dct = np.zeros(imsize, dtype=np.uint8)

    # Dividir a imagem em blocos de 16x16 e aplicar IDCT em cada bloco
    for i in np.arange(0, imsize[0], 16):
        for j in np.arange(0, imsize[1], 16):
            im_dct[i : (i + 16), j : (j + 16)] = idct2(im[i : (i + 16), j : (j + 16)])

    return im_dct.astype(np.uint8)


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
        img[:, :, 1] = cv2.resize(block_idct(imagem_amostrada[1]), res)
        img[:, :, 2] = cv2.resize(block_idct(imagem_amostrada[2]), res)
    else:
        # Aplicar IDCT a cada canal de cor e redimensionar os canais de croma
        img[:, :, 0] = idct2(imagem_amostrada[0])
        img[:, :, 1] = cv2.resize(idct2(imagem_amostrada[1]), res)
        img[:, :, 2] = cv2.resize(idct2(imagem_amostrada[2]), res)

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

    resolucao = img[:, :, 0].shape
    nova_res = [int(res * 0.5) for res in resolucao]
    nova_img[1] = cv2.resize(img[:, :, 1], nova_res)
    nova_img[2] = cv2.resize(img[:, :, 2], nova_res)
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
def salvar_imagem(caminho: str, matriz_imagem):
    img = cv2.cvtColor(matriz_imagem, cv2.COLOR_RGB2BGR)
    cv2.imwrite(caminho, img)
