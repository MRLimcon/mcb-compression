# Importe as bibliotecas necessárias
import utils
import numpy as np
import cv2

# Carregue a imagem de entrada usando uma função de utilidade personalizada "carregar_imagem"
img = utils.carregar_imagem("./teste.jpg")

# Obtenha a resolução (forma) da imagem de entrada
res = img.shape[:-1]

scaling = 2

# Calcule a resolução para a imagem de baixa resolução (a metade da resolução original)
low_res = [int(val / scaling) for val in res]

# Realize a FFT (Transformada Rápida de Fourier) em cada canal de cor da imagem de entrada
fft_high_res = utils.aplicar_fft(img)

# Exiba o logaritmo dos valores absolutos da FFT (usado para visualização)
utils.mostrar_img(np.log(np.abs(fft_high_res[:, :, 0]) + 1))

# Armazenar a FFT subdimensionada para retirar os valores de frequência altos
reconstructed_fft = utils.subdimensionar_fft(fft_high_res, low_res)

# Aplicar a inversa da FFT na FFT subdimensionada
new_low_res_img = utils.aplicar_ifft(reconstructed_fft)

# Arredonde a parte real da imagem de baixa resolução, divida por "scaling" (diminuiu o tamanho da imagem em "scaling" em cada eixo), limite os valores a [0, 255] e converta para inteiros sem sinal de 8 bits
new_low_res_img = np.round(np.clip(np.real(new_low_res_img) / scaling, 0, 255)).astype(
    np.uint8
)

# Salve a imagem de baixa resolução reconstruída
utils.salvar_imagem("imagem_fft_subamostrada.jpg", new_low_res_img)

# Retornar a FFT ao normal
reconstructed_fft = utils.superdimensionar_fft(reconstructed_fft, res)

# Exiba o logaritmo dos valores absolutos da FFT reconstruída
utils.mostrar_img(np.log(np.abs(reconstructed_fft[:, :, 0]) + 1))

# Aplicar a inversa da FFT
new_img = utils.aplicar_ifft(reconstructed_fft)

# Arredonde a parte real da imagem, limite os valores a [0, 255] e converta para inteiros sem sinal de 8 bits
new_img = np.round(np.clip(np.real(new_img), 0, 255)).astype(np.uint8)

# Salve a imagem de alta resolução reconstruída ("comprimida")
utils.salvar_imagem("imagem_fft_comprimida.jpg", new_img)

# Redimensione a imagem original para baixa resolução usando o OpenCV
low_res_img = cv2.resize(img, low_res)

# Realize a FFT em cada canal de cor da imagem de baixa resolução
fft_low_res = utils.aplicar_fft(low_res_img)

# Superdimensionar a FFT da imagem de baixa resolução
reconstructed_fft2 = utils.superdimensionar_fft(fft_low_res, res)

# Exiba o logaritmo dos valores absolutos da FFT reconstruída para a imagem de baixa resolução
utils.mostrar_img(np.log(np.abs(reconstructed_fft2[:, :, 0]) + 1))

# Aplicar a inversa da FFT
new_low_res_img = utils.aplicar_ifft(reconstructed_fft2)

# Arredonde a parte real da imagem de baixa resolução, multiplique por 2 (aumentou o tamanho da imagem em 2 em cada eixo), limite os valores a [0, 255] e converta para inteiros sem sinal de 8 bits
new_low_res_img = np.round(np.clip(np.real(new_low_res_img) * scaling, 0, 255)).astype(
    np.uint8
)

# Salve a imagem de baixa resolução reconstruída
utils.salvar_imagem("imagem_fft_amostrada.jpg", new_low_res_img)
