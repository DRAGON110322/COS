import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def to_grayscale(image):
    return 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]


def convolve3x3(image, kernel):
    h, w = image.shape
    padded = np.pad(image, 1, mode='edge')
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            result += kernel[i, j] * padded[i : i + h, j : j + w]
    return result


def apply_gaussian_filter(image):
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 16.0
    return convolve3x3(image, kernel)


def apply_sobel(image):
    kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    gx = convolve3x3(image, kx)
    gy = convolve3x3(image, ky)

    magnitude = np.sqrt(gx**2 + gy**2)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255.0
    return magnitude


def main():
    filepath = 'input.jpg'
    bgr_image = cv2.imread(filepath)
    if bgr_image is None:
        raise FileNotFoundError(f"Файл {filepath} не найден.")

    rgb_image = bgr_image[:, :, ::-1]
    gray_image = to_grayscale(bgr_image)

    filtered_image = apply_gaussian_filter(gray_image)
    sobel_magnitude = apply_sobel(filtered_image)

    initial_threshold = 40.0

    fig, (ax_orig, ax_res) = plt.subplots(1, 2, figsize=(11, 5))
    plt.subplots_adjust(bottom=0.22)

    ax_orig.imshow(rgb_image)
    ax_orig.set_title("Исходное изображение")
    ax_orig.axis("off")

    binary_edges = np.where(sobel_magnitude >= initial_threshold, 255, 0).astype(np.uint8)
    contour_plot = ax_res.imshow(binary_edges, cmap='gray')
    ax_res.set_title(f"Контур (Порог: {initial_threshold:.0f})")
    ax_res.axis("off")

    slider_ax = plt.axes([0.2, 0.08, 0.6, 0.04])
    threshold_slider = Slider(
        ax=slider_ax,
        label='Порог ',
        valmin=0.0,
        valmax=255.0,
        valinit=initial_threshold,
        valstep=1.0
    )

    def update(val):
        thresh = threshold_slider.val
        updated_edges = np.where(sobel_magnitude >= thresh, 255, 0).astype(np.uint8)
        contour_plot.set_data(updated_edges)
        ax_res.set_title(f"Контур (Порог: {thresh:.0f})")
        fig.canvas.draw_idle()

    threshold_slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    main()