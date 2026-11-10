import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_noise_percent(image: np.ndarray, noise_percent: float = 10.0, salt_ratio: float = 0.5) -> np.ndarray:

    noise_percent = np.clip(noise_percent, 0.0, 100.0)
    
    # Переводим процент в вероятность от 0.0 до 1.0
    total_prob = noise_percent / 100.0
    
    salt_prob = total_prob * salt_ratio
    pepper_prob = total_prob * (1.0 - salt_ratio)
    
    noisy_image = image.copy()
    h, w = image.shape[:2]
    
    # Создаем матрицу случайных чисел от 0.0 до 1.0 для каждого пикселя
    rand_matrix = np.random.rand(h, w)
    
    salt_mask = rand_matrix < salt_prob
    noisy_image[salt_mask] = 255
    
    pepper_mask = (rand_matrix >= salt_prob) & (rand_matrix < (salt_prob + pepper_prob))
    noisy_image[pepper_mask] = 0
    
    return noisy_image


def manual_median_filter(image: np.ndarray, kernel_size: tuple = (3, 5)) -> np.ndarray:

    kh, kw = kernel_size
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Размеры окна фильтра должны быть нечетными числами (например, (3, 3), (3, 5), (5, 5))!")

    pad_h = kh // 2
    pad_w = kw // 2
    
    is_color = (image.ndim == 3)

    if is_color:
        h, w, c = image.shape
        # Краевое дополнение (padding) методом отражения края (mode='edge'),
        # чтобы избавиться от черных рамок по краям кадра
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
        output = np.zeros_like(image)

        # Скользящее прямоугольное окно по каждому пикселю
        for i in range(h):
            for j in range(w):
                # Вырезаем область окна размером (kh, kw, c)
                window = padded[i : i + kh, j : j + kw]
                
                # Преобразуем окно к форме (kh * kw, c), объединяя пространственные координаты
                flat_window = window.reshape(-1, c)
                
                # Сортируем значения по каждому цветовому каналу независимо
                sorted_vals = np.sort(flat_window, axis=0)
                
                # Выбираем медианный (центральный) элемент
                median_idx = len(sorted_vals) // 2
                output[i, j] = sorted_vals[median_idx]
    else:
        h, w = image.shape
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(image)

        for i in range(h):
            for j in range(w):
                window = padded[i : i + kh, j : j + kw]
                sorted_vals = np.sort(window.flatten())
                median_idx = len(sorted_vals) // 2
                output[i, j] = sorted_vals[median_idx]

    return output


def main():
    input_path = "input.jpg"
    output_path = "output_cleaned.jpg"

    # Читаем исходное изображение
    original_img = cv2.imread(input_path)
    if original_img is None:
        raise FileNotFoundError(f"Не удалось прочитать файл {input_path}")

    print("1. Исходное фото успешно загружено.")

    # Наложение импульсного шума
    noise_level = 15.0
    noisy_img = add_noise_percent(original_img, noise_percent=noise_level)
    print("2. Шум добавлен.")

    # фильтрация
    rect_kernel = (3, 5)
    print(f"3. Выполняется медианная фильтрация с окном {rect_kernel}...")
    cleaned_img = manual_median_filter(noisy_img, kernel_size=rect_kernel)
    print("   Фильтрация завершена.")

    cv2.imwrite(output_path, cleaned_img)
    print(f"4. Очищенное фото успешно сохранено в '{output_path}'.")

    orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
    cleaned_rgb = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(orig_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Зашумленное фото ({noise_level}% шума)")
    plt.imshow(noisy_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Очищенное (Медиана {rect_kernel[0]}x{rect_kernel[1]})")
    plt.imshow(cleaned_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()