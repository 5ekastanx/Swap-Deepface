# Face Swap Project

Этот простой проект выполняет **замену лиц** между двумя изображениями с использованием Haar Cascade из OpenCV для обнаружения лиц. Полученное изображение сохраняется на вашем устройстве.

## Возможности

- Обнаружение лиц на двух изображениях.
- Замена обнаруженных лиц между изображениями.
- Сохранение результата как выходного изображения.

## Требования

- Python 3.x
- OpenCV
- NumPy

## Установка

1. **Создание виртуального окружения**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. **Установите зависимости:**
   Убедитесь, что у вас установлен Python, затем выполните:

   ```bash
   pip install opencv-python numpy
   ```

3. **Скачайте Haar Cascade XML:**

   - Haar Cascade — это предварительно обученная модель для обнаружения лиц.
   - Скачайте файл `haarcascade_frontalface_default.xml` из [репозитория OpenCV на GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).
   - Сохраните файл в той же директории, что и ваш скрипт, или укажите его полный путь в коде.
