"""
Simplified Wav2Lip inference code that uses non-ML based components for lip sync
"""
import cv2
import numpy as np
import os
import argparse
import subprocess
import wave
import sys
from tqdm import tqdm

# Параметры командной строки
parser = argparse.ArgumentParser(description='Упрощенная версия Wav2Lip без необходимости ML')
parser.add_argument('--face', type=str, required=True, help='Изображение лица')
parser.add_argument('--audio', type=str, required=True, help='Аудиофайл WAV')
parser.add_argument('--outfile', type=str, default='Wav2Lip/outputs/result.mp4', help='Выходной видеофайл')
parser.add_argument('--fps', type=float, default=25.0, help='FPS видео')
args = parser.parse_args()

def load_wav(path):
    """Загружает аудиофайл WAV"""
    with wave.open(path, 'rb') as wav_file:
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        data = wav_file.readframes(n_frames)
        data = np.frombuffer(data, dtype=np.int16)
        duration = n_frames / framerate
        return data, framerate, duration

def create_simple_animation(face_img, audio_data, fps, duration):
    """
    Создает простую анимацию движения губ на основе аудио
    """
    # Создаем временные файлы
    temp_dir = 'Wav2Lip/temp'
    os.makedirs(temp_dir, exist_ok=True)
    frames_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Очищаем папку с кадрами для каждого нового запуска
    # Это предотвратит смешивание старых и новых кадров
    for file in os.listdir(frames_dir):
        if file.endswith('.jpg'):
            os.remove(os.path.join(frames_dir, file))
            
    # Для отладки - выводим размер изображения
    print(f"Загружаем изображение: {face_img}")
    
    # Загружаем изображение лица
    img = cv2.imread(face_img)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {face_img}")
        
    # Для более надежной работы масштабируем изображение до разумного размера
    h, w = img.shape[:2]
    print(f"Исходный размер изображения: {w}x{h}")
    
    # Ограничиваем размер, чтобы избежать проблем с кодированием
    max_width = 1280
    if w > max_width:
        scale_factor = max_width / w
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)
        # Используем cv2.INTER_AREA для уменьшения размера
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Изображение масштабировано до: {new_width}x{new_height}")
    
    h, w = img.shape[:2]
    
    # Более точное определение лица и его частей
    # Используем каскады Хаара для обнаружения лица и рта
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Улучшаем контраст для лучшего обнаружения черт лица
    gray = cv2.equalizeHist(gray)
    
    # Поиск лица с улучшенными параметрами
    faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(80, 80))
    
    # Переменная для сохранения исходного изображения рта
    original_mouth_img = None
    face_found = False
    
    # Определяем область рта
    if len(faces) > 0:
        face_found = True
        # Берем первое найденное лицо
        x, y, w_face, h_face = faces[0]
        
        # Рисуем лицо для отладки
        img_with_face = img.copy()
        cv2.rectangle(img_with_face, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(temp_dir, 'face_detection.jpg'), img_with_face)
        
        # Уточняем область поиска рта - нижняя половина лица с расширением
        # Делаем область поиска шире и выше для повышения вероятности нахождения рта
        roi_top = y + int(h_face * 0.45)  # Начинаем выше
        roi_bottom = y + h_face
        roi_left = max(0, x - int(w_face * 0.1))  # Расширяем область поиска влево
        roi_right = min(w, x + w_face + int(w_face * 0.1))  # Расширяем область поиска вправо
        
        roi_gray = gray[roi_top:roi_bottom, roi_left:roi_right]
        roi_color = img[roi_top:roi_bottom, roi_left:roi_right]
        
        # Пробуем разные параметры для обнаружения рта
        mouth_detected = False
        
        # Набор параметров для поиска рта (scale_factor, min_neighbors, minSize)
        param_sets = [
            (1.1, 10, (25, 15)),
            (1.2, 8, (20, 12)),
            (1.3, 6, (15, 10)),
            (1.05, 12, (30, 20))
        ]
        
        for scale_factor, min_neighbors, min_size in param_sets:
            mouths = mouth_cascade.detectMultiScale(
                roi_gray, scale_factor, min_neighbors, minSize=min_size
            )
            
            if len(mouths) > 0:
                # Используем найденный рот
                mx, my, mw, mh = mouths[0]
                
                # Координаты рта относительно всего изображения
                mouth_left = roi_left + mx
                mouth_top = roi_top + my
                mouth_right = mouth_left + mw
                mouth_bottom = mouth_top + mh
                
                print(f"Рот обнаружен на координатах: {mouth_left}, {mouth_top}, {mouth_right}, {mouth_bottom}")
                mouth_detected = True
                break
        
        if not mouth_detected:
            # Если рот не найден, используем примерное расположение на основе пропорций лица
            mouth_top = int(y + h_face * 0.65)
            mouth_bottom = int(y + h_face * 0.85)
            mouth_left = int(x + w_face * 0.3)
            mouth_right = int(x + w_face * 0.7)
            print("Рот не обнаружен, используются приблизительные координаты на основе лица")
    
    # Если лицо не обнаружено, используем фиксированное положение
    if not face_found:
        # Используем приблизительные координаты для всего изображения
        mouth_top = int(h * 0.6)
        mouth_bottom = int(h * 0.8)
        mouth_left = int(w * 0.3)
        mouth_right = int(w * 0.7)
        print("Лицо не обнаружено, используются фиксированные координаты рта")
    
    # Убедимся, что область рта не выходит за границы изображения
    mouth_top = max(0, min(h-1, mouth_top))
    mouth_bottom = max(mouth_top+1, min(h, mouth_bottom))
    mouth_left = max(0, min(w-1, mouth_left))
    mouth_right = max(mouth_left+1, min(w, mouth_right))
    
    # Для отладки - смещаем прямоугольник вниз к фактическому рту
    # На основе анализа изображения face_detection.jpg, видно что обнаружение происходит выше фактического рта
    actual_mouth_offset = 10  # смещение вниз в пикселях для попадания на фактический рот
    
    # Корректируем координаты рта на основе визуального анализа фото
    mouth_top = mouth_top + actual_mouth_offset
    mouth_bottom = mouth_bottom + actual_mouth_offset
    
    # Убедимся, что область рта все еще не выходит за границы после корректировки
    mouth_top = max(0, min(h-1, mouth_top))
    mouth_bottom = max(mouth_top+1, min(h, mouth_bottom))
    
    # Сохраняем оригинальное изображение рта для дальнейшей обработки
    original_mouth_img = img[mouth_top:mouth_bottom, mouth_left:mouth_right].copy()
    
    # Рисуем рамку вокруг скорректированной области рта для отладки
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (mouth_left, mouth_top), (mouth_right, mouth_bottom), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(temp_dir, 'face_detection_corrected.jpg'), img_with_rect)
    
    # Рассчитываем количество кадров
    n_frames = int(duration * fps)
    print(f"Создание {n_frames} кадров...")
    
    # Рассчитываем аудио энергию для каждого кадра
    audio_chunks = np.array_split(audio_data, n_frames)
    # Используем среднеквадратичную энергию для лучшего отражения громкости
    energies = [np.sqrt(np.mean(np.square(chunk.astype(float)))) for chunk in audio_chunks]
    
    # Нормализуем энергии к диапазону [0, 1]
    if max(energies) > min(energies):
        # Применяем нелинейное масштабирование для усиления разницы между тихими и громкими сегментами
        energies = np.array(energies)
        energies = (energies - min(energies)) / (max(energies) - min(energies))
        # Применяем кубический корень для более естественного движения губ
        energies = np.power(energies, 1/3)
    else:
        energies = [0.5] * len(energies)
    
    # Сглаживаем движения губ, чтобы не было резких переходов
    smoothed_energies = []
    window_size = 3  # Размер окна сглаживания
    
    for i in range(len(energies)):
        # Для начала и конца используем доступные значения
        if i < window_size // 2:
            window = energies[:i + window_size // 2 + 1]
        elif i >= len(energies) - window_size // 2:
            window = energies[i - window_size // 2:]
        else:
            window = energies[i - window_size // 2:i + window_size // 2 + 1]
        
        smoothed_energies.append(np.mean(window))
    
    energies = smoothed_energies
    
    # Создаем последовательные нумерованные кадры для всей длительности аудио
    for i, energy in enumerate(tqdm(energies)):
        # Создаем новый кадр для каждого момента времени
        frame = img.copy()
        
        # Пропускаем деформацию, если область рта не найдена или некорректна
        if mouth_top >= mouth_bottom or mouth_left >= mouth_right or original_mouth_img is None:
            # В этом случае просто сохраняем исходный кадр
            output_path = os.path.join(frames_dir, f'{i:05d}.jpg')
            cv2.imwrite(output_path, frame)
            continue
            
        # Проверка размеров области рта
        if mouth_bottom - mouth_top < 5 or mouth_right - mouth_left < 5:
            # Если область рта слишком маленькая, используем фиксированный размер
            mouth_height = 30
            mouth_width = 60
            
            # Центр лица
            center_y = h // 2
            center_x = w // 2
            
            # Корректируем область рта
            mouth_top = center_y - mouth_height // 2
            mouth_bottom = center_y + mouth_height // 2
            mouth_left = center_x - mouth_width // 2
            mouth_right = center_x + mouth_width // 2
            
            # Обновляем оригинальное изображение рта
            original_mouth_img = img[mouth_top:mouth_bottom, mouth_left:mouth_right].copy()
            
        # Копируем оригинальное изображение рта
        mouth_img = original_mouth_img.copy()
        
        # Проверяем размеры изображения рта
        m_h, m_w = mouth_img.shape[:2]
        if m_h < 2 or m_w < 2:
            # Пропускаем деформацию, если область рта слишком маленькая
            frame[mouth_top:mouth_bottom, mouth_left:mouth_right] = mouth_img
            output_path = os.path.join(frames_dir, f'{i:05d}.jpg')
            cv2.imwrite(output_path, frame)
            continue
            
        # Центр рта
        m_center_x, m_center_y = m_w // 2, m_h // 2
        
        # Определяем середину рта (линию смыкания губ)
        lip_line_y = m_h // 2
        
        # Вычисляем степень открытия рта в зависимости от энергии звука
        # Энергия audio в диапазоне от 0 до 1
        open_amount = energy * 0.7  # Максимальное открытие 70% от высоты рта
        
        # Простая анимация открытия рта
        if energy > 0.05:  # Порог для открытия рта
            # Создаем маску для внутренней части рта
            inner_mouth_mask = np.zeros_like(mouth_img)
            
            # Рассчитываем параметры для эллипса внутренней части рта
            inner_mouth_width = int(m_w * 0.7)  # 70% от ширины рта
            inner_mouth_height = int(m_h * open_amount * 0.8) + 2  # Гарантируем минимальную высоту
            
            # Рисуем эллипс для внутренней части рта (черный)
            cv2.ellipse(mouth_img, 
                      (m_center_x, lip_line_y), 
                      (inner_mouth_width // 2, max(3, inner_mouth_height)), 
                      0, 0, 360, 
                      (40, 30, 80),  # Темный цвет для внутренней части рта
                      -1)
            
            # Добавляем эффект тени внутри рта
            cv2.ellipse(mouth_img, 
                      (m_center_x, lip_line_y + 2),  # Немного смещаем вниз
                      (inner_mouth_width // 3, max(2, inner_mouth_height - 2)), 
                      0, 0, 360, 
                      (20, 15, 50),  # Еще темнее
                      -1)
        
        # Вставляем модифицированный рот обратно в кадр
        try:
            frame[mouth_top:mouth_bottom, mouth_left:mouth_right] = mouth_img
        except ValueError as e:
            print(f"Ошибка при вставке рта: {e}")
            print(f"Размеры: mouth_img={mouth_img.shape}, область={mouth_bottom-mouth_top}x{mouth_right-mouth_left}")
            # В случае ошибки просто пропускаем деформацию
            
        # Сохраняем кадр с последовательной нумерацией
        output_path = os.path.join(frames_dir, f'{i:05d}.jpg')
        cv2.imwrite(output_path, frame)
    
    # Проверяем наличие созданных кадров и их непрерывность
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    frames.sort()
    
    if not frames:
        raise ValueError("Не удалось создать кадры для видео")
    
    print(f"Создано {len(frames)} кадров")
    
    # Проверяем размеры первого кадра
    first_frame_path = os.path.join(frames_dir, frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is not None:
        h, w = first_frame.shape[:2]
        print(f"Размер кадра для видео: {w}x{h}")
        
        # Проверяем на нечетные размеры
        if h % 2 != 0:
            # Изменяем размер всех кадров, чтобы высота была четной
            new_h = h - 1 if h % 2 != 0 else h
            new_w = w - 1 if w % 2 != 0 else w
            print(f"Корректируем размер кадров на четный: {new_w}x{new_h}")
            
            for frame_file in tqdm(frames):
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    cv2.imwrite(frame_path, resized_frame)
    
    # Прямое создание финального видео с аудио в один шаг
    # Эта команда решает проблемы с разной длительностью и форматированием
    try:
        print("Создание финального видео напрямую...")
        direct_cmd = (f"ffmpeg -y -framerate {fps} -i {frames_dir}/%05d.jpg "
                    f"-i {args.audio} -c:v libx264 -crf 23 -preset medium "
                    f"-c:a aac -b:a 128k -shortest -pix_fmt yuv420p {args.outfile}")
        print(f"Выполняем команду: {direct_cmd}")
        subprocess.call(direct_cmd, shell=True)
        
        # Проверяем результат
        if not os.path.exists(args.outfile) or os.path.getsize(args.outfile) < 10000:
            raise ValueError("Не удалось создать видео напрямую")
            
    except Exception as e:
        print(f"Ошибка при прямом создании: {e}")
        print("Пробуем двухэтапный метод...")
        
        # Создаем видео без аудио
        video_path = os.path.join(temp_dir, 'video.mp4')
        cmd = (f"ffmpeg -y -framerate {fps} -i {frames_dir}/%05d.jpg "
              f"-c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p {video_path}")
        print(f"Выполняем команду: {cmd}")
        subprocess.call(cmd, shell=True)
        
        # Добавляем аудио
        if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
            output_cmd = (f"ffmpeg -y -i {video_path} -i {args.audio} -c:v copy "
                        f"-c:a aac -shortest {args.outfile}")
            print(f"Выполняем команду: {output_cmd}")
            subprocess.call(output_cmd, shell=True)
    
    print(f"Результат сохранен в {args.outfile}")
    return args.outfile

def main():
    # Проверка наличия файлов
    if not os.path.isfile(args.face):
        print(f"Ошибка: файл изображения {args.face} не найден")
        return
    
    if not os.path.isfile(args.audio):
        print(f"Ошибка: аудиофайл {args.audio} не найден")
        return
    
    # Создание директории для результата
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    
    try:
        # Загружаем аудио
        audio_data, framerate, duration = load_wav(args.audio)
        
        # Создаем анимацию
        create_simple_animation(args.face, audio_data, args.fps, duration)
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()