# test_audio_extraction.py
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio.extractor import AudioExtractor


def main():
    # Путь к входному видеофайлу
    video_path = "data/input/lecture.mp4"  # Замените на имя вашего файла

    # Создаем экземпляр извлекателя аудио
    extractor = AudioExtractor()

    # Проверяем наличие ffmpeg
    if not extractor.check_ffmpeg_installed():
        print("❌ Ошибка: ffmpeg не установлен! Пожалуйста, установите ffmpeg.")
        return

    # Извлекаем аудио
    try:
        audio_path = extractor.extract_audio(video_path)
        print(f"✅ Аудио успешно извлечено и сохранено в: {audio_path}")
    except Exception as e:
        print(f"❌ Ошибка при извлечении аудио: {str(e)}")


if __name__ == "__main__":
    main()