# test_audio_extraction.py
from src.audio.extractor import AudioExtractor


def main():
    # Путь к входному видеофайлу
    video_path = "data/input/your_lecture.mp4"  # Замените на имя вашего файла

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