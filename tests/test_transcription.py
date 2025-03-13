# test_transcription.py
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from pathlib import Path

# Добавить корневой каталог проекта в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.audio.extractor import AudioExtractor
from src.transcription.speech_to_text import TranscriberFactory
from src.config import STT_SETTINGS

# Настроить логирование
logging.basicConfig(level=logging.INFO)


def main():
    # Шаг 1: Проверка наличия входного файла
    video_path = "data/input/lecture.mp4"  # Путь к тестовому видео

    if not os.path.exists(video_path):
        print(f"❌ Видеофайл не найден: {video_path}")
        print("Пожалуйста, поместите видеофайл в директорию data/input/")
        return

    # Шаг 2: Извлечение аудио (если еще не сделано)
    extractor = AudioExtractor()

    try:
        audio_path = extractor.extract_audio(video_path)
        print(f"✅ Аудио извлечено: {audio_path}")
    except Exception as e:
        print(f"❌ Ошибка при извлечении аудио: {str(e)}")
        return

    # Шаг 3: Транскрибирование аудио
    print("\n🔍 Запуск транскрибирования аудио...")

    # Создаем транскрайбер с помощью фабрики
    transcriber = TranscriberFactory.create_transcriber()

    try:
        # Транскрибируем аудио
        transcript_text, metadata = transcriber.transcribe(audio_path)

        # Выводим результаты
        print("\n✅ Транскрибирование завершено успешно!")
        print(
            f"📊 Обнаружен язык: {metadata.get('language', 'не определен')} (вероятность: {metadata.get('language_probability', 0):.2f})")
        print(f"⏱️ Длительность аудио: {metadata.get('duration', 0):.2f} секунд")

        # Показываем первые 200 символов транскрипции
        print("\n📝 Начало транскрипции:")
        print("-" * 80)
        print(transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text)
        print("-" * 80)

        # Показываем путь к файлам
        segments = metadata.get("segments", [])
        print(f"\n📊 Получено {len(segments)} сегментов")
        print(f"💾 Полный текст сохранен в папке data/transcripts/")

    except Exception as e:
        print(f"❌ Ошибка при транскрибировании: {str(e)}")


if __name__ == "__main__":
    main()