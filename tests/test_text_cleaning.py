# test_text_cleaning.py
import os
import sys
import logging
from pathlib import Path

# Добавить корневой каталог проекта в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.text.cleaner import TextCleaner
from src.utils.helpers import load_text

# Настроить логирование
logging.basicConfig(level=logging.INFO)


def main():
    # Шаг 1: Проверка наличия входного файла (транскрипта)
    transcript_dir = Path("data/transcripts")

    # Ищем файлы транскриптов с расширением .txt
    transcript_files = list(transcript_dir.glob("*_transcript.txt"))

    if not transcript_files:
        print(f"❌ Файлы транскриптов не найдены в директории {transcript_dir}")
        print("Сначала запустите шаг 1.2 для создания транскриптов")
        return

    # Берем первый найденный файл транскрипта
    transcript_path = transcript_files[0]
    print(f"✅ Найден файл транскрипта: {transcript_path}")

    # Шаг 2: Создаем объект для очистки текста
    cleaner = TextCleaner()

    try:
        # Загружаем исходный текст
        original_text = load_text(transcript_path)

        # Показываем часть исходного текста
        print("\n📝 Исходный текст (первые 200 символов):")
        print("-" * 80)
        print(original_text[:200] + "..." if len(original_text) > 200 else original_text)
        print("-" * 80)

        # Шаг 3: Очищаем текст
        print("\n🔍 Начинаем очистку текста...")

        # Очищаем текст (это сохранит результат в файл)
        cleaned_path = cleaner.process_file(transcript_path)

        # Загружаем очищенный текст
        cleaned_text = load_text(cleaned_path)

        # Показываем результат
        print("\n✅ Очистка текста завершена!")
        print(f"💾 Очищенный текст сохранен в: {cleaned_path}")

        print("\n📝 Очищенный текст (первые 200 символов):")
        print("-" * 80)
        print(cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text)
        print("-" * 80)

        # Показываем статистику
        original_words = len(original_text.split())
        cleaned_words = len(cleaned_text.split())
        reduction = (1 - cleaned_words / original_words) * 100 if original_words > 0 else 0

        print("\n📊 Статистика:")
        print(f"Исходный текст: {original_words} слов")
        print(f"Очищенный текст: {cleaned_words} слов")
        print(f"Сокращение: {reduction:.2f}%")

    except Exception as e:
        print(f"❌ Ошибка при очистке текста: {str(e)}")


if __name__ == "__main__":
    main()