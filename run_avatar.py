#!/usr/bin/env python
"""
Скрипт создает говорящего аватара, читающего текст из файла.
Также может извлекать текст из видеофайлов.
Использование: просто запустите python run_avatar.py
"""

import os
import sys
import subprocess
import argparse
from gtts import gTTS
from src.video_processor import VideoProcessor

# Конфигурация (постоянные пути)
TEXT_FILE = 'sample_text.txt'    # Файл с текстом
FACE_IMAGE = 'Wav2Lip/inputs/img.png'  # Изображение лица
OUTPUT_VIDEO = 'Wav2Lip/outputs/result.mp4'  # Выходное видео
LANGUAGE = 'en'  # Язык текста (ru - русский, en - английский)

# Пути к аудиофайлам
MP3_FILE = 'Wav2Lip/inputs/speech.mp3'
WAV_FILE = 'Wav2Lip/inputs/speech.wav'

def text_to_speech(text, output_file, lang='en'):
    """Преобразует текст в речь и сохраняет в MP3"""
    print(f"Создание аудио из текста...")
    tts = gTTS(text=text, lang=lang)
    tts.save(output_file)
    print(f"Аудио создано: {output_file}")

def convert_to_wav(mp3_file, wav_file):
    """Конвертирует MP3 в WAV с параметрами для Wav2Lip"""
    print(f"Конвертирование MP3 в WAV...")
    cmd = f"ffmpeg -y -i {mp3_file} -acodec pcm_s16le -ar 16000 -ac 1 {wav_file}"
    subprocess.call(cmd, shell=True)
    print(f"WAV файл создан: {wav_file}")

def create_talking_avatar(face_img, audio_file, output_file):
    """Создает видео с говорящим аватаром"""
    cmd = f"python modified_inference.py --face {face_img} --audio {audio_file} --outfile {output_file}"
    subprocess.call(cmd, shell=True)
    print(f"Видео создано: {output_file}")

def main():
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description='Создание говорящего аватара и извлечение текста из видео')
    parser.add_argument('--extract', action='store_true', help='Извлечь текст из видео в директории Wav2Lip/inputs/audios')
    parser.add_argument('--avatar', action='store_true', help='Создать аватара, читающего текст из файла sample_text.txt')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'ru'], help='Язык текста (en - английский, ru - русский)')
    args = parser.parse_args()
    
    # Устанавливаем язык из аргументов
    global LANGUAGE
    LANGUAGE = args.language
    
    # Словарь соответствия языков для разных библиотек
    language_map = {
        'en': 'en-US',  # Для Google Speech API и Vosk
        'ru': 'ru-RU'   # Для Google Speech API и Vosk
    }
    
    # Если не указаны флаги, включаем оба режима по умолчанию
    if not args.extract and not args.avatar:
        args.extract = True
        args.avatar = True
    
    # Создаем необходимые директории
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    os.makedirs('Wav2Lip/inputs', exist_ok=True)
    os.makedirs('Wav2Lip/inputs/audios', exist_ok=True)
    os.makedirs('Wav2Lip/temp', exist_ok=True)
    
    # Режим извлечения текста из видео
    if args.extract:
        processor = VideoProcessor(language=language_map[LANGUAGE])
        processor.run()
    
    # Режим создания аватара
    if args.avatar:
        print("=== СОЗДАНИЕ ГОВОРЯЩЕГО АВАТАРА ===")
        
        # Проверяем наличие текстового файла
        if not os.path.isfile(TEXT_FILE):
            print(f"Ошибка: Файл {TEXT_FILE} не найден")
            print("Создаю пример текстового файла...")
            with open(TEXT_FILE, 'w', encoding='utf-8') as f:
                if LANGUAGE == 'en':
                    f.write("Hello! I'm a talking avatar that reads text from a file.")
                else:
                    f.write("Привет! Я говорящий аватар, который читает текст из файла.")
            print(f"Файл {TEXT_FILE} успешно создан. Пожалуйста, отредактируйте его и запустите скрипт снова.")
            return
        
        # Проверяем наличие изображения
        if not os.path.isfile(FACE_IMAGE):
            print(f"Ошибка: Изображение {FACE_IMAGE} не найдено")
            print("Убедитесь, что в папке Wav2Lip/inputs есть изображение img.png")
            return
        
        # Чтение текста из файла
        try:
            with open(TEXT_FILE, 'r', encoding='utf-8') as file:
                text = file.read()
                if not text.strip():
                    print("Ошибка: Текстовый файл пуст")
                    return
                print(f"Прочитан текст длиной {len(text)} символов:")
                print(f"«{text[:100]}...»" if len(text) > 100 else f"«{text}»")
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            return
        
        # Полный процесс создания видео
        try:
            # Шаг 1: Создание аудио из текста
            text_to_speech(text, MP3_FILE, LANGUAGE)
            
            # Шаг 2: Конвертирование MP3 в WAV
            convert_to_wav(MP3_FILE, WAV_FILE)
            
            # Шаг 3: Создание видео
            create_talking_avatar(FACE_IMAGE, WAV_FILE, OUTPUT_VIDEO)
            
            print(f"\nГотово! Видео сохранено в {OUTPUT_VIDEO}")
            print(f"Для просмотра откройте файл в видеоплеере")
            
        except Exception as e:
            print(f"Ошибка при создании видео: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()