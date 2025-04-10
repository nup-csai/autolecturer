#!/usr/bin/env python
"""
Модуль для обработки видеофайлов и их конвертации в текст.
"""

import os
import re
import subprocess
from typing import List, Optional
from src.audio_transcriber import AudioTranscriber
from pytube import YouTube

class VideoProcessor:
    """Класс для обработки видеофайлов и извлечения из них текста."""
    
    def __init__(
        self, 
        input_dir: str = "Wav2Lip/inputs/audios", 
        output_text_file: str = "sample_text.txt",
        language: str = "en-US",
        youtube_url: str = None
    ):
        """
        Инициализация класса VideoProcessor.
        
        Args:
            input_dir: Директория с входными видеофайлами
            output_text_file: Файл для сохранения распознанного текста
            language: Код языка для распознавания речи
            youtube_url: URL видео с YouTube для скачивания
        """
        self.input_dir = input_dir
        self.output_text_file = output_text_file
        self.language = language
        self.youtube_url = youtube_url
        self.transcriber = AudioTranscriber(video_dir=input_dir)
        
        # Создаем директорию, если она не существует
        os.makedirs(input_dir, exist_ok=True)
        
        # Скачиваем видео с YouTube, если указан URL
        if self.youtube_url:
            self.download_youtube_video()
    
    def get_video_files(self) -> List[str]:
        """
        Получает список видеофайлов из директории input_dir.
        
        Returns:
            Список имен видеофайлов
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if not os.path.exists(self.input_dir):
            return []
            
        return [
            f for f in os.listdir(self.input_dir) 
            if os.path.isfile(os.path.join(self.input_dir, f)) and 
            any(f.lower().endswith(ext) for ext in video_extensions)
        ]
    
    def process_latest_video(self) -> Optional[str]:
        """
        Обрабатывает самый последний (по времени изменения) видеофайл из директории.
        
        Returns:
            Текст, распознанный из видео, или None, если видеофайлов нет
        """
        video_files = self.get_video_files()
        if not video_files:
            print(f"В директории {self.input_dir} не найдено видеофайлов")
            return None
            
        # Найдем самый свежий файл
        latest_video = max(
            video_files, 
            key=lambda f: os.path.getmtime(os.path.join(self.input_dir, f))
        )
        
        print(f"Обработка последнего видеофайла: {latest_video}")
        text = self.transcriber.process_video(latest_video, self.language)
        
        # Если получили ошибку распознавания речи, сохраняем заглушку
        if text.startswith("Ошибка") or text.startswith("Не удалось"):
            print(f"Ошибка распознавания: {text}")
            print("Сохраняем стандартный текст вместо распознанного")
            text = "Failed to recognize speech from video. This is default text for testing the avatar."
        
        return text
    
    def save_text_to_file(self, text: str) -> None:
        """
        Сохраняет распознанный текст в файл.
        
        Args:
            text: Текст для сохранения
        """
        with open(self.output_text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Текст сохранен в файл: {self.output_text_file}")
    
    def download_youtube_video(self) -> str:
        """
        Скачивает видео с YouTube по указанному URL.
        
        Returns:
            Имя файла скачанного видео или None при ошибке
        """
        try:
            # Проверка формата URL
            if not self.youtube_url or not re.search(r'(youtube\.com|youtu\.be)', self.youtube_url):
                print(f"Ошибка: Неверный формат URL YouTube: {self.youtube_url}")
                return None
                
            print(f"Загрузка видео с YouTube: {self.youtube_url}")
            
            try:
                # Альтернативный способ скачивания видео с YouTube через subprocess и yt-dlp
                print(f"Пробуем скачать видео через yt-dlp: {self.youtube_url}")
                # Проверим, установлен ли yt-dlp
                try:
                    subprocess.run(["yt-dlp", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("yt-dlp не установлен. Устанавливаем...")
                    subprocess.run(["pip", "install", "yt-dlp"], check=True)
                
                # Скачиваем видео с помощью yt-dlp, принудительно перезаписывая
                output_file = os.path.join(self.input_dir, "lecture.mp4")
                # Удаляем существующий файл, если такой есть
                if os.path.exists(output_file):
                    os.remove(output_file)
                    print(f"Удален существующий файл: {output_file}")
                subprocess.run(["yt-dlp", "--force-overwrites", "-f", "best[ext=mp4]", "-o", output_file, self.youtube_url], check=True)
                print(f"Видео успешно загружено через yt-dlp: {output_file}")
                return "lecture.mp4"
            except Exception as e:
                print(f"Ошибка при использовании yt-dlp: {str(e)}")
                print("Пробуем использовать pytube...")
                
                # Если yt-dlp не сработал, пробуем стандартный способ с pytube
                yt = YouTube(
                    self.youtube_url,
                    use_oauth=False,
                    allow_oauth_cache=False
                )
            
            # Выбираем поток с видео (высокое качество, но не слишком большой размер)
            video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not video_stream:
                print("Ошибка: Не удалось найти подходящий видеопоток")
                return None
                
            # Скачиваем видео в указанную директорию
            output_file = os.path.join(self.input_dir, "lecture.mp4")
            video_stream.download(output_path=self.input_dir, filename="lecture.mp4")
            
            print(f"Видео успешно загружено: {output_file}")
            return "lecture.mp4"
            
        except Exception as e:
            print(f"Ошибка при загрузке видео с YouTube: {str(e)}")
            return None
            
    def run(self) -> bool:
        """
        Запускает процесс обработки видео и сохранения текста.
        
        Returns:
            True, если текст успешно извлечен и сохранен, иначе False
        """
        print("=== ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ ВИДЕО ===")
        
        # Если был указан URL YouTube, но загрузка не удалась, выходим
        if self.youtube_url and not os.path.exists(os.path.join(self.input_dir, "lecture.mp4")):
            print("Ошибка: Не удалось загрузить видео с YouTube")
            return False
            
        text = self.process_latest_video()
        
        if text is None:
            return False
            
        print(f"Текст для сохранения: {text[:100]}..." if len(text) > 100 else f"Текст для сохранения: {text}")
        self.save_text_to_file(text)
        return True