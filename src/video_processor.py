#!/usr/bin/env python
"""
Модуль для обработки видеофайлов и их конвертации в текст.
"""

import os
from typing import List, Optional
from src.audio_transcriber import AudioTranscriber

class VideoProcessor:
    """Класс для обработки видеофайлов и извлечения из них текста."""
    
    def __init__(
        self, 
        input_dir: str = "Wav2Lip/inputs/audios", 
        output_text_file: str = "sample_text.txt",
        language: str = "en-US"
    ):
        """
        Инициализация класса VideoProcessor.
        
        Args:
            input_dir: Директория с входными видеофайлами
            output_text_file: Файл для сохранения распознанного текста
            language: Код языка для распознавания речи
        """
        self.input_dir = input_dir
        self.output_text_file = output_text_file
        self.language = language
        self.transcriber = AudioTranscriber(video_dir=input_dir)
        
        # Создаем директорию, если она не существует
        os.makedirs(input_dir, exist_ok=True)
    
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
    
    def run(self) -> bool:
        """
        Запускает процесс обработки видео и сохранения текста.
        
        Returns:
            True, если текст успешно извлечен и сохранен, иначе False
        """
        print("=== ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ ВИДЕО ===")
        text = self.process_latest_video()
        
        if text is None:
            return False
            
        print(f"Текст для сохранения: {text[:100]}..." if len(text) > 100 else f"Текст для сохранения: {text}")
        self.save_text_to_file(text)
        return True