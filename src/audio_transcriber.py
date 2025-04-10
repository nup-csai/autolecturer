#!/usr/bin/env python
"""
Модуль для извлечения аудио из видео и преобразования аудио в текст.
"""

import os
import subprocess
import json
import wave
from typing import Optional
import time
import requests
from tqdm import tqdm

# Для облачного распознавания
import speech_recognition as sr

# Для локального распознавания
from vosk import Model, KaldiRecognizer, SetLogLevel

class AudioTranscriber:
    """Класс для извлечения аудио из видео и преобразования аудио в текст."""
    
    def __init__(
        self, 
        video_dir: str, 
        temp_audio_path: str = "temp_audio.wav",
        use_local_model: bool = True,
        model_path: str = "models/vosk-model-small-en-us"
    ):
        """
        Инициализация класса AudioTranscriber.
        
        Args:
            video_dir: Директория, где находятся видеофайлы
            temp_audio_path: Путь для временного аудиофайла
            use_local_model: Использовать локальную модель (Vosk) вместо облачной (Google)
            model_path: Путь к модели Vosk
        """
        self.video_dir = video_dir
        self.temp_audio_path = temp_audio_path
        self.use_local_model = use_local_model
        self.model_path = model_path
        
        # Инициализация распознавателя Google (на случай, если локальная модель не доступна)
        self.recognizer = sr.Recognizer()
        
        # Отключаем вывод отладочной информации Vosk
        SetLogLevel(-1)
    
    def extract_audio_from_video(self, video_path: str, output_audio_path: Optional[str] = None) -> str:
        """
        Извлекает аудио из видеофайла.
        
        Args:
            video_path: Путь к видеофайлу
            output_audio_path: Путь для сохранения аудио. Если None, используется self.temp_audio_path
            
        Returns:
            Путь к извлеченному аудиофайлу
        """
        if output_audio_path is None:
            output_audio_path = self.temp_audio_path
            
        # Используем ffmpeg для извлечения аудио
        cmd = f"ffmpeg -y -i {video_path} -acodec pcm_s16le -ar 16000 -ac 1 {output_audio_path}"
        subprocess.call(cmd, shell=True)
        
        return output_audio_path
    
    def transcribe_with_google(self, audio_path: str, language: str = "en-US") -> str:
        """
        Преобразует аудиофайл в текст с помощью Google Speech Recognition.
        
        Args:
            audio_path: Путь к аудиофайлу
            language: Код языка (по умолчанию английский)
            
        Returns:
            Распознанный текст
        """
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                
                # Добавляем обработку ошибок и повторные попытки
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        text = self.recognizer.recognize_google(audio_data, language=language)
                        return text
                    except sr.RequestError as e:
                        if attempt < max_attempts - 1:
                            print(f"Ошибка при попытке {attempt+1}, повторяем через 3 секунды...")
                            time.sleep(3)
                        else:
                            return f"Ошибка сервиса распознавания речи: {e}"
                    except sr.UnknownValueError:
                        return "Не удалось распознать речь в аудио"
        except Exception as e:
            return f"Ошибка при обработке аудио: {e}"
    
    def download_vosk_model(self) -> bool:
        """
        Загружает модель Vosk для английского языка из интернета.
        
        Returns:
            True, если модель установлена успешно, иначе False
        """
        if os.path.exists(self.model_path):
            return True
            
        try:
            # URL для малой английской модели Vosk
            model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            zip_path = os.path.join(os.path.dirname(self.model_path), "model.zip")
            
            # Создаем директорию для модели
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            print(f"Скачиваем модель из {model_url}...")
            
            # Скачиваем модель
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(zip_path, 'wb') as f:
                for data in tqdm(response.iter_content(block_size), 
                                total=total_size//block_size, 
                                unit='KB', 
                                unit_scale=True):
                    f.write(data)
            
            print("Распаковываем модель...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.model_path))
            
            # Переименовываем директорию (содержит версию в имени)
            extracted_dir = [d for d in os.listdir(os.path.dirname(self.model_path)) 
                             if d.startswith("vosk-model-small-en-us") and os.path.isdir(os.path.join(os.path.dirname(self.model_path), d))][0]
            
            if extracted_dir != os.path.basename(self.model_path):
                os.rename(
                    os.path.join(os.path.dirname(self.model_path), extracted_dir),
                    self.model_path
                )
            
            # Удаляем архив
            os.remove(zip_path)
            
            print(f"Модель загружена в {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return False
    
    def transcribe_with_vosk(self, audio_path: str) -> str:
        """
        Преобразует аудиофайл в текст с помощью Vosk (локальная модель).
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Распознанный текст
        """
        if not os.path.exists(self.model_path):
            print("Локальная модель распознавания не найдена.")
            model_downloaded = self.download_vosk_model()
            if not model_downloaded:
                return "Ошибка: не удалось загрузить модель распознавания речи"
        
        try:
            # Загружаем модель
            model = Model(self.model_path)
            
            # Открываем аудиофайл
            wf = wave.open(audio_path, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                return "Ошибка: аудиофайл должен быть в формате WAV mono PCM"
            
            # Создаем распознаватель
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            
            # Читаем аудиофайл и распознаем
            result_text = ""
            
            # Обрабатываем аудио по блокам
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                    
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    if 'text' in part_result:
                        result_text += part_result['text'] + " "
            
            # Получаем финальный результат
            final_result = json.loads(rec.FinalResult())
            if 'text' in final_result:
                result_text += final_result['text']
            
            return result_text.strip()
            
        except Exception as e:
            return f"Ошибка при распознавании с Vosk: {e}"
    
    def transcribe_audio(self, audio_path: str, language: str = "en-US") -> str:
        """
        Преобразует аудиофайл в текст, используя выбранный метод.
        
        Args:
            audio_path: Путь к аудиофайлу
            language: Код языка (по умолчанию английский)
            
        Returns:
            Распознанный текст
        """
        if self.use_local_model:
            # Используем локальную модель Vosk
            text = self.transcribe_with_vosk(audio_path)
            if text.startswith("Ошибка"):
                print("Ошибка при использовании локальной модели, пробуем облачную модель...")
                return self.transcribe_with_google(audio_path, language)
            return text
        else:
            # Используем облачную модель Google
            return self.transcribe_with_google(audio_path, language)
    
    def process_video(self, video_filename: str, language: str = "en-US") -> str:
        """
        Обрабатывает видеофайл: извлекает аудио и преобразует в текст.
        
        Args:
            video_filename: Имя видеофайла в директории video_dir
            language: Код языка для распознавания
            
        Returns:
            Распознанный текст из видео
        """
        video_path = os.path.join(self.video_dir, video_filename)
        if not os.path.exists(video_path):
            return f"Ошибка: Видеофайл {video_path} не найден"
        
        # Извлекаем аудио
        print(f"Извлекаем аудио из видео {video_filename}...")
        audio_path = self.extract_audio_from_video(video_path)
        
        # Распознаем текст
        print(f"Распознаем речь из аудио...")
        text = self.transcribe_audio(audio_path, language)
        
        # Удаляем временный аудиофайл после успешного распознавания
        if os.path.exists(audio_path) and text and not text.startswith("Ошибка"):
            os.remove(audio_path)
            
        return text