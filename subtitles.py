import os
import re
from typing import List, Tuple
import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.VideoClip import TextClip, VideoClip


def text_to_subtitles(text_file: str, words_per_second: float = 2.5) -> List[Tuple[Tuple[float, float], str]]:
    """Convert text file to subtitles with timing"""
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    subtitles = []
    current_time = 0.0

    for sentence in sentences:
        duration = max(1.5, len(sentence.split()) / words_per_second)
        subtitles.append(((current_time, current_time + duration), sentence))
        current_time += duration

    return subtitles


def create_subtitle_clip(subtitles: List[Tuple[Tuple[float, float], str]],
                         video_width: int = 1920) -> SubtitlesClip:
    """Create subtitle clip with styling"""

    def make_textclip(txt):
        clip = TextClip(txt, font='Arial', fontsize=24, color='white',
                        align='center', method='caption', size=(video_width, None))

        bg = clip.on_color(size=(video_width, clip.h + 20),
                           color=(0, 0, 0), pos='center', col_opacity=0.5)

        return bg.set_position(('center', 1))

    return SubtitlesClip(subtitles, make_textclip)


def add_subtitles_to_video(video_path: str, text_file: str, output_path: str) -> str:
    """Add subtitles from text file to video"""
    video = mp.VideoFileClip(video_path)
    subtitles = text_to_subtitles(text_file)

    subtitle_clip = create_subtitle_clip(subtitles, video.w)
    final_video = mp.CompositeVideoClip([video, subtitle_clip])

    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    video.close()

    return output_path


if __name__ == "__main__":
    video_path = "video.mp4"
    text_file = "text_to_speak.txt"
    output_path = "output_with_subtitles.mp4"

    output_path = add_subtitles_to_video(video_path, text_file, output_path)
    print(f"Done! Output video saved to: {output_path}")