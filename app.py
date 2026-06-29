
import streamlit as st
import os
import asyncio
import tempfile
from pathlib import Path
import cv2
import numpy as np
from google import genai
import edge_tts
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    ColorClip
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Movie Trailer Generator",
    page_icon="🎬",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main {
        background-color: #0e0e0e;
        color: white;
    }
    .stApp {
        background-color: #0e0e0e;
    }
    h1, h2, h3 {
        color: #FFD700;
        font-family: 'Georgia', serif;
    }
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 18px;
        width: 100%;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #FFA500;
        color: black;
    }
    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #FFD700;
        border-radius: 8px;
    }
    .stSelectbox>div>div>select {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #FFD700;
    }
    .stFileUploader {
        background-color: #1e1e1e;
        border: 2px dashed #FFD700;
        border-radius: 10px;
        padding: 10px;
    }
    .success-box {
        background-color: #1a3a1a;
        border: 1px solid #00ff00;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #1a1a3a;
        border: 1px solid #FFD700;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; font-size: 48px;'>
        🎬 AI Movie Trailer Generator
    </h1>
    <p style='text-align: center; color: #aaaaaa; font-size: 18px;'>
        Powered by Gemini AI · Edge TTS · MoviePy
    </p>
    <hr style='border: 1px solid #FFD700;'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────
OUTPUT_DIR = Path("/content/trailer_output")
UPLOAD_DIR = Path("/content/trailer_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

VOICE_PATH   = OUTPUT_DIR / "trailer_voice.mp3"
TRAILER_PATH = OUTPUT_DIR / "cinematic_trailer.mp4"

# ─────────────────────────────────────────────
# SIDEBAR — API KEY + MOVIE DETAILS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here"
    )

    st.markdown("---")
    st.markdown("## 🎬 Movie Details")

    movie_title = st.text_input(
        "Movie Title",
        placeholder="e.g. The Last Kingdom"
    )

    genre = st.text_input(
        "Genre",
        placeholder="e.g. Action, Thriller, Romance"
    )

    main_character = st.text_input(
        "Main Character",
        placeholder="e.g. John Wick"
    )

    setting = st.text_input(
        "Setting / World",
        placeholder="e.g. Post-apocalyptic city"
    )

    tone = st.selectbox(
        "Tone",
        ["Epic", "Dark", "Romantic", "Action", "Thriller", "Mystery"]
    )

    st.markdown("---")
    st.markdown("## 🎥 Upload Scene Videos")
    st.markdown("Upload 3 video clips for your trailer")

    scene1 = st.file_uploader(
        "Scene 1 — Opening",
        type=["mp4", "mov", "webm", "avi"]
    )
    scene2 = st.file_uploader(
        "Scene 2 — Conflict",
        type=["mp4", "mov", "webm", "avi"]
    )
    scene3 = st.file_uploader(
        "Scene 3 — Climax",
        type=["mp4", "mov", "webm", "avi"]
    )

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def save_uploaded_video(uploaded_file, filename):
    path = UPLOAD_DIR / filename
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def zoom_in_effect(clip, zoom_ratio=0.03):
    def effect(get_frame, t):
        img   = get_frame(t)
        h, w  = img.shape[:2]
        scale = 1 + zoom_ratio * t
        new_w = int(w / scale)
        new_h = int(h / scale)
        x1    = (w - new_w) // 2
        y1    = (h - new_h) // 2
        cropped = img[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (w, h))
    return clip.fl(effect)

def zoom_out_effect(clip, zoom_ratio=0.03):
    def effect(get_frame, t):
        img   = get_frame(t)
        h, w  = img.shape[:2]
        scale = 1 + zoom_ratio * (clip.duration - t)
        new_w = int(w / scale)
        new_h = int(h / scale)
        x1    = (w - new_w) // 2
        y1    = (h - new_h) // 2
        cropped = img[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (w, h))
    return clip.fl(effect)

def cinematic_bars(clip, bar_height_ratio=0.08):
    w, h       = clip.size
    bar_height = int(h * bar_height_ratio)
    top_bar    = ColorClip(
        size=(w, bar_height),
        color=(0, 0, 0)
    ).set_duration(clip.duration).set_position(("center", "top"))
    bottom_bar = ColorClip(
        size=(w, bar_height),
        color=(0, 0, 0)
    ).set_duration(clip.duration).set_position(("center", "bottom"))
    return CompositeVideoClip([clip, top_bar, bottom_bar])

def color_grade(clip, brightness=1.05, contrast=1.1, saturation=0.9):
    def grade(get_frame, t):
        img  = get_frame(t).astype(np.float32)
        img  = img * brightness
        img  = (img - 128) * contrast + 128
        img  = np.clip(img, 0, 255).astype(np.uint8)
        hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv  = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return clip.fl(grade)

async def generate_voice_async(text, voice, output_path):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="-5%",
        volume="+10%"
    )
    await communicate.save(str(output_path))

def generate_voice(text, voice, output_path):
    asyncio.run(generate_voice_async(text, voice, output_path))

def build_trailer(scene_paths, audio_path, output_path,
                  target_resolution=(1280, 720), fps=24):
    audio          = AudioFileClip(str(audio_path))
    total_duration = audio.duration
    scene_duration = total_duration / len(scene_paths)

    effects = [
        {"zoom": "in",  "fade_in": 1.0, "fade_out": 0.8},
        {"zoom": "out", "fade_in": 0.8, "fade_out": 0.8},
        {"zoom": "in",  "fade_in": 0.8, "fade_out": 1.5},
    ]

    processed_clips = []

    for i, (path, fx) in enumerate(zip(scene_paths, effects), 1):
        raw_clip = VideoFileClip(str(path))

        if raw_clip.duration < scene_duration:
            loops    = int(scene_duration / raw_clip.duration) + 1
            raw_clip = concatenate_videoclips([raw_clip] * loops)

        clip = raw_clip.subclip(0, scene_duration)
        clip = clip.resize(target_resolution)
        clip = clip.without_audio()
        clip = color_grade(clip)

        if fx["zoom"] == "in":
            clip = zoom_in_effect(clip)
        else:
            clip = zoom_out_effect(clip)

        clip = cinematic_bars(clip)
        clip = clip.fadein(fx["fade_in"]).fadeout(fx["fade_out"])
        processed_clips.append(clip)

    final_video = concatenate_videoclips(
        processed_clips,
        method="compose"
    )
    final_video = final_video.set_audio(audio)
    final_video.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        bitrate="5000k",
        preset="medium",
        verbose=False,
        logger=None
    )

    audio.close()
    for c in processed_clips:
        c.close()

# ─────────────────────────────────────────────
# MAIN GENERATE BUTTON
# ─────────────────────────────────────────────
st.markdown("## 🚀 Generate Your Trailer")

generate_clicked = st.button("🎬 Generate Trailer Now")

if generate_clicked:

    # ── Validation ───────────────────────────
    errors = []

    if not api_key:
        errors.append("❌ Please enter your Gemini API key in the sidebar")
    if not movie_title:
        errors.append("❌ Please enter a movie title")
    if not genre:
        errors.append("❌ Please enter a genre")
    if not scene1:
        errors.append("❌ Please upload Scene 1 video")
    if not scene2:
        errors.append("❌ Please upload Scene 2 video")
    if not scene3:
        errors.append("❌ Please upload Scene 3 video")

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    # ── Connect to Gemini ─────────────────────
    try:
        client = genai.Client(api_key=api_key)
        GEMINI_MODEL = "gemini-2.5-flash"
        st.success("✅ Connected to Gemini!")
    except Exception as e:
        st.error(f"❌ Gemini connection failed: {e}")
        st.stop()

    # ── Save uploaded videos ──────────────────
    with st.spinner("💾 Saving uploaded videos..."):
        scene_paths = []
        for i, scene in enumerate([scene1, scene2, scene3], 1):
            ext  = scene.name.split(".")[-1]
            path = save_uploaded_video(scene, f"scene_{i}.{ext}")
            scene_paths.append(path)
        st.success("✅ Videos saved!")

    # ── Generate Story ────────────────────────
    with st.spinner("📖 Generating movie story..."):
        story_prompt = f"""
        You are a Hollywood screenwriter.
        Create a gripping cinematic movie story.

        Movie Title    : {movie_title}
        Genre          : {genre}
        Main Character : {main_character or 'the protagonist'}
        Setting        : {setting or 'a mysterious world'}
        Tone           : {tone}

        Requirements:
        - Emotionally engaging and visually rich
        - Clear beginning, conflict, hint at resolution
        - Maximum 180 words
        - Present tense
        """
        story_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=story_prompt
        )
        movie_story = story_response.text

    st.markdown("### 📖 Movie Story")
    st.markdown(f"""
        <div class='info-box'>
        {movie_story}
        </div>
    """, unsafe_allow_html=True)

    # ── Generate Trailer Script ───────────────
    with st.spinner("🎬 Generating trailer script..."):
        trailer_prompt = f"""
        You are a professional movie trailer editor.
        Generate a powerful 3-scene movie trailer script.

        Movie Story: {movie_story}
        Tone: {tone}

        Format:
        SCENE 1:
        VISUAL: ...
        NARRATION: ...
        MOOD: ...

        SCENE 2:
        VISUAL: ...
        NARRATION: ...
        MOOD: ...

        SCENE 3:
        VISUAL: ...
        NARRATION: ...
        MOOD: ...

        BACKGROUND MUSIC: ...
        TITLE CARD: ...
        """
        trailer_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=trailer_prompt
        )
        trailer_script = trailer_response.text

    st.markdown("### 🎬 Trailer Script")
    st.markdown(f"""
        <div class='info-box'>
        {trailer_script}
        </div>
    """, unsafe_allow_html=True)

    # ── Extract Narration ─────────────────────
    with st.spinner("🎙️ Extracting narration..."):
        narration_prompt = f"""
        Extract ONLY the narration lines from this trailer script.
        Make it smooth and flowing.
        Tone: {tone}

        Trailer: {trailer_script}
        """
        narration_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=narration_prompt
        )
        narration_text = narration_response.text

    st.markdown("### 🎙️ Voice-Over Narration")
    st.markdown(f"""
        <div class='info-box'>
        {narration_text}
        </div>
    """, unsafe_allow_html=True)

    # ── Generate Voice-Over ───────────────────
    VOICE_OPTIONS = {
        "epic"     : "en-US-GuyNeural",
        "dark"     : "en-US-GuyNeural",
        "romantic" : "en-US-JennyNeural",
        "action"   : "en-US-DavisNeural",
        "thriller" : "en-US-TonyNeural",
        "mystery"  : "en-US-TonyNeural",
        "default"  : "en-US-GuyNeural"
    }

    selected_voice = VOICE_OPTIONS.get(
        tone.lower(),
        VOICE_OPTIONS["default"]
    )

    with st.spinner("🔊 Generating voice-over..."):
        generate_voice(narration_text, selected_voice, VOICE_PATH)
        st.success("✅ Voice-over generated!")

    st.markdown("### 🔊 Voice-Over Preview")
    st.audio(str(VOICE_PATH))

    # ── Build Trailer ─────────────────────────
    with st.spinner("🎬 Building your cinematic trailer... This may take a few minutes..."):
        try:
            build_trailer(
                scene_paths=scene_paths,
                audio_path=VOICE_PATH,
                output_path=TRAILER_PATH
            )
            st.success("✅ Trailer built successfully!")
        except Exception as e:
            st.error(f"❌ Trailer build failed: {e}")
            st.stop()

    # ── Show Final Trailer ────────────────────
    st.markdown("---")
    st.markdown("""
        <h2 style='text-align:center; color:#FFD700;'>
            🎬 Your Cinematic Trailer Is Ready!
        </h2>
    """, unsafe_allow_html=True)

    st.video(str(TRAILER_PATH))

    # ── Download Button ───────────────────────
    with open(str(TRAILER_PATH), "rb") as f:
        st.download_button(
            label="⬇️ Download Your Trailer",
            data=f,
            file_name=f"{movie_title}_trailer.mp4",
            mime="video/mp4"
        )

    st.balloons()
