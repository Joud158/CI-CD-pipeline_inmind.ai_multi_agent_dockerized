# Assignment 3

from __future__ import annotations
import argparse # command-line modes and flags
import csv # saving evaluation results
import os # env variables
from pathlib import Path # file handling
from statistics import mean # to compute summary metrics
from typing import Any, Dict, List 
from rag_core import DigitalAgronomistRAG

# list of 15 agronomy questions
TESTSET: List[Dict[str, Any]] = [
    {
        "question": "My zucchini leaves have white powdery spots. What is the likely problem and what should I do first?",
        "expected_source": "01_crop_diagnostics.txt",
        "reference_keywords": ["powdery mildew", "air movement", "remove heavily infected leaves", "monitor"],
    },
    {
        "question": "What field pattern usually suggests irrigation or equipment issues instead of disease?",
        "expected_source": "01_crop_diagnostics.txt",
        "reference_keywords": ["straight lines", "irrigation", "equipment"],
    },
    {
        "question": "What symptom pattern suggests early blight on tomato?",
        "expected_source": "01_crop_diagnostics.txt",
        "reference_keywords": ["target rings", "older leaves", "bottom"],
    },
    {
        "question": "What is the best irrigation style for sandy soil during hot periods?",
        "expected_source": "02_irrigation_water.txt",
        "reference_keywords": ["shorter", "more frequent", "daily"],
    },
    {
        "question": "During tomato flowering and fruit set, why should irrigation stay consistent?",
        "expected_source": "02_irrigation_water.txt",
        "reference_keywords": ["water stress", "fruit set", "consistent irrigation"],
    },
    {
        "question": "Give a simple sandy-soil drip irrigation recipe and what signs to watch.",
        "expected_source": "02_irrigation_water.txt",
        "reference_keywords": ["20-30 minutes", "2 cycles", "severe droop"],
    },
    {
        "question": "What does a sticky soil ribbon indicate when checking irrigation timing?",
        "expected_source": "02_irrigation_water.txt",
        "reference_keywords": ["too wet", "sticky ribbon"],
    },
    {
        "question": "Why should you start with a soil test before fertilizing?",
        "expected_source": "03_soil_nutrition.txt",
        "reference_keywords": ["prevents guessing", "pH", "EC", "N P K"],
    },
    {
        "question": "Why do split fertilizer applications help in low-organic-matter or sandy soils?",
        "expected_source": "03_soil_nutrition.txt",
        "reference_keywords": ["less leaching", "stable growth", "easier to correct"],
    },
    {
        "question": "What potassium-related warning sign is mentioned in the notes?",
        "expected_source": "03_soil_nutrition.txt",
        "reference_keywords": ["leaf edges scorch", "salinity", "potassium"],
    },
    {
        "question": "How should lettuce be handled when there is no cold chain?",
        "expected_source": "04_postharvest_supplychain.txt",
        "reference_keywords": ["shorten time to market", "wet burlap", "transport at night"],
    },
    {
        "question": "What are the three main post-harvest rules?",
        "expected_source": "04_postharvest_supplychain.txt",
        "reference_keywords": ["right maturity", "remove field heat", "protect from crushing"],
    },
    {
        "question": "What transport checklist items reduce post-harvest losses?",
        "expected_source": "04_postharvest_supplychain.txt",
        "reference_keywords": ["clean truck bed", "shade", "secure crates"],
    },
    {
        "question": "How do you decide whether buying a fungicide makes financial sense?",
        "expected_source": "05_farm_business_pricing.txt",
        "reference_keywords": ["yield at risk", "probability", "all-in cost", "expected saved revenue"],
    },
    {
        "question": "What are two practical habits for managing exchange-rate risk on the farm?",
        "expected_source": "05_farm_business_pricing.txt",
        "reference_keywords": ["records in both LBP and USD", "buy high-impact inputs early", "group purchasing"],
    },
]

# speech to text
def transcribe_audio(audio_path: str, model_name: str = "base") -> str:
    """
    Voice input via Whisper.
    For WAV files, load directly with scipy to avoid ffmpeg dependency.
    For other formats, Whisper may still require ffmpeg.
    """
    try:
        import whisper
        import numpy as np
        from scipy.io.wavfile import read as wav_read
    except ImportError as exc:
        raise RuntimeError(
            "Missing packages. Run:\n"
            "pip install openai-whisper scipy numpy"
        ) from exc
    # does the audio file exist
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    # load whisper model
    model = whisper.load_model(model_name)
    # if the file is .wav, it uses scipy.io.wavfile.read directly
    if audio_file.suffix.lower() == ".wav":
        sample_rate, audio = wav_read(str(audio_file))
        # Convert stereo to mono if needed
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        # Convert to float32 in [-1, 1] normalization
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                max_val = np.iinfo(audio.dtype).max
                audio = audio.astype(np.float32) / max_val
            else:
                audio = audio.astype(np.float32)
        if sample_rate != 16000:
            raise RuntimeError(
                f"WAV must be 16kHz for this fast path, but got {sample_rate} Hz."
            )
        result = model.transcribe(audio)
    else:
        # Non-WAV files may still need ffmpeg
        result = model.transcribe(str(audio_file))
    text = result.get("text", "").strip()
    if not text:
        raise RuntimeError("No speech was transcribed from the audio.")
    return text

# text to speech
def synthesize_speech(
    text: str,
    output_path: str = "voice_answer.wav",
    prefer_offline: bool = True,
) -> str:
    """
    Voice output:
    - offline path: pyttsx3 -> WAV
    - online fallback: gTTS -> MP3
    """
    if prefer_offline:
        try:
            import pyttsx3
            wav_path = str(Path(output_path).with_suffix(".wav"))
            engine = pyttsx3.init()
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
            return wav_path
        except Exception:
            pass
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise RuntimeError(
            "Both pyttsx3 and gTTS failed/unavailable. Install one of them:\n"
            "pip install pyttsx3\nor\npip install gTTS"
        ) from exc
    mp3_path = str(Path(output_path).with_suffix(".mp3"))
    tts = gTTS(text=text, lang="en")
    tts.save(mp3_path)
    return mp3_path

# playing the result
def open_audio_file(audio_path: str) -> None:
    """
    Open the generated audio file with the system default player when possible.
    """
    try:
        if os.name == "nt":
            os.startfile(audio_path)
        else:
            print(f"Audio saved to: {audio_path}")
    except Exception:
        print(f"Audio saved to: {audio_path}")

# mic capture
def record_from_microphone(
    seconds: int = 8,
    sample_rate: int = 16000,
    output_path: str = "mic_question.wav",
) -> str:
    """
    Record speech from the microphone into a WAV file.
    Requires:
      pip install sounddevice scipy
    """
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ImportError as exc:
        raise RuntimeError(
            "Microphone recording packages are missing. Run:\n"
            "pip install sounddevice scipy"
        ) from exc
    print(f"Speak now for {seconds} seconds...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    wav_path = str(Path(output_path).with_suffix(".wav"))
    write(wav_path, sample_rate, audio)
    print(f"Recorded microphone audio to: {wav_path}")
    return wav_path

# separation between system evaluation and demo interaction
def answer_without_llm(
    rag: DigitalAgronomistRAG,
    question: str,
    return_metadata: bool = False,
) -> Any:
    """
    Fast evaluation path: retrieval + fallback grounded answer only.
    This avoids making 15 external LLM calls during evaluation mode.
    """
    retrieved = rag.retrieve(question)
    answer = rag.fallback_answer_from_context(retrieved)
    if return_metadata:
        return {
            "question": question,
            "retrieved": retrieved,
            "answer": answer,
        }
    return answer

def voice_chat_from_file(
    audio_path: str,
    whisper_model: str = "base",
    auto_play: bool = False,
) -> Dict[str, str]:
    rag = DigitalAgronomistRAG().build()
    question = transcribe_audio(audio_path, model_name=whisper_model)
    answer = rag.answer(question)
    audio_answer_path = synthesize_speech(str(answer), "voice_answer")
    if auto_play:
        open_audio_file(audio_answer_path)
    return {
        "transcribed_question": question,
        "answer_text": str(answer),
        "audio_answer_path": audio_answer_path,
    }

def voice_chat_from_mic(
    seconds: int = 8,
    whisper_model: str = "base",
    auto_play: bool = False,
) -> Dict[str, str]:
    mic_audio_path = record_from_microphone(seconds=seconds, output_path="mic_question.wav")
    return voice_chat_from_file(
        audio_path=mic_audio_path,
        whisper_model=whisper_model,
        auto_play=auto_play,
    )

# how many of the expected reference keywords appear in the answer text and returns the fraction matched
def keyword_recall(answer_text: str, reference_keywords: List[str]) -> float:
    text = answer_text.lower()
    hits = sum(1 for kw in reference_keywords if kw.lower() in text)
    return hits / max(len(reference_keywords), 1)

def run_manual_eval(csv_out: str = "manual_eval_results.csv") -> List[Dict[str, Any]]:
    rag = DigitalAgronomistRAG().build()
    rows: List[Dict[str, Any]] = []
    print(f"Starting evaluation on {len(TESTSET)} questions...")
    for i, item in enumerate(TESTSET, start=1):
        print(f"Evaluating question {i}/{len(TESTSET)}")
        result = answer_without_llm(rag, item["question"], return_metadata=True)
        retrieved_sources = [r["source"] for r in result["retrieved"]]
        retrieved_ids = [r["chunk_id"] for r in result["retrieved"]]
        retrieval_hit = 1 if item["expected_source"] in retrieved_sources else 0
        citation_present = 1 if "[" in result["answer"] and "]" in result["answer"] else 0
        kw_recall = keyword_recall(result["answer"], item["reference_keywords"])
        top1_score = result["retrieved"][0]["score"] if result["retrieved"] else 0.0
        rows.append(
            {
                "question": item["question"],
                "expected_source": item["expected_source"],
                "retrieved_sources": " | ".join(retrieved_sources),
                "retrieved_chunk_ids": " | ".join(retrieved_ids),
                "retrieval_hit_at_k": retrieval_hit,
                "citation_present": citation_present,
                "keyword_recall_proxy": round(kw_recall, 3),
                "top1_similarity": round(top1_score, 4),
                "answer": result["answer"],
                "human_correctness_0_to_1": "",
                "human_groundedness_0_to_1": "",
                "human_helpfulness_0_to_1": "",
            }
        )
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows

# computes overall summary statistics
def summarize_eval(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "n_questions": len(rows),
        "retrieval_hit_rate_at_k": round(mean(r["retrieval_hit_at_k"] for r in rows), 3),
        "citation_presence_rate": round(mean(r["citation_present"] for r in rows), 3),
        "avg_keyword_recall_proxy": round(mean(r["keyword_recall_proxy"] for r in rows), 3),
        "avg_top1_similarity": round(mean(r["top1_similarity"] for r in rows), 3),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 3: voice I/O + manual evaluation")
    parser.add_argument(
        "--mode",
        choices=["eval", "mic", "file"],
        default="eval",
        help="eval = run 15-question evaluation, mic = speak into microphone, file = use existing audio file",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="",
        help="Path to an existing audio file when --mode file is used",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=8,
        help="Number of seconds to record from microphone in mic mode",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model name: tiny, base, small, medium, turbo",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Automatically open the generated answer audio file",
    )
    args = parser.parse_args()
    if args.mode == "eval":
        rows = run_manual_eval()
        metrics = summarize_eval(rows)
        print("\nEvaluation summary:")
        for k, v in metrics.items():
            print(f"- {k}: {v}")
        print("\nSaved detailed results to: manual_eval_results.csv")
        return
    if args.mode == "file":
        if not args.audio:
            raise ValueError("In file mode, provide --audio path_to_file.wav")
        result = voice_chat_from_file(
            audio_path=args.audio,
            whisper_model=args.whisper_model,
            auto_play=args.play,
        )
        print("\nTranscribed question:")
        print(result["transcribed_question"])
        print("\nAnswer:")
        print(result["answer_text"])
        print(f"\nAnswer audio file: {result['audio_answer_path']}")
        return
    if args.mode == "mic":
        result = voice_chat_from_mic(
            seconds=args.seconds,
            whisper_model=args.whisper_model,
            auto_play=args.play,
        )
        print("\nTranscribed question:")
        print(result["transcribed_question"])
        print("\nAnswer:")
        print(result["answer_text"])
        print(f"\nAnswer audio file: {result['audio_answer_path']}")
        return

if __name__ == "__main__":
    main()