import React, { useRef, useState, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import "./AudioPlayer.css";

export default function AudioPlayer() {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // useEffect(() => {
  //   const wavesurfer = WaveSurfer.create({
  //     container: "#waveform",
  //     waveColor: "#a0a0a0",
  //     progressColor: "#2684ff",
  //     height: 80,
  //     responsive: true,
  //     barWidth: 2,
  //   });

  //   wavesurfer.load("/your-audio-file.mp3");

  //   return () => wavesurfer.destroy();
  // }, []);

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (audio.paused) {
      audio.play();
      setIsPlaying(true);
    } else {
      audio.pause();
      setIsPlaying(false);
    }
  };

  return (
    <div className="audio-player">
      <div className="audio-header">
        <h3>Original Audio</h3>
        <p>Clean audio sample with no adversarial perturbations</p>
      </div>

      <audio ref={audioRef} src="/your-audio-file.mp3"></audio>

      <div className="waveform" id="waveform"></div>

      <div className="controls">
        <button className="play-button" onClick={togglePlayback}>
          {isPlaying ? (
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path d="M8 5v14l11-7z" fill="currentColor" />
            </svg>
          )}
        </button>

        <div className="timeline">
          <div className="progress"></div>
        </div>

        <div className="volume-control">
          <svg viewBox="0 0 24 24" width="18" height="18">
            <path d="M5 9v6h4l5 5V4l-5 5H5z" fill="currentColor" />
          </svg>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            className="volume-slider"
            onChange={(e) => (audioRef.current.volume = e.target.value)}
          />
        </div>
      </div>
    </div>
  );
}
