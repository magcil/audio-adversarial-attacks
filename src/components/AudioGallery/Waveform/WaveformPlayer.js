// WaveformPlayer.jsx
import { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import './WaveformPlayer.css';

const WaveformPlayer = ({ audioFile }) => {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (wavesurfer.current) {
      wavesurfer.current.destroy();
    }

    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#d9dcff',
      progressColor: '#4353ff',
      cursorColor: '#4353ff',
      barWidth: 3,
      barRadius: 3,
      responsive: true,
      height: 100,
      normalize: true,
      partialRender: true,
    });

    wavesurfer.current.load(audioFile);

    return () => {
      wavesurfer.current.destroy();
    };
  }, [audioFile]);

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
    wavesurfer.current.playPause();
  };

  return (
    <div className="waveform-container">
      <div ref={waveformRef} className="waveform" />
      <button onClick={togglePlayback} className="play-button">
        {isPlaying ? 'Pause' : 'Play'}
      </button>
    </div>
  );
};

export default WaveformPlayer;
