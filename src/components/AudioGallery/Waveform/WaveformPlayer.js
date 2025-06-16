import { useRef, useState, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import "./WaveformPlayer.css";

import pauseicon from "../../Images/Waveform/pause.svg"
import audioIcon from "../../Images/Waveform/audio.svg"
import playicon from "../../Images/Waveform/play.svg"

const formWaveSurferOptions = (ref) => ({
  container: ref,
  waveColor: "#ccc",
  progressColor: "#0178ff",
  cursorColor: "transparent",
  responsive: true,
  height: 80,
  normalize: true,
  backend: "WebAudio",
  barWidth: 2,
  barGap: 3,
});

export default function WaveformPlayer({ audioFile, title, description }) {
	const waveformRef = useRef(null);
	const wavesurfer = useRef(null);
	const [volume, setVolume] = useState(0.5);
	const [playing, setPlaying] = useState(false);

	// Get Filename
	let filename = audioFile.split('/').pop(); 
	filename = filename.split('_')[0]; 
	
	console.log(audioFile)
	useEffect(() => {
		if (!waveformRef.current) return;

		const options = formWaveSurferOptions(waveformRef.current);
		wavesurfer.current = WaveSurfer.create(options);

		wavesurfer.current.on('ready', () => {
			console.log("WaveSurfer is ready and audio is loaded.");
		});
		
		wavesurfer.current.load(audioFile).catch((err) => {
		if (err.name === 'AbortError') {
			console.log("Audio loading was aborted.");
		} else {
			console.error("WaveSurfer load error:", err);
		}
		});

		// When audio finishes, reset the playing state to false
		wavesurfer.current.on('finish', () => {
			setPlaying(false);
		});

		return () => {
		if (wavesurfer.current) {
			wavesurfer.current.stop();
			setPlaying(false);  
			wavesurfer.current.destroy();
		}
		};
	}, [audioFile]);

	// Toggle playback of audio
	const handlePlayPause = () => {
		setPlaying(!playing);
		wavesurfer.current.playPause();
	};

	// Adjust audio volume
	const handleVolumeChange = (newVolume) => {
		setVolume(newVolume);
		wavesurfer.current.setVolume(newVolume);
	};


	return (
	<div className="audio-player">
		<div className="audio-header">
			<h3>{title}</h3>
			<p>{description}</p>
			<p>Filename : {filename}</p>
		</div>

		<div className="waveform" ref={waveformRef}></div>


		<div className="audio-icons">
			<button onClick={handlePlayPause}>
				<img
					src={playing ? pauseicon : playicon}
					alt={playing ? 'Pause' : 'Play'}
					className="icon-size"
				/>
			</button>
			
			<img className = "icon-size" src = {audioIcon} alt = "audio"></img>
			<input
				type='range'
				id='volume'
				name='volume'
				min='0'
				max='1'
				step='0.05'
				value={volume}
				onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
			/>
		</div>

	</div>
	);
}
