import "./AudioGallery.css";
// import AudioPlayer from "../AudioPlayer/AudioPlayer";
import WaveformPlayer from "../Waveform/WaveformPlayer";

const AudioGallery = () => {
 
    return (
        <div className="audio-section">
            {/* <AudioPlayer></AudioPlayer> */}
            <WaveformPlayer audioUrl="/audio/1.wav"></WaveformPlayer>
        </div>
  );
};

export default AudioGallery;
