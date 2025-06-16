import { useState } from 'react';
import "./AudioGallery.css";
import WaveformPlayer from "../Waveform/WaveformPlayer";
import DropdownMenu from "../../DropdownMenu/DropdownMenu";

import filter1 from "../../Images/Filters/filter1.svg"
import filter2 from "../../Images/Filters/filter2.svg"
import filter3 from "../../Images/Filters/filter3.svg"

// Initialize filters data
const models = ['BEATs', 'PaSST', 'AST'];
const SNRs = [5, 10, 15, 20, 25, 30];
const Classes = [ "Sounds of things", "Animal", "Music", "Human sounds", "Source-ambiguous sounds"
    ,"Natural sounds","Channel, environment and background"];

const datasets = ['AusioSet', 'ESC-50'];    

const AudioGallery = () => {
    const [selectedModel, setselectedModel] = useState('BEATs');
    const [selectedSNR, setselectedSNR] = useState(20);
    const [Class, setClass] = useState('Music');
    const [dataset, setDataset] = useState('AudioSet');

    return (
        <div className="audio-section">
            <hr></hr>
            <div className="audio-intro">
                <div>
                    <h3>Audio Examples</h3>
                    <p>Listen to various adversarial audio examples across <br/>different models and SNR values.</p>
                </div>

                <div className="filters">
                    <div  className='filters-container'>
                        <div className='filters-wrap'>
                            <span className="filters-tag">Model</span>
                            <DropdownMenu
                                title="Available Models"
                                options={models}
                                value={selectedModel}
                                onChange={setselectedModel}
                                leadingIcon={<img src={filter1} alt="Model Filter icon" />}>
                            </DropdownMenu>
                        </div>
                        <div className='filters-wrap'>
                            <span className="filters-tag"> Dataset </span>
                            <DropdownMenu
                                title="Datasets"
                                options={datasets}
                                value={dataset}
                                onChange={setDataset}
                                leadingIcon={<img src={filter3} alt="Class Filter icon" />}>
                            </DropdownMenu>
                        </div>
                        
                    </div>
                     <div className='filters-container'>
                        <div className='filters-wrap'>
                            <span className="filters-tag">SNR </span>
                            <DropdownMenu
                                title="SNR Values"
                                options={SNRs}
                                value={selectedSNR}
                                onChange={setselectedSNR}
                                leadingIcon={<img src={filter2} alt="SNR Filter icon" />}>
                            </DropdownMenu>
                        </div>
                        <div className='filters-wrap'>
                            <span className="filters-tag"> Class </span>
                            <DropdownMenu
                                title="Classes"
                                options={Classes}
                                value={Class}
                                onChange={setClass}
                                leadingIcon={<img src={filter3} alt="Class Filter icon" />}>
                            </DropdownMenu>

                        </div>
                    </div>
                </div>
            </div>
            <div className="waveforms-div">
                <WaveformPlayer audioFile={process.env.PUBLIC_URL + `/audio/${selectedModel}/${Class}/SNR_${selectedSNR}/adversary.wav`} title = "Original Audio" description= "Clean audio sample with no adversarial perturbations"></WaveformPlayer>
                <WaveformPlayer audioFile={process.env.PUBLIC_URL + `/audio/${selectedModel}/${Class}/Original/original.wav`}  title = "Adversarial Example" description= "Adversarial Example using PSO recognized as: Cat"></WaveformPlayer>
            </div>
        </div>
  );
};

export default AudioGallery;
