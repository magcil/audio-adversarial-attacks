import { useState, useEffect } from 'react';
import "./AudioGallery.css";
import WaveformPlayer from "../Waveform/WaveformPlayer";
import DropdownMenu from "../../DropdownMenu/DropdownMenu";

import filter1 from "../../Images/Filters/filter1.svg"
import filter2 from "../../Images/Filters/filter2.svg"
import filter3 from "../../Images/Filters/filter3.svg"

// Initialize filters data
const models = ['BEATs', 'PaSST', 'AST'];
const SNRs = [5, 10, 15, 20, 25, 30];
const Audioset_Classes = [ "Sounds of things", "Animal", "Music", "Human sounds", "Source-ambiguous sounds"
    ,"Natural sounds","Channel, environment and background"];
const ESC_Classes = ["Animals", "Natural soundscapes & water sounds", "Human, non-speech sounds", "Interior domestic sounds", "Exterior urban noises"]
const datasets = ['AudioSet', 'ESC-50'];    


const AudioGallery = () => {
    const [metadata, setMetadata] = useState([]);
    const [selectedModel, setselectedModel] = useState('BEATs');
    const [selectedSNR, setselectedSNR] = useState(20);
    const [Class, setClass] = useState('Music');
    const [dataset, setDataset] = useState('AudioSet');

    // Hook to reinitialize the dataset filters.
    useEffect(() => {
        if (dataset === "AudioSet") {
            setClass(Audioset_Classes[0]);
        } else {
            setClass(ESC_Classes[0]);
        }
    }, [dataset]);


    useEffect(() => {
        fetch(`${process.env.PUBLIC_URL}/wavs_metadata.json`)
        .then(res => res.json())
        .then(data => setMetadata(data))
        .catch(err => console.error("Failed to load metadata", err));
    }, []);
    

    let filepath =  `audio/${dataset}/${selectedModel}/${Class}/SNR_${selectedSNR}`;
    let match = metadata.find(item => item.path === filepath);
    let filename = match?.filename || null;
    let predictedClass = match?.class || null;

    console.log(filepath)


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
                                options={dataset === "AudioSet" ? Audioset_Classes : ESC_Classes}
                                value={Class}
                                onChange={setClass}
                                leadingIcon={<img src={filter3} alt="Class Filter icon" />}>
                            </DropdownMenu>

                        </div>
                    </div>
                </div>
            </div>
            <div className="waveforms-div">
                <WaveformPlayer audioFile={process.env.PUBLIC_URL + `/audio/${dataset}/${selectedModel}/${Class}/Original/original.wav`} filename = {filename} title = "Original Audio" description= "Clean audio sample with no adversarial perturbations"></WaveformPlayer>
                <WaveformPlayer audioFile={process.env.PUBLIC_URL + `/audio/${dataset}/${selectedModel}/${Class}/SNR_${selectedSNR}/adversary.wav`}  filename = {filename} title = "Adversarial Example" description= "Adversarial Example using PSO test"  predictedClass={predictedClass}></WaveformPlayer>
            </div>
            
        </div>
  );
};

export default AudioGallery;
