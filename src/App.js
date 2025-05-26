import './App.css';
import Navbar from './components/Header/Navbar';
import Hero from './components/Hero/Hero'

import AudioGallery from './components/AudioGallery/AudioGallery/AudioGallery';
function App() {
  return (
    <div className="App">
        <Navbar></Navbar>
        <Hero></Hero>
        <AudioGallery></AudioGallery>
    </div>
  );
}

export default App;
