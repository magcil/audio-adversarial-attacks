import './App.css';
import Navbar from './components/Header/Navbar';
import Hero from './components/Hero/Hero'
import AudioGallery from './components/AudioGallery/AudioGallery/AudioGallery';
import About from './components/AboutSection/About';
import Footer from './components/Footer/Footer';

function App() {

  return (
    <div className="App">
        <Navbar></Navbar>
        <Hero></Hero>
        <AudioGallery></AudioGallery>
        <About></About>
        <Footer></Footer>
      
    </div>
  );
}

export default App;
