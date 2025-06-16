import "./Navbar.css";
import { useState } from 'react';
import mylogo from "../Images/Logo/Logo.png"

const Navbar = () => {

    const [isOpen, setIsOpen] = useState(false);

    const toggleMenu = () => {
      setIsOpen(prev => !prev);
    };

    return (
    <nav className="navbar">
      <a className="navbar_logo" href="/"><img src={mylogo} alt="Logo" /></a>
      <ul className={`navbar_menu ${isOpen ? "active" : ""}`}>
        <li><a href="#home" onClick={() => setIsOpen(false)}>Home</a></li>
        <li><a href="#contact" onClick={() => setIsOpen(false)}>Project Overview</a></li>
        <li><a href="https://github.com/magcil/audio-adversarial-attacks" target="_blank" rel="noopener noreferrer" onClick={() => setIsOpen(false)}>Github</a></li>
        {/* <li><a href="https://github.com/magcil/audio-adversarial-attacks" target="_blank" rel="noopener noreferrer" onClick={() => setIsOpen(false)}>Paper</a></li> */}
      </ul>
                  
      <div className="hamburger" onClick={toggleMenu}>
          <i className="fas fa-bars"></i>
      </div>

    </nav>
  );
};

export default Navbar;
