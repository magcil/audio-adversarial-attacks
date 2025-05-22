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
        <li><a href="#about" onClick={() => setIsOpen(false)}>About</a></li>
        <li><a href="#skills" onClick={() => setIsOpen(false)}>Skills</a></li>
        <li><a href="#projects" onClick={() => setIsOpen(false)}>Projects</a></li>
        <li><a href="#contact" onClick={() => setIsOpen(false)}>Contact</a></li>
      </ul>
                  
      <div className="hamburger" onClick={toggleMenu}>
          <i className="fas fa-bars"></i>
      </div>

    </nav>
  );
};

export default Navbar;
