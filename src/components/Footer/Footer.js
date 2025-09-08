import "./Footer.css";


const Footer = () => {
 
    return (
        <div className="footer-wrapper">
            <hr></hr>
            <div className="footer-section">
                <p>Adversarial Audio Research • 2025 </p>
                <div className="footer-refs">
                    <a  href="https://github.com/magcil/audio-adversarial-attacks" target="_blank" rel="noopener noreferrer" >Github</a>
                    <a>Paper</a>
                    {/* <a  href="" target="_blank" rel="noopener noreferrer" > Contact</a> */}
                </div>
            </div>
        </div>
  );
};

export default Footer;
