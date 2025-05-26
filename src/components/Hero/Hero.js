import "./Hero.css";


const Hero = () => {
 
    return (
        <div className="hero-section">
            <div className="hero-info">
                <span className="tag">Research Project</span>
                <h1>Audio Adversarial Attacks</h1>
                <p>Exporing vulnerabilities in sound event classification systems through the generation of   
                <br/> imperceptible adversarial examples</p>
                <div className="info-boxes">
                    <div className="info-box">
                        <h3>Objective</h3>
                        <p>
                        To demonstrate how minor perturbations to audio can cause misclassification in machine learning models.
                        </p>
                    </div>
                    <div className="info-box">
                        <h3>Methodology</h3>
                        <p>
                        Using gradient-based optimization to create imperceptible adversarial perturbations in audio samples.
                        </p>
                    </div>
                    <div className="info-box">
                        <h3>Impact</h3>
                        <p>
                        Highlighting security vulnerabilities in sound classification systems used for security applications.
                        </p>
                    </div>
                </div>
            </div>
            <hr></hr>
        </div>
  );
};

export default Hero;
