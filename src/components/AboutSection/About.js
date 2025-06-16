import "./About.css";


const About = () => {
 
    return (
        <div className="about-section">
            <h3>About this Research</h3>
            <p>This research project demonstrates how commersial deep learning models used for audio processing can be vulnerable to carefully 
            crafted adversarial attacks. These attacks introduce subtle, often imperceptible perturbations to audio signals that can mislead the
            model into making incorrect predictions—posing potential security risks.  Despite their high reported accuracies, such models often 
            lack robustness, making them susceptible to attacks even without access to their internal architecture.
            <br/>
            <br/>

            The samples above, illustrate the impact of adversarial attacks on various state-of-the-art audio models. Although the adversarial audio 
            sounds nearly identical to the original to the human ear, it can cause the models to produce drastically different outputs. You can explore 
            how different signal-to-noise ratio (SNR) levels affect model predictions and observe how even minimal perturbations can alter outcomes.
            </p>
        </div>
  );
};

export default About;
