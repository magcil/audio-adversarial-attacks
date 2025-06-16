import { useState, useRef, useEffect } from 'react';
import "./DropdownMenu.css";

export default function DropdownMenu({ title = "Options", options = [], value, onChange,leadingIcon = null}) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const toggleDropdown = () => setIsOpen(prev => !prev);

  const handleOptionClick = (option) => {
    onChange(option);
    setIsOpen(false);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className={'dropdown'}>
      <div className="dropdown-button" onClick={toggleDropdown}>
        {leadingIcon && <span className="leading-icon">{leadingIcon}</span>}
        {value}
        <span className="arrow">▼</span>
      </div>
      {isOpen && (
        <div className="dropdown-content">
          <h4>{title}</h4>
          {options.map(option => (
            <div
              key={option}
              className={`option ${option === value ? 'selected' : ''}`}
              onClick={() => handleOptionClick(option)}
            >
              {option}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}