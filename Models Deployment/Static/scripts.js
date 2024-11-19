document.addEventListener('DOMContentLoaded', function() {
    // Set initial state on page load (neutral position and default color)
    document.documentElement.style.setProperty('--slider-bg-color', '#1B4A99'); // Default slider color (blue)
    
    let circle = document.querySelector('.circle');
    let slider = document.querySelector('.solider');

    // Get the result value from the HTML element
    let resultValue = document.getElementById("result").innerText.trim();

    // Add a smooth transition to the slider's background and the circle's movement
    circle.style.transition = 'transform 0.5s ease';
    slider.style.transition = 'background-color 0.5s ease';

    // Check if there is a valid result
    if (resultValue === "N/A" || resultValue === "" || resultValue === null) {
        // Keep the initial state (no changes)
        console.log("No result available. Keeping the initial state.");
        // Center the circle for the neutral state
        circle.style.transform = 'translateX(0)'; // Reset circle to center
        circle.classList.remove('negative', 'positive'); // Remove any existing classes
    } else {
        // Reset classes before applying new styles
        circle.classList.remove('negative', 'positive');

        // Check if the result is Negative or Positive
        if (resultValue === "Negative") {
            // Change the CSS variable for slider background to red and move the circle left
            document.documentElement.style.setProperty('--slider-bg-color', 'red');
            circle.style.transform = 'translateX(-37.5px)';
            circle.classList.add('negative'); // Add negative class
        } else if (resultValue === "Positive") {
            // Change the CSS variable for slider background to green and move the circle right
            document.documentElement.style.setProperty('--slider-bg-color', 'green');
            circle.style.transform = 'translateX(37.5px)';
            circle.classList.add('positive'); // Add positive class
        } else {
            console.log("Invalid result value.");
            // Reset circle to center for any other state
            circle.style.transform = 'translateX(0)'; // Reset circle to center
            circle.classList.remove('negative', 'positive'); // Remove any existing classes
        }
    }
});
