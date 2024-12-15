document.getElementById('uploadButton').addEventListener('click', async () => {
  const fileInput = document.getElementById('imageUpload');
  const file = fileInput.files[0];

  if (!file) {
      alert('No file selected. Please choose a file to upload.');
      console.error('No file selected.');
      return;
  }

  // Create form data to send the file
  const formData = new FormData();
  formData.append('file', file);

  try {
      // Send file to backend
      const response = await fetch('/classify', {
          method: 'POST',
          body: formData,
      });

      if (!response.ok) {
          const errorMessage = await response.text();
          throw new Error(`Failed to upload image: ${errorMessage}`);
      }

      // Get the prediction results from the backend
      const data = await response.json();
      console.log('Predictions:', data.results);

      // Update taskbars with the accuracy of predictions for all 10 classes
      updateTaskbars(data.results);

  } catch (error) {
      console.error('Error uploading file:', error);
      alert(`Error: ${error.message}`);
  }
});

function updateTaskbars(predictions) {
  const taskbarContainer = document.getElementById('taskbarContainer');
  taskbarContainer.innerHTML = ''; // Clear previous taskbars

  // Loop through the predictions and create a taskbar for each one
  predictions.forEach(prediction => {
      const taskbar = document.createElement('div');
      taskbar.classList.add('taskbar');
      
      const label = document.createElement('span');
      label.textContent = `${prediction.label}: ${prediction.probability.toFixed(2)}%`;
      
      const fill = document.createElement('div');
      fill.classList.add('fill');
      fill.style.width = `${prediction.probability}%`;

      taskbar.appendChild(label);
      taskbar.appendChild(fill);
      taskbarContainer.appendChild(taskbar);
  });
}
