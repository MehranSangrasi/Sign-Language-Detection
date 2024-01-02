<h1>YOLO (You Only Look Once) American Sign Language Detection Project</h1>

<p>This repository implements American Sign Lnaguage detection system using the YOLO (You Only Look Once) model. Below is an overview of the project's structure and usage.</p>

<h2>Contributors</h2>
<ul>
    <li>Abdullah Saim</li>
    <li>Mehran.</li>
    <li>M. Danish Azeem</li>
</ul>

<h2>Files and Usage</h2>

<ol>
    <li><strong>best.pt</strong>
    <ul>
        <li>The pre-trained YOLO model weights (<code>best.pt</code>).</li>
    </ul>
</li>
    <li><strong>TestingYOLO.py</strong>
    <ul>
        <li>Script demonstrating the initial working of the YOLO model in real-time using a webcam.</li>
        <li>Loads pre-trained weights from <code>best.pt</code> for sign language detection.</li>
    </ul>
    <p><strong>Usage:</strong></p>
    <pre><code>python TestingYOLO.py</code></pre>
</li>
    <li><strong>final.py</strong>
    <ul>
        <li>Contains the final implementation of the YOLO object detection system.</li>
        <li>Can be customized for specific use cases or integrated into larger projects.</li>
    </ul>
    <p><strong>Usage:</strong></p>
    <pre><code>python final.py</code></pre>
    </li>
    <li><strong>Sign_Language_Detection_using_YOLOv8.ipynb</strong>
    <ul>
        <li>Jupyter notebook showcasing the training and testing process on the American Sign Language (ASL) dataset.</li>
        <li>Provides code, visualizations, and explanations for better understanding.</li>
    </ul>
    <p><strong>Usage:</strong></p>
    <p>Open the Jupyter notebook in a Jupyter environment and follow the instructions within the notebook.</p>
    </li>
</ol>

<h2>Dependencies</h2>

<p>Ensure you have the following dependencies installed before running the scripts:</p>
<ul>
    <li>PyTorch</li>
    <li>OpenCV</li>
    <li>NumPy</li>
    <li>Ultralytics</li>
    <li>Inference</li>
</ul>

<p>Install the dependencies using:</p>
<pre><code>pip install torch opencv-python numpy ultralytics inference</code></pre>

<h2>Note</h2>

<ul>
    <li>Check the code comments and documentation for implementation details and customization options.</li>
    <li>Feel free to reach out for questions or issues related to the project.</li>
</ul>
