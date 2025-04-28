async function convertText() {
    const button = document.querySelector('button');
    const text = document.getElementById('textInput').value;

    if (!text) return;

    button.disabled = true; // Disable the button

    try {
        const response = await fetch('/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                speaker: "dilyara",
                sample_rate: 48000,
                put_accent: true
            }),
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = audioUrl;
        audioPlayer.style.display = 'block';
        audioPlayer.play();
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to convert text to speech');
    } finally {
        button.disabled = false; // Re-enable the button
    }
}