import {useState} from "react";
import axios from 'axios';
import './App.css'


const buttons = ['ә', 'җ', 'ң', 'ө', 'ү', 'һ']

function App() {
    const [text, setText] = useState('');
    const [loading, setLoading] = useState(false);
    const audioPlayer = document.getElementById('audioPlayer') as HTMLAudioElement;

    const convertText = async () => {

        if (!text) return;

        try {
            setLoading(true);
            const response = await axios.post('/tts', {
                text
            }, {
                responseType: 'arraybuffer'
            });

            const audioBlob = new Blob([response.data], {type: 'audio/mp3'});
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayer.src = audioUrl;
            audioPlayer.style.display = 'block';
            audioPlayer.play();
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to convert text to speech');
        } finally {
            setLoading(false);
        }
    }

    return (
        <>
            <h1 className='text-3xl'>Tatar Text-to-Speech</h1>
            <form
                className='mt-4'
                onSubmit={(event) => {
                    event.preventDefault()
                    convertText()
                }}>
                <textarea placeholder="Enter Tatar text here..." rows={6} autoFocus required value={text}
                          onChange={(e) => setText(e.target.value)}></textarea>
                <ul className='flex gap-3 mt-2 justify-center'>
                    {buttons.map((button, index) => (
                        <li key={index}>
                            <button type="button" onClick={() => setText(prev => prev + button)}>{button}</button>
                        </li>
                    ))}
                </ul>
                <button type="submit" disabled={loading} className='mt-8 !px-12 !py-4 mx-auto'>Convert to Speech
                </button>
            </form>

            <div className='mt-5'>
                <audio id="audioPlayer" controls className='hidden mx-auto'></audio>
            </div>
        </>
    )
}

export default App
