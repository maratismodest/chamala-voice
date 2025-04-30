import {useState} from "react";
import axios from 'axios';
// import {tatarLetters} from "./shared/constants";
import {SubmitHandler, useForm} from "react-hook-form";
import {Spinner} from "./shared/ui/Spinner.tsx";

type Inputs = {
    text: string
}

function App() {
    const {
        register,
        handleSubmit,
        // setValue,
        // control,
    } = useForm<Inputs>()
    // const text = useWatch({control, name: 'text'})
    const onSubmit: SubmitHandler<Inputs> = async (data) => {
        const {text} = data;
        if (!text) return;
        await convertText(text)
    }
    const [loading, setLoading] = useState(false);
    const audioPlayer = document.getElementById('audioPlayer') as HTMLAudioElement;

    const convertText = async (text: string) => {
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
            <header>
                <h1 className='text-3xl'>Tatar Text-to-Speech</h1>
            </header>
            <main>
                <form
                    className='mt-4'
                    onSubmit={handleSubmit(onSubmit)}>
                <textarea
                    placeholder="Enter Tatar text here..."
                    rows={6}
                    autoFocus
                    required
                    {...register("text")}
                />
                    {/*<ul className='flex gap-3 mt-2 justify-center'>*/}
                    {/*    {tatarLetters.map((button, index) => (*/}
                    {/*        <li key={index}>*/}
                    {/*            <button type="button" onClick={() => setValue('text', text + button)}>{button}</button>*/}
                    {/*        </li>*/}
                    {/*    ))}*/}
                    {/*</ul>*/}
                    <div className='h-8 flex items-center justify-center'>
                        {loading && <Spinner/>}
                    </div>
                    <button type="submit" disabled={loading}
                            className='mt-2 !px-12 !py-4 mx-auto'>

                        Convert to Speech
                    </button>
                </form>

                <div className='mt-5'>
                    <audio id="audioPlayer" controls className='hidden mx-auto'></audio>
                </div>
            </main>
            <footer className='mt-auto'>@2025 Marat Faizerakhmanov</footer>
        </>
    )
}

export default App
