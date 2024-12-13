import numpy as np
from music21 import instrument, note, chord, stream
from keras.models import load_model

def generate_notes(model, network_input, pitch_names, n_vocab, length=500):
    int_to_note = {number: note for number, note in enumerate(pitch_names)}
    pattern = network_input[0]  # Use the first pattern
    prediction_output = []

    for _ in range(length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern[1:], index)

    return prediction_output

def create_midi(prediction_output, output_path="output/generated_music.mid"):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_path)
    print(f"Generated music saved to {output_path}")

if __name__ == "__main__":
    model = load_model("models/music_generator.h5")
    network_input = np.load("data/network_input.npy")  # Load input used for training
    pitch_names = np.load("data/pitch_names.npy")
    n_vocab = len(pitch_names)

    prediction_output = generate_notes(model, network_input, pitch_names, n_vocab)
    create_midi(prediction_output)
