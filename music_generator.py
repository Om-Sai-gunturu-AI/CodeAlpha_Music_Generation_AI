import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.utils import np_utils
import glob
import os

# Load MIDI files
def parse_midi_files(data_path):
    notes = []
    for file in glob.glob(data_path + "/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts:  # File has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Prepare sequences for the LSTM model
def prepare_sequences(notes, sequence_length):
    pitch_names = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitch_names))
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output, pitch_names

# Build LSTM model
def build_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

if __name__ == "__main__":
    # Load data
    data_path = "data"
    sequence_length = 100
    notes = parse_midi_files(data_path)
    network_input, network_output, pitch_names = prepare_sequences(notes, sequence_length)

    # Build and train model
    model = build_model(network_input.shape[1:], len(pitch_names))
    model.fit(network_input, network_output, epochs=50, batch_size=64)

    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/music_generator.h5")
    print("Model saved to models/music_generator.h5")
