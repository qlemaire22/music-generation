import pickle
import numpy
from music21 import instrument, note, stream, chord
import network
import data
import argparse
import config

def generate():
    """ Generate a piano midi file """
    normalized_input, network_output, n_vocab, pitchnames, network_input = data.prepare_sequences()

    net = network.Network(n_vocab)
    model = net.model
    model.load_weights('results/run1/weights-1.9942.hdf5')
    prediction_output = generate_notes(model, list(network_input), list(pitchnames), n_vocab)
    create_midi(prediction_output)

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = list(network_input[start])
    prediction_output = []

    # generate 500 notes
    for note_index in range(config.GENERATION_LENGTH):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--number', default=1,
                        help="int, number of music you want to generate.", type=int)

    args = parser.parse_args()

    for i in range(args.number):
        generate()
