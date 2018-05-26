from midi2audio import FluidSynth

# Installation of FluidSynth has to be done

def listenToMidi(path):
    FluidSynth().play_midi(path)
