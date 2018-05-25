from midi2audio import FluidSynth

# Installation of FluidSynth has to be done 

def listenMidi(path):
    FluidSynth().play_midi(path)
