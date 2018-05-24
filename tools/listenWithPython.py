from midi2audio import FluidSynth

# Installation of FluidSynth has to be normalOrder

def listenMidi(path):
    FluidSynth().play_midi(path)
