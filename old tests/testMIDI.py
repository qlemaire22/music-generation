import midi
song = midi.read_midifile("/Users/quentin/Computer/DataSet/Music DataSet/MIDI/lmd_full/0/0a0a2b0e4d3b7bf4c5383ba025c4683e.mid")
#print song
print(song[1])
print(song[2][:10])
print(song[1][0])
datas = song[1][0].data
#
# song = song[:2]
# print(len(song))
#
# for i in range(len(song)):
#     print(song[i][0])
#
#
# print(song[1])

# print(len(song))
#
#
# for i in range(len(song)):
#     n = len(song[i])
#     deleted = 0
#     j = 0
#     while j<n-1:
#         try:
#             keep = True
#             l = song[i][j-deleted].data
#             for k in range(len(l)):
#                 if not(song[i][j-deleted].data[k] in datas):
#                     keep = False
#             if not(keep):
#                 song[i].remove(song[i][j-deleted])
#                 deleted +=1
#
#
#         except:
#             None
#         j+=1
#
#
midi.write_midifile("test.midi", song)
