import mediapipe as mp
# print all available face mesh connections
for name in dir(mp.solutions.face_mesh):
    if name.startswith("FACEMESH_"):
        print(name)
