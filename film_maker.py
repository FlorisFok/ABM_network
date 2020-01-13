import cv2
import os

def make_video(fileloc_filename, name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{name}.mp4v', fourcc, 1.0, (1200, 900))

    for i in range(0, 14000, 10):
        img_path = f"{fileloc_filename}{i}.png"
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


def make_video2(fileloc, project_name, fps, frame_name=None):
    import cv2
    import os

    img_array = []
    for i, filename in enumerate(os.listdir(fileloc)):
        if frame_name == None:
            img = cv2.imread(os.path.join(fileloc, filename))
        else:
            img = cv2.imread(os.path.join(fileloc, frame_name+str(i)+'.png'))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(f'{project_name}.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)) :
        out.write(img_array[i])
    out.release()

if __name__ == "__main__":
    name = "test_evo"
    fileloc_filename = r"C:\Users\FlorisFok\Documents\Master\Evo Pro\video\frame"
    make_video(fileloc_filename, name)