import cv2,numpy,os,re
import PIL.Image as IMG
import datetime







recoginer = cv2.face_LBPHFaceRecognizer.create()
pgm_path = r'F:\Test_Data\BioID-FaceDatabase-V1'
yml_path = r'F:\Test_Data\TrainerData\\'
face_detecter = cv2.CascadeClassifier(
            'D:\openCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
yml_list = []

def calander():
    timestamps = datetime.datetime.now()
    calendar1 = timestamps[0:4] + '_' + timestamps[5:7] + timestamps[7:10]

    return calendar1


def read_yml(yml_num):
    # count = 0
    for yml in os.listdir(yml_path):

        yml_list.append(yml)
        yml_name = 'TrainerData' + yml_num + '.yml'
        recoginer.read(os.path.join(yml_path,yml_name))
        print(yml_name)
        # count += 1
        # if count > 0:
        #    continue
        # yml_name = yml.split('.')[0]
        # ids = re.search('\d+',yml_name)
        # # ids1 = int(ids.group())
        # print(ids.group())
        # return ids.group()
        return

def compare_pgm(path):

    for f in os.listdir(path):
        if f.endswith('.pgm'):
            yml_num = re.search('\d+',f.split('.')[0]).group()
            read_yml(yml_num)
            imgPaths = os.path.join(path, f)
            img = cv2.imread(imgPaths)
            grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_detecter.detectMultiScale(grey_img,scaleFactor=1.1,minNeighbors=2,minSize=(53,53))   # .detectMulttiscale 方法返回四个坐标值
            print('f:',f)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(50, 50, 200), thickness=2)  # 调用同时返回对象
                id,confidence = recoginer.predict(grey_img[x:x+w,y:y+h])
                print('id',id+1,'置信评分',confidence)
                # print(faces)

            cv2.imshow('Result',img)
            if ord('q') == cv2.waitKey(0):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    # read_yml()
    compare_pgm(pgm_path)