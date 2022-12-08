import cv2
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, model_selection=0, detectionconf=0.5):
        self.model_selection = model_selection
        self.detectionconf = detectionconf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.model_selection, self.detectionconf)

    def findface(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        if self.results.detections:
            for detection in self.results.detections:
                if draw:
                    self.mpDraw.draw_detection(img, detection, self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

        return img

    def pointface(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceelmt=[]
 
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection)
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                facepoint = detection.location_data.relative_keypoints
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                for i in range(0,6):
                    faceelmt.append((int(facepoint[i].x * iw),int(facepoint[i].y * ih)))
                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1]-20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    for i in range(0,6):
                        cv2.putText(img, f'{i+1}', faceelmt[i], cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return faceelmt

           
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = faceDetector()
    while True:
        success, img = cap.read()
        img = detector.findface(img)
        points = detector.pointface(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()